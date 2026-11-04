//! Slow, explicit FP32 model oracle. This is a bring-up backend, not a serving path.
use crate::{
    gguf::{ModelIndex, type_layout},
    model::{Config, ple_rows},
    mtp::{Backend, Step},
    quant::decode_row,
};
use anyhow::{Context, Result, ensure};
use memmap2::Mmap;
use std::{collections::BTreeMap, fs::File, sync::Arc};

#[derive(Clone)]
enum Data {
    Owned(Arc<Vec<f32>>),
    Mapped {
        map: Arc<Mmap>,
        offset: usize,
        kind: u32,
    },
}
#[derive(Clone)]
pub struct Tensor {
    pub dims: Vec<usize>,
    data: Data,
}
impl Tensor {
    pub fn owned(dims: Vec<usize>, values: Vec<f32>) -> Result<Self> {
        ensure!(
            dims.iter().product::<usize>() == values.len(),
            "weight shape mismatch"
        );
        Ok(Self {
            dims,
            data: Data::Owned(Arc::new(values)),
        })
    }
    pub fn row(&self, row: usize) -> Result<Vec<f32>> {
        let width = self.dims[0];
        let rows = self.dims.iter().skip(1).product::<usize>();
        ensure!(row < rows, "weight row out of range");
        match &self.data {
            Data::Owned(v) => Ok(v[row * width..(row + 1) * width].to_vec()),
            Data::Mapped { map, offset, kind } => {
                let (block, bytes) = type_layout(*kind)?;
                let rb = width / (block as usize) * bytes as usize;
                let start = offset + row * rb;
                let mut y = vec![0.; width];
                decode_row(*kind, &map[start..start + rb], &mut y)?;
                Ok(y)
            }
        }
    }
    fn all(&self) -> Result<Vec<f32>> {
        let mut a = Vec::new();
        for r in 0..self.dims.iter().skip(1).product() {
            a.extend(self.row(r)?);
        }
        Ok(a)
    }
    fn mm(&self, x: &[f32], expert: usize) -> Result<Vec<f32>> {
        ensure!(x.len() == self.dims[0], "matvec input width mismatch");
        let n = self.dims.get(1).copied().unwrap_or(1);
        let e = self.dims.get(2).copied().unwrap_or(1);
        ensure!(expert < e, "expert out of range");
        (0..n)
            .map(|j| Ok(dot(&self.row(expert * n + j)?, x)))
            .collect()
    }
}

#[derive(Clone, Default)]
pub struct Weights(pub BTreeMap<String, Tensor>);
impl Weights {
    /// The caller must not modify/truncate mapped model files while this owner lives.
    pub fn open(index: &ModelIndex) -> Result<Self> {
        let mut w = Self::default();
        for s in &index.shards {
            let f = File::open(&s.path)?;
            ensure!(
                f.metadata()?.len() == s.header.file_bytes,
                "model file size changed"
            );
            // Files are read-only model artifacts, retained by each mapped tensor owner.
            let map = Arc::new(unsafe { Mmap::map(&f)? });
            for t in &s.header.tensors {
                w.0.insert(
                    t.name.clone(),
                    Tensor {
                        dims: t.dimensions.iter().map(|&x| x as usize).collect(),
                        data: Data::Mapped {
                            map: map.clone(),
                            offset: (s.header.data_offset + t.offset) as usize,
                            kind: t.kind,
                        },
                    },
                );
            }
        }
        Ok(w)
    }
    fn get(&self, n: &str) -> Result<&Tensor> {
        self.0.get(n).with_context(|| format!("missing weight {n}"))
    }
    fn v(&self, n: &str) -> Result<Vec<f32>> {
        self.get(n)?.all()
    }
    fn mm(&self, n: &str, x: &[f32]) -> Result<Vec<f32>> {
        self.get(n)?.mm(x, 0)
    }
    fn has(&self, n: &str) -> bool {
        self.0.contains_key(n)
    }
    /// Import only the shared sidecar's block. Embedding/head owners stay those
    /// of the target; GGUF norm scales are already converted (do not add 1).
    pub fn attach_shared_mtp(&mut self, index: &ModelIndex, target: &Config) -> Result<()> {
        crate::binding::validate_shared_mtp(index, target)?;
        let mtp = Self::open(index)?;
        ensure!(
            mtp.0.keys().all(|n| !self.0.contains_key(n)),
            "MTP would replace target tensors"
        );
        self.get("token_embd.weight")?;
        self.get("output.weight")?;
        self.0.extend(mtp.0);
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GdnState {
    pub matrix: Vec<f32>,
    pub history: Vec<Vec<f32>>,
}
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AttentionState {
    pub keys: Vec<Vec<f32>>,
    pub values: Vec<Vec<f32>>,
    pub raw_tail: Vec<Vec<f32>>,
    pub pooled: Vec<Vec<f32>>,
}
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    pub position: usize,
    pub tokens: Vec<u32>,
    pub gdn: Vec<GdnState>,
    pub attention: Vec<AttentionState>,
    pub ple_history: Vec<Vec<Vec<f32>>>,
}
impl State {
    pub fn new(c: &Config) -> Self {
        Self {
            position: 0,
            tokens: Vec::new(),
            gdn: (0..c.layers + 1)
                .map(|_| GdnState {
                    matrix: vec![0.; c.value_heads * c.gdn_dim * c.gdn_dim],
                    history: vec![vec![0.; c.conv_channels()]; c.conv_kernel - 1],
                })
                .collect(),
            attention: vec![AttentionState::default(); c.layers + 1],
            ple_history: vec![Vec::new(); c.layers + 1],
        }
    }
}

pub struct Reference {
    pub config: Config,
    pub weights: Weights,
    pub max_context: usize,
}
impl Reference {
    pub fn new(config: Config, weights: Weights, max_context: usize) -> Result<Self> {
        config.validate()?;
        ensure!(max_context > 0, "zero context");
        Ok(Self {
            config,
            weights,
            max_context,
        })
    }
    fn name(il: usize, s: &str) -> String {
        format!("blk.{il}.{s}")
    }
    fn mm(&self, il: usize, s: &str, x: &[f32]) -> Result<Vec<f32>> {
        self.weights.mm(&Self::name(il, s), x)
    }
    fn v(&self, il: usize, s: &str) -> Result<Vec<f32>> {
        self.weights.v(&Self::name(il, s))
    }
    fn pair(
        &self,
        il: usize,
        merged: &str,
        a: &str,
        b: &str,
        x: &[f32],
        split: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if self.weights.has(&Self::name(il, merged)) {
            let mut y = self.mm(il, merged, x)?;
            ensure!(split < y.len(), "invalid merged projection");
            let z = y.split_off(split);
            Ok((y, z))
        } else {
            Ok((self.mm(il, a, x)?, self.mm(il, b, x)?))
        }
    }
    fn hc_mix(&self, r: &[f32], prefix: &str, inject: bool) -> Result<(Vec<f32>, Vec<f32>)> {
        let c = &self.config;
        let gamma = self.weights.v(&format!("{prefix}_norm.weight"))?;
        let xn = group_norm(r, &gamma, c.width, c.rms_epsilon)?;
        let (mut low, mut injection) =
            if inject && self.weights.has(&format!("{prefix}_down_inject.weight")) {
                let mut a = self
                    .weights
                    .mm(&format!("{prefix}_down_inject.weight"), &xn)?;
                ensure!(a.len() == c.hc_rank + c.hc, "bad HC merged shape");
                let b = a.split_off(c.hc_rank);
                (a, b)
            } else {
                (
                    self.weights.mm(&format!("{prefix}_down.weight"), &xn)?,
                    if inject {
                        self.weights.mm(&format!("{prefix}_inject.weight"), &xn)?
                    } else {
                        Vec::new()
                    },
                )
            };
        for x in &mut low {
            *x = silu(*x / c.hc as f32);
        }
        for x in &mut injection {
            *x /= c.hc as f32;
        }
        let gates = self.weights.mm(&format!("{prefix}_up.weight"), &low)?;
        ensure!(gates.len() == r.len(), "HC gate width mismatch");
        let mut mixed = vec![0.; c.width];
        for s in 0..c.hc {
            for j in 0..c.width {
                mixed[j] += xn[s * c.width + j] * sigmoid(gates[s * c.width + j]) / c.hc as f32;
            }
        }
        Ok((mixed, injection))
    }
    fn head(&self, r: &[f32], mtp: bool) -> Result<Vec<f32>> {
        let prefix = if mtp {
            format!("blk.{}.nextn.hc_head", self.config.layers)
        } else {
            "output_hc".into()
        };
        let (x, _) = self.hc_mix(r, &prefix, false)?;
        let name = if self.weights.has("output.weight") {
            "output.weight"
        } else {
            "token_embd.weight"
        };
        self.weights.mm(name, &x)
    }
    fn gdn(&self, il: usize, x: &[f32], state: &mut GdnState) -> Result<Vec<f32>> {
        let c = &self.config;
        let s = c.gdn_dim;
        let qk = c.key_heads * s;
        let (raw, z) = self.pair(
            il,
            "attn_qkv_gate.weight",
            "attn_qkv.weight",
            "attn_gate.weight",
            x,
            c.conv_channels(),
        )?;
        ensure!(
            raw.len() == c.conv_channels() && z.len() == c.value_heads * s,
            "GDN projection shape mismatch"
        );
        let (beta, alpha) = self.pair(
            il,
            "ssm_beta_alpha.weight",
            "ssm_beta.weight",
            "ssm_alpha.weight",
            x,
            c.value_heads,
        )?;
        let dt = self.v(il, "ssm_dt.bias")?;
        let a = self.v(il, "ssm_a")?;
        let cw = self.v(il, "ssm_conv1d.weight")?;
        ensure!(
            cw.len() == raw.len() * c.conv_kernel
                && dt.len() == c.value_heads
                && a.len() == c.value_heads,
            "GDN parameter shape mismatch"
        );
        let mut h = vec![0.; raw.len()];
        for ch in 0..raw.len() {
            let mut v = cw[ch * c.conv_kernel + c.conv_kernel - 1] * raw[ch];
            for t in 0..c.conv_kernel - 1 {
                v += cw[ch * c.conv_kernel + t] * state.history[t][ch];
            }
            h[ch] = silu(v);
        }
        if !state.history.is_empty() {
            state.history.rotate_left(1);
            *state.history.last_mut().unwrap() = raw;
        }
        for head in 0..2 * c.key_heads {
            l2_norm(&mut h[head * s..(head + 1) * s], c.rms_epsilon);
        }
        let mut out = vec![0.; c.value_heads * s];
        for vh in 0..c.value_heads {
            let kh = vh % c.key_heads;
            let q = &h[kh * s..(kh + 1) * s];
            let k = &h[qk + kh * s..qk + (kh + 1) * s];
            let v = &h[2 * qk + vh * s..2 * qk + (vh + 1) * s];
            let decay = (softplus(alpha[vh] + dt[vh]) * a[vh]).exp();
            let beta = sigmoid(beta[vh]);
            let matrix = &mut state.matrix[vh * s * s..(vh + 1) * s * s];
            for col in 0..s {
                let mut prediction = 0.;
                for row in 0..s {
                    prediction += k[row] * matrix[row * s + col];
                }
                let delta = (v[col] - decay * prediction) * beta;
                let mut output = 0.;
                for row in 0..s {
                    matrix[row * s + col] = decay * matrix[row * s + col] + k[row] * delta;
                    output += q[row] * matrix[row * s + col];
                }
                out[vh * s + col] = output / (s as f32).sqrt();
            }
        }
        let gamma = self.v(il, "ssm_norm.weight")?;
        let mut out = group_norm(&out, &gamma, s, c.rms_epsilon)?;
        for (i, y) in out.iter_mut().enumerate() {
            *y *= sigmoid(z[i]);
        }
        self.mm(il, "ssm_out.weight", &out)
    }
    fn attention(
        &self,
        il: usize,
        x: &[f32],
        pos: usize,
        state: &mut AttentionState,
    ) -> Result<Vec<f32>> {
        let c = &self.config;
        let d = c.head_dim;
        let qg = self.mm(il, "attn_q.weight", x)?;
        let mut k = self.mm(il, "attn_k.weight", x)?;
        let v = self.mm(il, "attn_v.weight", x)?;
        ensure!(
            qg.len() == c.attention_heads * 2 * d
                && k.len() == c.kv_heads * d
                && v.len() == k.len(),
            "attention shape mismatch"
        );
        let qnorm = self.v(il, "attn_q_norm.weight")?;
        let knorm = self.v(il, "attn_k_norm.weight")?;
        let iq = self.mm(il, "indexer.q_proj.weight", x)?;
        let ik = self.mm(il, "indexer.k_proj.weight", x)?;
        ensure!(
            ik.len() == c.index_dim && iq.len() == c.index_heads * c.index_dim,
            "indexer shape mismatch"
        );
        state.raw_tail.push(ik);
        if state.raw_tail.len() == c.index_block {
            let mut pooled = vec![0.; c.index_dim];
            for row in &state.raw_tail {
                for (i, &v) in row.iter().enumerate() {
                    pooled[i] += v / c.index_block as f32;
                }
            }
            pooled = group_norm(
                &pooled,
                &self.v(il, "indexer.k_norm.weight")?,
                c.index_dim,
                c.rms_epsilon,
            )?;
            rope(&mut pooled, c.rope_dim, pos, c.rope_base);
            state.pooled.push(pooled);
            state.raw_tail.clear();
        }
        let mut iq = group_norm(
            &iq,
            &self.v(il, "indexer.q_norm.weight")?,
            c.index_dim,
            c.rms_epsilon,
        )?;
        for q in iq.chunks_exact_mut(c.index_dim) {
            rope(q, c.rope_dim, pos, c.rope_base);
        }
        let mut scores: Vec<_> = state
            .pooled
            .iter()
            .enumerate()
            .map(|(i, k)| {
                let score = iq
                    .chunks_exact(c.index_dim)
                    .map(|q| dot(q, k).max(0.) / (c.index_dim as f32).sqrt())
                    .sum::<f32>();
                (i, score)
            })
            .collect();
        ensure!(
            scores.iter().all(|(_, s)| s.is_finite()),
            "nonfinite QSA scores"
        );
        scores.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        scores.truncate(c.index_top_k / c.index_block);
        scores.sort_by_key(|x| x.0);
        let mut selected = Vec::new();
        for (i, _) in scores {
            selected.extend(i * c.index_block..(i + 1) * c.index_block);
        }
        selected.extend(state.pooled.len() * c.index_block..pos + 1);
        k = group_norm(&k, &knorm, d, c.rms_epsilon)?;
        for row in k.chunks_exact_mut(d) {
            rope(row, c.rope_dim, pos, c.rope_base);
        }
        state.keys.push(k);
        state.values.push(v);
        ensure!(
            state.keys.len() == pos + 1 && !selected.is_empty(),
            "attention position/state mismatch"
        );
        let mut output = vec![0.; c.attention_heads * d];
        for head in 0..c.attention_heads {
            let mut q = group_norm(
                &qg[head * 2 * d..head * 2 * d + d],
                &qnorm,
                d,
                c.rms_epsilon,
            )?;
            rope(&mut q, c.rope_dim, pos, c.rope_base);
            let kh = head / (c.attention_heads / c.kv_heads);
            let a: Vec<_> = selected
                .iter()
                .map(|&i| dot(&q, &state.keys[i][kh * d..(kh + 1) * d]) / (d as f32).sqrt())
                .collect();
            let probs = softmax(&a)?;
            for (&i, &p) in selected.iter().zip(&probs) {
                for j in 0..d {
                    output[head * d + j] += p * state.values[i][kh * d + j];
                }
            }
            for j in 0..d {
                output[head * d + j] *= sigmoid(qg[head * 2 * d + d + j]);
            }
        }
        self.mm(il, "attn_output.weight", &output)
    }
    fn moe(&self, il: usize, x: &[f32]) -> Result<Vec<f32>> {
        let c = &self.config;
        let logits = self.mm(il, "ffn_gate_inp.weight", x)?;
        ensure!(logits.len() == c.experts, "router expert count mismatch");
        let probs = softmax(&logits)?;
        let mut ids: Vec<_> = (0..c.experts).collect();
        ids.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]).then(a.cmp(&b)));
        ids.truncate(c.top_k);
        let sum: f32 = ids.iter().map(|&i| probs[i]).sum();
        ensure!(sum > 0., "empty router probability mass");
        let mut output = vec![0.; c.width];
        for e in ids {
            let (gate, up) = if self.weights.has(&Self::name(il, "ffn_gate_up_exps.weight")) {
                let mut y = self
                    .weights
                    .get(&Self::name(il, "ffn_gate_up_exps.weight"))?
                    .mm(x, e)?;
                ensure!(
                    y.len() == 2 * c.expert_width,
                    "merged expert shape mismatch"
                );
                let up = y.split_off(c.expert_width);
                (y, up)
            } else {
                (
                    self.weights
                        .get(&Self::name(il, "ffn_gate_exps.weight"))?
                        .mm(x, e)?,
                    self.weights
                        .get(&Self::name(il, "ffn_up_exps.weight"))?
                        .mm(x, e)?,
                )
            };
            let hidden: Vec<_> = gate.iter().zip(up).map(|(&g, u)| silu(g) * u).collect();
            let down = self
                .weights
                .get(&Self::name(il, "ffn_down_exps.weight"))?
                .mm(&hidden, e)?;
            for (i, y) in output.iter_mut().enumerate() {
                *y += down[i] * (probs[e] / sum) * c.expert_scale;
            }
        }
        let gate = self.mm(il, "ffn_gate_shexp.weight", x)?;
        let up = self.mm(il, "ffn_up_shexp.weight", x)?;
        let hidden: Vec<_> = gate.iter().zip(up).map(|(&g, u)| silu(g) * u).collect();
        let down = self.mm(il, "ffn_down_shexp.weight", &hidden)?;
        let shared = self.mm(il, "ffn_gate_inp_shexp.weight", x)?;
        ensure!(shared.len() == 1, "shared gate is not scalar");
        for (i, y) in output.iter_mut().enumerate() {
            *y += down[i] * sigmoid(shared[0]);
        }
        Ok(output)
    }
    fn ple(&self, il: usize, token: u32, r: &mut [f32], state: &mut State) -> Result<()> {
        let c = &self.config;
        let p = c.ple.as_ref().context("PLE not configured")?;
        let rows = ple_rows(p, &state.tokens, token);
        let mut emb = Vec::new();
        for row in rows {
            emb.extend(
                self.weights
                    .get("per_layer_token_embd.weight")?
                    .row(row as usize)?,
            );
        }
        let key = group_norm(
            &self.mm(il, "ple_key.weight", &emb)?,
            &self.v(il, "ple_norm_key.weight")?,
            c.width,
            c.rms_epsilon,
        )?;
        let query = group_norm(
            r,
            &self.v(il, "ple_norm_query.weight")?,
            c.width,
            c.rms_epsilon,
        )?;
        let value = self.mm(il, "ple_value.weight", &emb)?;
        let mut gated = vec![0.; r.len()];
        for stream in 0..c.hc {
            let lo = stream * c.width;
            let score =
                dot(&key[lo..lo + c.width], &query[lo..lo + c.width]) / (c.width as f32).sqrt();
            // f32::signum(0) is +1, while the model's sgn(0) is zero.
            let gate = if score == 0.0 {
                0.5
            } else {
                sigmoid(score.signum() * score.abs().max(1e-6).sqrt())
            };
            for j in 0..c.width {
                gated[lo + j] = value[j] * gate;
            }
        }
        let normalized = group_norm(
            &gated,
            &self.v(il, "ple_norm_conv.weight")?,
            c.width,
            c.rms_epsilon,
        )?;
        let cw = self.v(il, "ple_conv1d.weight")?;
        let history = &mut state.ple_history[il];
        let mut conv = vec![0.; r.len()];
        for ch in 0..r.len() {
            for tap in 0..p.conv_kernel {
                let delay = (p.conv_kernel - 1 - tap) * p.ngram;
                let v = if delay == 0 {
                    normalized[ch]
                } else {
                    history
                        .len()
                        .checked_sub(delay)
                        .map(|i| history[i][ch])
                        .unwrap_or(0.)
                };
                conv[ch] += cw[ch * p.conv_kernel + tap] * v;
            }
        }
        history.push(normalized);
        let cap = (p.conv_kernel - 1) * p.ngram;
        if history.len() > cap {
            history.remove(0);
        }
        for i in 0..r.len() {
            r[i] += gated[i] + silu(conv[i]);
        }
        Ok(())
    }
    fn block(&self, il: usize, r: &mut [f32], state: &mut State) -> Result<()> {
        let (x, inject) = self.hc_mix(r, &format!("blk.{il}.hc_attn"), true)?;
        let y = if self.config.is_attention(il) {
            self.attention(il, &x, state.position, &mut state.attention[il])?
        } else {
            self.gdn(il, &x, &mut state.gdn[il])?
        };
        scatter(r, &y, &inject, self.config.width)?;
        let (x, inject) = self.hc_mix(r, &format!("blk.{il}.hc_ffn"), true)?;
        let y = self.moe(il, &x)?;
        scatter(r, &y, &inject, self.config.width)
    }
}
impl Backend for Reference {
    type Target = State;
    type Draft = State;
    fn target(&mut self, state: &mut State, token: u32) -> Result<Step> {
        ensure!(
            state.position < self.max_context && (token as usize) < self.config.vocab,
            "context/token outside prepared capacity"
        );
        let emb = self.weights.get("token_embd.weight")?.row(token as usize)?;
        let mut r = emb.repeat(self.config.hc);
        for il in 0..self.config.layers {
            if self
                .config
                .ple
                .as_ref()
                .is_some_and(|p| p.layers.contains(&il))
            {
                self.ple(il, token, &mut r, state)?;
            }
            self.block(il, &mut r, state)?;
        }
        let logits = self.head(&r, false)?;
        state.tokens.push(token);
        state.position += 1;
        Ok(Step { logits, hidden: r })
    }
    fn draft(
        &mut self,
        state: &mut State,
        hidden: &[f32],
        next_token: u32,
        position: usize,
    ) -> Result<Step> {
        let c = &self.config;
        ensure!(
            position == state.position
                && position < self.max_context
                && hidden.len() == c.width * c.hc,
            "MTP position/hidden contract mismatch"
        );
        let il = c.layers;
        let e = self
            .weights
            .get("token_embd.weight")?
            .row(next_token as usize)?;
        let e = group_norm(
            &e,
            &self.v(il, "nextn.enorm.weight")?,
            c.width,
            c.rms_epsilon,
        )?;
        let h = group_norm(
            hidden,
            &self.v(il, "nextn.hnorm.weight")?,
            c.width,
            c.rms_epsilon,
        )?;
        let mut r = Vec::new();
        for stream in 0..c.hc {
            let mut joined = e.clone();
            joined.extend_from_slice(&h[stream * c.width..(stream + 1) * c.width]);
            r.extend(self.mm(il, "nextn.eh_proj.weight", &joined)?);
        }
        self.block(il, &mut r, state)?;
        let logits = self.head(&r, true)?;
        state.tokens.push(next_token);
        state.position += 1;
        Ok(Step { logits, hidden: r })
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn sigmoid(x: f32) -> f32 {
    if x >= 0. {
        1. / (1. + (-x).exp())
    } else {
        let e = x.exp();
        e / (1. + e)
    }
}
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}
fn softplus(x: f32) -> f32 {
    if x > 20. { x } else { x.exp().ln_1p() }
}
fn group_norm(x: &[f32], gamma: &[f32], width: usize, eps: f32) -> Result<Vec<f32>> {
    ensure!(
        width > 0 && x.len() % width == 0 && (gamma.len() == width || gamma.len() == x.len()),
        "normalization geometry mismatch"
    );
    let mut y = x.to_vec();
    for (group, row) in y.chunks_exact_mut(width).enumerate() {
        let scale = 1. / (row.iter().map(|x| x * x).sum::<f32>() / width as f32 + eps).sqrt();
        for (j, x) in row.iter_mut().enumerate() {
            *x *= scale
                * gamma[if gamma.len() == width {
                    j
                } else {
                    group * width + j
                }];
        }
    }
    Ok(y)
}
fn l2_norm(x: &mut [f32], eps: f32) {
    let scale = 1. / (x.iter().map(|x| x * x).sum::<f32>() + eps).sqrt();
    for v in x {
        *v *= scale;
    }
}
fn rope(x: &mut [f32], rot: usize, pos: usize, base: f32) {
    for j in 0..rot / 2 {
        let theta = pos as f32 * base.powf(-((2 * j) as f32) / rot as f32);
        let (s, c) = theta.sin_cos();
        let a = x[j];
        let b = x[j + rot / 2];
        x[j] = a * c - b * s;
        x[j + rot / 2] = a * s + b * c;
    }
}
fn softmax(x: &[f32]) -> Result<Vec<f32>> {
    ensure!(
        !x.is_empty() && x.iter().all(|x| x.is_finite()),
        "nonfinite/empty softmax"
    );
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut y: Vec<_> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = y.iter().sum();
    for v in &mut y {
        *v /= sum;
    }
    Ok(y)
}
fn scatter(r: &mut [f32], output: &[f32], inject: &[f32], width: usize) -> Result<()> {
    ensure!(
        output.len() == width && r.len() == inject.len() * width,
        "HC scatter geometry mismatch"
    );
    for (stream, &g) in inject.iter().enumerate() {
        let g = 2. * sigmoid(g);
        for j in 0..width {
            r[stream * width + j] += g * output[j];
        }
    }
    Ok(())
}

/// A deterministic miniature architecture exercises actual GDN/QSA/PLE/MoE/MTP.
pub fn tiny_weights(c: &Config) -> Weights {
    let mut w = Weights::default();
    let mut seed = 7u64;
    let mut add = |name: String, dims: Vec<usize>, norm: bool| {
        let n = dims.iter().product();
        let values = (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                if norm {
                    1.0
                } else {
                    ((seed >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 0.3
                }
            })
            .collect();
        w.0.insert(name, Tensor::owned(dims, values).unwrap());
    };
    add("token_embd.weight".into(), vec![c.width, c.vocab], false);
    add("output.weight".into(), vec![c.width, c.vocab], false);
    for prefix in [
        "output_hc".to_string(),
        format!("blk.{}.nextn.hc_head", c.layers),
    ] {
        add(format!("{prefix}_norm.weight"), vec![c.width, c.hc], true);
        add(
            format!("{prefix}_down.weight"),
            vec![c.width * c.hc, c.hc_rank],
            false,
        );
        add(
            format!("{prefix}_up.weight"),
            vec![c.hc_rank, c.width * c.hc],
            false,
        );
    }
    for il in 0..=c.layers {
        for hc in ["hc_attn", "hc_ffn"] {
            add(
                format!("blk.{il}.{hc}_norm.weight"),
                vec![c.width, c.hc],
                true,
            );
            add(
                format!("blk.{il}.{hc}_down.weight"),
                vec![c.width * c.hc, c.hc_rank],
                false,
            );
            add(
                format!("blk.{il}.{hc}_up.weight"),
                vec![c.hc_rank, c.width * c.hc],
                false,
            );
            add(
                format!("blk.{il}.{hc}_inject.weight"),
                vec![c.width * c.hc, c.hc],
                false,
            );
        }
        if c.is_attention(il) {
            for (name, k, n) in [
                ("attn_q", c.width, c.attention_heads * c.head_dim * 2),
                ("attn_k", c.width, c.kv_heads * c.head_dim),
                ("attn_v", c.width, c.kv_heads * c.head_dim),
                ("attn_output", c.attention_heads * c.head_dim, c.width),
                ("indexer.q_proj", c.width, c.index_heads * c.index_dim),
                ("indexer.k_proj", c.width, c.index_dim),
            ] {
                add(format!("blk.{il}.{name}.weight"), vec![k, n], false);
            }
            for (name, d) in [
                ("attn_q_norm", c.head_dim),
                ("attn_k_norm", c.head_dim),
                ("indexer.q_norm", c.index_dim),
                ("indexer.k_norm", c.index_dim),
            ] {
                add(format!("blk.{il}.{name}.weight"), vec![d], true);
            }
        } else {
            for (name, k, n) in [
                ("attn_qkv", c.width, c.conv_channels()),
                ("attn_gate", c.width, c.value_heads * c.gdn_dim),
                ("ssm_alpha", c.width, c.value_heads),
                ("ssm_beta", c.width, c.value_heads),
                ("ssm_out", c.value_heads * c.gdn_dim, c.width),
                ("ssm_conv1d", c.conv_kernel, c.conv_channels()),
            ] {
                add(format!("blk.{il}.{name}.weight"), vec![k, n], false);
            }
            add(format!("blk.{il}.ssm_dt.bias"), vec![c.value_heads], false);
            add(format!("blk.{il}.ssm_a"), vec![c.value_heads], false);
            add(format!("blk.{il}.ssm_norm.weight"), vec![c.gdn_dim], true);
        }
        for (name, k, n, e) in [
            ("ffn_gate_inp", c.width, c.experts, 1),
            ("ffn_gate_exps", c.width, c.expert_width, c.experts),
            ("ffn_up_exps", c.width, c.expert_width, c.experts),
            ("ffn_down_exps", c.expert_width, c.width, c.experts),
            ("ffn_gate_shexp", c.width, c.shared_width, 1),
            ("ffn_up_shexp", c.width, c.shared_width, 1),
            ("ffn_down_shexp", c.shared_width, c.width, 1),
            ("ffn_gate_inp_shexp", c.width, 1, 1),
        ] {
            add(format!("blk.{il}.{name}.weight"), vec![k, n, e], false);
        }
    }
    let il = c.layers;
    add(format!("blk.{il}.nextn.enorm.weight"), vec![c.width], true);
    add(
        format!("blk.{il}.nextn.hnorm.weight"),
        vec![c.width, c.hc],
        true,
    );
    add(
        format!("blk.{il}.nextn.eh_proj.weight"),
        vec![2 * c.width, c.width],
        false,
    );
    if let Some(p) = &c.ple {
        let rows = p
            .offsets
            .iter()
            .zip(&p.vocabs)
            .map(|(a, b)| a + b)
            .max()
            .unwrap() as usize;
        add(
            "per_layer_token_embd.weight".into(),
            vec![p.head_dim, rows],
            false,
        );
        for &il in &p.layers {
            add(
                format!("blk.{il}.ple_key.weight"),
                vec![c.width, c.width * c.hc],
                false,
            );
            add(
                format!("blk.{il}.ple_value.weight"),
                vec![c.width, c.width],
                false,
            );
            for name in ["ple_norm_key", "ple_norm_query", "ple_norm_conv"] {
                add(format!("blk.{il}.{name}.weight"), vec![c.width, c.hc], true);
            }
            add(
                format!("blk.{il}.ple_conv1d.weight"),
                vec![p.conv_kernel, c.width * c.hc],
                false,
            );
        }
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mtp::{Session, argmax};
    #[test]
    fn full_model_mtp_matches_target_and_continuation_state() {
        let c = Config::tiny();
        let w = tiny_weights(&c);
        let mut ordinary = Reference::new(c.clone(), w.clone(), 64).unwrap();
        let mut baseline = State::new(&c);
        let prompt = [1, 3, 2, 4, 7];
        let mut tail = None;
        for &t in &prompt {
            tail = Some(ordinary.target(&mut baseline, t).unwrap());
        }
        let mut expected = Vec::new();
        for _ in 0..13 {
            let t = argmax(&tail.as_ref().unwrap().logits).unwrap();
            expected.push(t);
            tail = Some(ordinary.target(&mut baseline, t).unwrap());
        }
        let model = Reference::new(c.clone(), w, 64).unwrap();
        let mut session = Session::new(model, State::new(&c), State::new(&c), 3, None).unwrap();
        session.prefill(&prompt).unwrap();
        assert_eq!(session.generate(13).unwrap(), expected);
        assert_eq!(session.target, baseline);
        assert_eq!(session.target.attention[1].pooled.len(), 9);
        assert!(!session.target.ple_history[0].is_empty());
        assert!(session.stats.proposed > 0);
    }
    #[test]
    fn prompt_chunk_boundaries_do_not_change_model_state() {
        let c = Config::tiny();
        let w = tiny_weights(&c);
        let make = || {
            Session::new(
                Reference::new(c.clone(), w.clone(), 64).unwrap(),
                State::new(&c),
                State::new(&c),
                3,
                None,
            )
            .unwrap()
        };
        let mut a = make();
        let mut b = make();
        a.prefill(&[1, 2, 3, 4, 5]).unwrap();
        b.prefill(&[1, 2]).unwrap();
        b.prefill(&[3, 4, 5]).unwrap();
        assert_eq!(a.target, b.target);
        assert_eq!(a.draft, b.draft);
        assert_eq!(a.generate(4).unwrap(), b.generate(4).unwrap());
    }
}
