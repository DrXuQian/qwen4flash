use crate::gguf::{ModelIndex, Value};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Config {
    pub layers: usize,
    pub width: usize,
    pub vocab: usize,
    pub hc: usize,
    pub hc_rank: usize,
    pub experts: usize,
    pub top_k: usize,
    pub expert_width: usize,
    pub shared_width: usize,
    pub gdn_dim: usize,
    pub key_heads: usize,
    pub value_heads: usize,
    pub conv_kernel: usize,
    pub attention_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub rope_base: f32,
    pub rms_epsilon: f32,
    pub attention_interval: usize,
    pub index_heads: usize,
    pub index_dim: usize,
    pub index_block: usize,
    pub index_top_k: usize,
    pub expert_scale: f32,
    pub embedded_mtp_layers: usize,
    pub ple: Option<PleConfig>,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PleConfig {
    pub layers: Vec<usize>,
    pub ngram: usize,
    pub heads_per_ngram: usize,
    pub conv_kernel: usize,
    pub eos: u32,
    pub head_dim: usize,
    pub multipliers: Vec<u64>,
    pub offsets: Vec<u64>,
    pub vocabs: Vec<u64>,
}
impl Config {
    pub fn from_index(index: &ModelIndex) -> Result<Self> {
        let m = index.metadata();
        ensure!(
            m.get("general.architecture")
                .context("missing architecture")?
                .text()?
                == "qwen4exp",
            "only qwen4exp is supported"
        );
        let key = |s: &str| format!("qwen4exp.{s}");
        let u = |s: &str| -> Result<usize> {
            usize::try_from(
                m.get(&key(s))
                    .with_context(|| format!("missing {}", key(s)))?
                    .u64()?,
            )
            .context("metadata exceeds address space")
        };
        let opt = |s: &str, d: usize| -> Result<usize> {
            m.get(&key(s))
                .map(|v| {
                    v.u64()
                        .and_then(|x| usize::try_from(x).context("metadata overflow"))
                })
                .unwrap_or(Ok(d))
        };
        let f = |s: &str, d: f32| -> Result<f32> {
            Ok(m.get(&key(s))
                .map(Value::f64)
                .transpose()?
                .unwrap_or(d as f64) as f32)
        };
        let arr = |s: &str| -> Result<Vec<u64>> {
            m.get(&key(s))
                .with_context(|| format!("missing {}", key(s)))?
                .array()?
                .iter()
                .map(Value::u64)
                .collect()
        };
        let embedded_mtp_layers = opt("nextn_predict_layers", 0)?;
        let blocks = u("block_count")?;
        let layers = blocks
            .checked_sub(embedded_mtp_layers)
            .context("MTP count exceeds block count")?;
        let ratios = arr("attention.compress_ratios")?;
        let index_block = *ratios
            .iter()
            .find(|&&x| x > 0)
            .context("no QSA compression ratio")? as usize;
        ensure!(
            ratios
                .iter()
                .filter(|&&x| x > 0)
                .all(|&x| x as usize == index_block),
            "varying QSA compression ratios are unsupported"
        );
        let vocab = m
            .get("tokenizer.ggml.tokens")
            .map(|v| v.array().map(|x| x.len()))
            .transpose()?
            .unwrap_or(opt("vocab_size", 0)?);
        let ple = if m.contains_key(&key("ple.layers")) {
            let layers = arr("ple.layers")?;
            if layers.is_empty() {
                None
            } else {
                Some(PleConfig {
                    layers: layers.into_iter().map(|x| x as usize).collect(),
                    ngram: u("ple.ngram_size")?,
                    heads_per_ngram: u("ple.heads_per_ngram")?,
                    conv_kernel: u("ple.conv_kernel")?,
                    eos: u("ple.eos_token_id")? as u32,
                    head_dim: u("embedding_length_per_layer_input")?,
                    multipliers: arr("ple.layer_multipliers")?,
                    offsets: arr("ple.head_offsets")?,
                    vocabs: arr("ple.head_vocab_sizes")?,
                })
            }
        } else {
            None
        };
        let expert_width = match m
            .get(&key("expert_feed_forward_length"))
            .context("missing expert width")?
        {
            Value::Array(v) => {
                let a = v.first().context("empty expert widths")?.u64()?;
                ensure!(
                    v.iter().all(|x| x.u64().ok() == Some(a)),
                    "varying expert widths require per-layer plans"
                );
                a as usize
            }
            v => v.u64()? as usize,
        };
        let c = Self {
            layers,
            width: u("embedding_length")?,
            vocab,
            hc: u("hyper_connection.count")?,
            hc_rank: u("hyper_connection.low_rank")?,
            experts: u("expert_count")?,
            top_k: u("expert_used_count")?,
            expert_width,
            shared_width: opt("expert_shared_feed_forward_length", expert_width)?,
            gdn_dim: u("ssm.state_size")?,
            key_heads: u("ssm.group_count")?,
            value_heads: u("ssm.time_step_rank")?,
            conv_kernel: u("ssm.conv_kernel")?,
            attention_heads: u("attention.head_count")?,
            kv_heads: u("attention.head_count_kv")?,
            head_dim: u("attention.key_length")?,
            rope_dim: u("rope.dimension_count")?,
            rope_base: f("rope.freq_base", 1e7)?,
            rms_epsilon: f("attention.layer_norm_rms_epsilon", 1e-6)?,
            attention_interval: opt("full_attention_interval", 4)?,
            index_heads: u("attention.indexer.head_count")?,
            index_dim: u("attention.indexer.key_length")?,
            index_block,
            index_top_k: u("attention.indexer.top_k")?,
            expert_scale: f("expert_weights_scale", 1.0)?,
            embedded_mtp_layers,
            ple,
        };
        c.validate()?;
        Ok(c)
    }
    pub fn validate(&self) -> Result<()> {
        for (name, n) in [
            ("layers", self.layers),
            ("width", self.width),
            ("vocab", self.vocab),
            ("hc_rank", self.hc_rank),
            ("experts", self.experts),
            ("top_k", self.top_k),
            ("expert width", self.expert_width),
            ("shared width", self.shared_width),
            ("GDN dimension", self.gdn_dim),
            ("key heads", self.key_heads),
            ("value heads", self.value_heads),
            ("conv kernel", self.conv_kernel),
            ("attention heads", self.attention_heads),
            ("KV heads", self.kv_heads),
            ("head dimension", self.head_dim),
            ("attention interval", self.attention_interval),
            ("index heads", self.index_heads),
            ("index dimension", self.index_dim),
            ("index block", self.index_block),
            ("index top k", self.index_top_k),
        ] {
            ensure!(n > 0 && n <= 1 << 24, "invalid {name}: {n}");
        }
        ensure!(
            self.hc > 1 && self.hc <= 16,
            "invalid hyperconnection count"
        );
        ensure!(
            self.top_k <= self.experts
                && self.value_heads % self.key_heads == 0
                && self.attention_heads % self.kv_heads == 0,
            "invalid head/expert geometry"
        );
        ensure!(
            self.index_block > 1 && self.index_top_k % self.index_block == 0,
            "invalid QSA budget"
        );
        ensure!(
            self.rope_dim > 0
                && self.rope_dim % 2 == 0
                && self.rope_dim <= self.head_dim
                && self.rope_dim <= self.index_dim,
            "invalid rotary geometry"
        );
        ensure!(
            self.rms_epsilon.is_finite()
                && self.rms_epsilon > 0.0
                && self.rope_base.is_finite()
                && self.rope_base > 1.0
                && self.expert_scale.is_finite(),
            "invalid numerical metadata"
        );
        ensure!(
            self.embedded_mtp_layers <= 1,
            "qwen4exp requires exactly one MTP block when embedded"
        );
        if let Some(p) = &self.ple {
            let heads = p.ngram.saturating_sub(1) * p.heads_per_ngram;
            ensure!(
                p.ngram >= 2
                    && p.ngram <= 32
                    && p.conv_kernel > 0
                    && heads > 0
                    && p.head_dim * heads == self.width,
                "invalid PLE geometry"
            );
            ensure!(
                p.multipliers.len() >= p.ngram
                    && p.offsets.len() == heads
                    && p.vocabs.len() == heads
                    && p.vocabs.iter().all(|&x| x > 0),
                "invalid PLE hash contract"
            );
            ensure!(
                p.layers.iter().all(|&x| x < self.layers),
                "PLE layer outside trunk"
            );
        }
        Ok(())
    }
    pub fn is_attention(&self, layer: usize) -> bool {
        layer >= self.layers || (layer + 1) % self.attention_interval == 0
    }
    pub fn conv_channels(&self) -> usize {
        (2 * self.key_heads + self.value_heads) * self.gdn_dim
    }
    pub fn tiny() -> Self {
        Self {
            layers: 2,
            width: 8,
            vocab: 16,
            hc: 2,
            hc_rank: 4,
            experts: 3,
            top_k: 2,
            expert_width: 4,
            shared_width: 4,
            gdn_dim: 4,
            key_heads: 1,
            value_heads: 2,
            conv_kernel: 4,
            attention_heads: 2,
            kv_heads: 1,
            head_dim: 4,
            rope_dim: 4,
            rope_base: 10000.0,
            rms_epsilon: 1e-6,
            attention_interval: 2,
            index_heads: 2,
            index_dim: 4,
            index_block: 2,
            index_top_k: 4,
            expert_scale: 1.0,
            embedded_mtp_layers: 1,
            ple: Some(PleConfig {
                layers: vec![0],
                ngram: 3,
                heads_per_ngram: 1,
                conv_kernel: 3,
                eos: 0,
                head_dim: 4,
                multipliers: vec![1, 17, 31],
                offsets: vec![0, 16],
                vocabs: vec![16, 16],
            }),
        }
    }
}

pub fn ple_rows(p: &PleConfig, history: &[u32], token: u32) -> Vec<u64> {
    let mut context = vec![p.eos; p.ngram];
    context[0] = token;
    let mut cut = false;
    for (j, c) in context.iter_mut().enumerate().skip(1) {
        let v = history.len().checked_sub(j).map(|i| history[i]);
        cut |= v.is_none() || v == Some(p.eos);
        if !cut {
            *c = v.unwrap();
        }
    }
    let mut rows = Vec::new();
    let mut mixed = (context[0] as u64).wrapping_mul(p.multipliers[0]);
    for (j, &token) in context.iter().enumerate().skip(1) {
        mixed ^= (token as u64).wrapping_mul(p.multipliers[j]);
        for g in 0..p.heads_per_ngram {
            let h = (j - 1) * p.heads_per_ngram + g;
            rows.push(mixed % p.vocabs[h] + p.offsets[h]);
        }
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn eos_cuts_predecessors_not_current_token() {
        let p = Config::tiny().ple.unwrap();
        assert_eq!(ple_rows(&p, &[3, 0], 5), ple_rows(&p, &[], 5));
        assert_ne!(ple_rows(&p, &[3, 4], 0), ple_rows(&p, &[], 0));
    }
    #[test]
    fn metadata_rejects_bad_geometry() {
        let mut c = Config::tiny();
        c.validate().unwrap();
        c.top_k = 4;
        assert!(c.validate().is_err());
    }
}
