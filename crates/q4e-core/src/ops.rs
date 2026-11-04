//! Operator seam shared by the scalar model and explicit device-validation backends.
//! Model semantics and MTP transactions stay in Rust; this seam does not move state ownership.
use crate::reference::Tensor;
use anyhow::{Result, ensure};
use serde::Serialize;

pub struct Attention<'a> {
    pub query: &'a [f32],
    pub keys: &'a [Vec<f32>],
    pub values: &'a [Vec<f32>],
    pub selected: &'a [usize],
    pub heads: usize,
    pub kv_heads: usize,
    pub dim: usize,
}
pub struct DeltaStep<'a> {
    pub qkv: &'a [f32],
    pub decay: &'a [f32],
    pub beta: &'a [f32],
    pub key_heads: usize,
    pub value_heads: usize,
    pub dim: usize,
}
#[derive(Debug, Default, Serialize)]
pub struct OperatorStats {
    pub matmuls: u64,
    pub attention: u64,
    pub gdn: u64,
    pub resident_weight_bytes: u64,
    pub weight_uploads: u64,
}

pub trait Operators {
    fn name(&self) -> &str;
    fn matvec(&self, name: &str, tensor: &Tensor, expert: usize, input: &[f32])
    -> Result<Vec<f32>>;
    fn delta(&self, p: &DeltaStep<'_>, matrix: &mut [f32]) -> Result<Vec<f32>>;
    fn attention(&self, p: &Attention<'_>) -> Result<Vec<f32>>;
    fn stats(&self) -> OperatorStats {
        OperatorStats::default()
    }
}
pub struct Scalar;
impl Operators for Scalar {
    fn name(&self) -> &str {
        "scalar_reference"
    }
    fn matvec(
        &self,
        _name: &str,
        tensor: &Tensor,
        expert: usize,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        tensor.mm(input, expert)
    }
    fn delta(&self, p: &DeltaStep<'_>, matrix: &mut [f32]) -> Result<Vec<f32>> {
        scalar_delta(p, matrix)
    }
    fn attention(&self, p: &Attention<'_>) -> Result<Vec<f32>> {
        scalar_attention(p)
    }
}

pub fn scalar_delta(p: &DeltaStep<'_>, state: &mut [f32]) -> Result<Vec<f32>> {
    let s = p.dim;
    ensure!(
        s > 0 && p.key_heads > 0 && p.value_heads > 0 && p.value_heads % p.key_heads == 0,
        "invalid GDN heads"
    );
    ensure!(
        p.qkv.len() == (2 * p.key_heads + p.value_heads) * s
            && p.decay.len() == p.value_heads
            && p.beta.len() == p.value_heads
            && state.len() == p.value_heads * s * s,
        "invalid GDN extents"
    );
    let qk = p.key_heads * s;
    let mut out = vec![0.; p.value_heads * s];
    for vh in 0..p.value_heads {
        let kh = vh % p.key_heads;
        let q = &p.qkv[kh * s..(kh + 1) * s];
        let k = &p.qkv[qk + kh * s..qk + (kh + 1) * s];
        let v = &p.qkv[2 * qk + vh * s..2 * qk + (vh + 1) * s];
        let m = &mut state[vh * s * s..(vh + 1) * s * s];
        for col in 0..s {
            let mut prediction = 0.;
            for row in 0..s {
                prediction += k[row] * m[row * s + col];
            }
            let delta = (v[col] - p.decay[vh] * prediction) * p.beta[vh];
            let mut output = 0.;
            for row in 0..s {
                m[row * s + col] = p.decay[vh] * m[row * s + col] + k[row] * delta;
                output += q[row] * m[row * s + col];
            }
            out[vh * s + col] = output / (s as f32).sqrt();
        }
    }
    Ok(out)
}
pub fn scalar_attention(p: &Attention<'_>) -> Result<Vec<f32>> {
    ensure!(
        p.dim > 0
            && p.heads > 0
            && p.kv_heads > 0
            && p.heads % p.kv_heads == 0
            && p.query.len() == p.heads * p.dim
            && !p.selected.is_empty(),
        "invalid attention geometry"
    );
    let mut out = vec![0.; p.heads * p.dim];
    for h in 0..p.heads {
        let kh = h / (p.heads / p.kv_heads);
        let q = &p.query[h * p.dim..(h + 1) * p.dim];
        let mut scores = Vec::with_capacity(p.selected.len());
        for &i in p.selected {
            ensure!(
                i < p.keys.len()
                    && i < p.values.len()
                    && p.keys[i].len() == p.kv_heads * p.dim
                    && p.values[i].len() == p.kv_heads * p.dim,
                "invalid selected KV row"
            );
            let k = &p.keys[i][kh * p.dim..(kh + 1) * p.dim];
            scores.push(q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() / (p.dim as f32).sqrt());
        }
        ensure!(
            scores.iter().all(|x| x.is_finite()),
            "nonfinite attention score"
        );
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for a in &mut scores {
            *a = (*a - max).exp();
        }
        let sum: f32 = scores.iter().sum();
        for (&i, a) in p.selected.iter().zip(scores) {
            let probability = a / sum;
            for j in 0..p.dim {
                out[h * p.dim + j] += probability * p.values[i][kh * p.dim + j];
            }
        }
    }
    Ok(out)
}
