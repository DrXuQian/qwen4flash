//! Byte-preserving, bounded preparation helpers independent of ggml.
use crate::reference::Tensor;
use anyhow::{Context, Result, ensure};

/// Join gate then up for ONE expert, preserving every GGUF block byte.
/// Qwen's N=640 becomes N=1280 without padding/requantization. The returned
/// tensor has one expert; callers append its canonical planes at that expert's
/// offsets. Processing one expert at a time avoids a model-sized CPU mirror.
pub fn join_expert_gate_up(gate: &Tensor, up: &Tensor, expert: usize) -> Result<Tensor> {
    ensure!(
        (2..=3).contains(&gate.dims.len()) && gate.dims == up.dims,
        "gate/up shapes differ"
    );
    let experts = gate.dims.get(2).copied().unwrap_or(1);
    ensure!(expert < experts, "gate/up expert out of range");
    let (gkind, g) = gate.storage();
    let (ukind, u) = up.storage();
    ensure!(
        gkind == ukind && g.len() == u.len(),
        "gate/up storage formats differ"
    );
    let bytes = g.len() / experts;
    let offset = expert
        .checked_mul(bytes)
        .context("expert offset overflow")?;
    let mut joined = Vec::with_capacity(bytes.checked_mul(2).context("joined size overflow")?);
    joined.extend_from_slice(&g[offset..offset + bytes]);
    joined.extend_from_slice(&u[offset..offset + bytes]);
    Tensor::encoded(
        vec![
            gate.dims[0],
            gate.dims[1].checked_mul(2).context("joined N overflow")?,
            1,
        ],
        gkind,
        joined,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn expert_join_keeps_original_blocks_and_expert_order() {
        let per = 640 * 144;
        let g: Vec<_> = (0..per * 3).map(|i| (i % 251) as u8).collect();
        let u: Vec<_> = (0..per * 3).map(|i| (255 - i % 253) as u8).collect();
        let gate = Tensor::encoded(vec![256, 640, 3], 12, g.clone()).unwrap();
        let up = Tensor::encoded(vec![256, 640, 3], 12, u.clone()).unwrap();
        let t = join_expert_gate_up(&gate, &up, 2).unwrap();
        assert_eq!(t.dims, vec![256, 1280, 1]);
        let (kind, bytes) = t.storage();
        assert_eq!(kind, 12);
        assert_eq!(&bytes[..per], &g[2 * per..]);
        assert_eq!(&bytes[per..], &u[2 * per..]);
        assert!(join_expert_gate_up(&gate, &up, 3).is_err());
    }
    #[test]
    fn equal_byte_sizes_do_not_authorize_a_different_quant_format() {
        let gate = Tensor::encoded(vec![32, 2], 2, vec![0; 36]).unwrap();
        let up = Tensor::encoded(vec![32, 2], 20, vec![0; 36]).unwrap();
        assert!(join_expert_gate_up(&gate, &up, 0).is_err());
    }
}
