//! Independent scalar readers used for numerical bring-up, not PPU performance.
use crate::gguf::type_layout;
use anyhow::{Result, bail, ensure};
use half::{bf16, f16};

fn h(b: &[u8], i: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([b[i], b[i + 1]])).to_f32()
}
pub fn decode_row(kind: u32, bytes: &[u8], out: &mut [f32]) -> Result<()> {
    let (block, size) = type_layout(kind)?;
    let (block, size) = (block as usize, size as usize);
    ensure!(
        out.len() % block == 0 && bytes.len() == out.len() / block * size,
        "quant row byte/element mismatch"
    );
    for (b, y) in bytes.chunks_exact(size).zip(out.chunks_exact_mut(block)) {
        match kind {
            0 => y[0] = f32::from_le_bytes(b.try_into().unwrap()),
            1 => y[0] = h(b, 0),
            30 => y[0] = bf16::from_bits(u16::from_le_bytes(b.try_into().unwrap())).to_f32(),
            8 => {
                let d = h(b, 0);
                for i in 0..32 {
                    y[i] = d * (b[i + 2] as i8 as f32);
                }
            }
            2 | 3 | 6 | 7 | 20 => {
                const IQ: [i8; 16] = [
                    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
                ];
                let d = h(b, 0);
                let affine = kind == 3 || kind == 7;
                let m = if affine { h(b, 2) } else { 0. };
                let five = kind == 6 || kind == 7;
                let qoff = 2 + if affine { 2 } else { 0 };
                let qh = if five {
                    u32::from_le_bytes(b[qoff..qoff + 4].try_into().unwrap())
                } else {
                    0
                };
                let qs = &b[qoff + if five { 4 } else { 0 }..];
                for i in 0..32 {
                    let mut q = ((qs[i % 16] >> (4 * (i / 16))) & 15) as i32;
                    if five {
                        q |= (((qh >> i) & 1) as i32) << 4;
                    }
                    y[i] = if kind == 20 {
                        d * IQ[q as usize] as f32
                    } else {
                        d * (q - if affine {
                            0
                        } else if five {
                            16
                        } else {
                            8
                        }) as f32
                            + m
                    };
                }
            }
            12 | 13 => {
                let d = h(b, 0);
                let dm = h(b, 2);
                let scales = &b[4..16];
                let qh = if kind == 13 { &b[16..48] } else { &b[0..0] };
                let qs = &b[if kind == 13 { 48 } else { 16 }..];
                for group in 0..8 {
                    let (sc, m) = if group < 4 {
                        (scales[group] & 63, scales[group + 4] & 63)
                    } else {
                        (
                            (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4),
                            (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
                        )
                    };
                    for j in 0..32 {
                        let low = (qs[(group / 2) * 32 + j] >> (4 * (group % 2))) & 15;
                        let high = if kind == 13 {
                            ((qh[j] >> group) & 1) * 16
                        } else {
                            0
                        };
                        y[group * 32 + j] = d * sc as f32 * (low + high) as f32 - dm * m as f32;
                    }
                }
            }
            14 => {
                let d = h(b, 208);
                let ql = &b[..128];
                let qh = &b[128..192];
                let scales = &b[192..208];
                for half in 0..2 {
                    for l in 0..32 {
                        let is = l / 16;
                        let lo = half * 64 + l;
                        let hi = qh[half * 32 + l];
                        let q = [
                            (ql[lo] & 15) | ((hi & 3) << 4),
                            (ql[lo + 32] & 15) | (((hi >> 2) & 3) << 4),
                            (ql[lo] >> 4) | (((hi >> 4) & 3) << 4),
                            (ql[lo + 32] >> 4) | (((hi >> 6) & 3) << 4),
                        ];
                        for g in 0..4 {
                            y[half * 128 + g * 32 + l] = d
                                * (scales[half * 8 + is + g * 2] as i8 as f32)
                                * (q[g] as i32 - 32) as f32;
                        }
                    }
                }
            }
            42 => {
                let d = h(b, 0);
                for i in 0..64 {
                    y[i] = d * (((b[2 + i / 4] >> (2 * (i % 4))) & 3) as i32 - 1) as f32;
                }
            }
            _ => bail!("scalar reference reader for GGML type {kind} is not implemented"),
        }
    }
    ensure!(
        out.iter().all(|x| x.is_finite()),
        "nonfinite dequantized weights"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn q5_1_preserves_high_bits_and_min() {
        let mut b = vec![0; 24];
        b[..2].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        b[2..4].copy_from_slice(&f16::from_f32(-2.).to_bits().to_le_bytes());
        b[4..8].copy_from_slice(&0x8000_0001u32.to_le_bytes());
        b[8..].fill(0x21);
        let mut y = [0.; 32];
        decode_row(7, &b, &mut y).unwrap();
        assert_eq!(y[0], 6.5);
        assert_eq!(y[1], -1.5);
        assert_eq!(y[16], -1.);
        assert_eq!(y[31], 7.);
    }
    #[test]
    fn iq4_and_linear_q4_are_distinct() {
        let mut b = vec![0; 18];
        b[..2].copy_from_slice(&f16::ONE.to_bits().to_le_bytes());
        b[2..].fill(0xf0);
        let mut y = [0.; 32];
        decode_row(20, &b, &mut y).unwrap();
        assert_eq!(y[0], -127.);
        assert_eq!(y[16], 113.);
    }
    #[test]
    fn q2_0_mtp_grid() {
        let mut b = vec![0; 18];
        b[..2].copy_from_slice(&f16::ONE.to_bits().to_le_bytes());
        b[2..].fill(0b11100100);
        let mut y = [0.; 64];
        decode_row(42, &b, &mut y).unwrap();
        assert_eq!(&y[..8], &[-1., 0., 1., 2., -1., 0., 1., 2.]);
    }
}
