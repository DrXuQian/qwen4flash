use half::f16;
use q4e_core::{
    model::Config,
    mtp::{Backend, Session, argmax},
    ops::{Attention, DeltaStep, Operators, scalar_attention, scalar_delta},
    reference::{Reference, State, Tensor, tiny_weights},
};
use q4e_cuda::CudaProxy;
fn proxy() -> CudaProxy {
    CudaProxy::load(
        std::path::Path::new(
            &std::env::var("Q4E_CUDA_PROXY_LIB").expect("set the compiled CUDA proxy library path"),
        ),
        0,
        1 << 30,
    )
    .unwrap()
}
fn close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        assert!(
            (x - y).abs() <= tol * (1. + y.abs()),
            "at {i}: {x} versus {y}"
        );
    }
}

#[test]
#[ignore = "requires a real CUDA device and Q4E_CUDA_PROXY_LIB"]
fn same_rust_graph_and_mtp_transaction_run_on_cuda() {
    let c = Config::tiny();
    let w = tiny_weights(&c);
    let prompt = [1, 3, 2, 4, 7];
    let mut cpu = Reference::new(c.clone(), w.clone(), 64).unwrap();
    let mut cpu_state = State::new(&c);
    let mut gpu = Reference::new(c.clone(), w.clone(), 64)
        .unwrap()
        .with_operators(Box::new(proxy()));
    let mut gpu_state = State::new(&c);
    let mut last = None;
    for t in prompt {
        let a = cpu.target(&mut cpu_state, t).unwrap();
        let b = gpu.target(&mut gpu_state, t).unwrap();
        close(&b.logits, &a.logits, 3e-4);
        close(&b.hidden, &a.hidden, 3e-4);
        last = Some(b);
    }
    let mut ordinary = Vec::new();
    for _ in 0..17 {
        let t = argmax(&last.as_ref().unwrap().logits).unwrap();
        ordinary.push(t);
        last = Some(gpu.target(&mut gpu_state, t).unwrap());
    }
    let model = Reference::new(c.clone(), w, 64)
        .unwrap()
        .with_operators(Box::new(proxy()));
    let mut mtp = Session::new(model, State::new(&c), State::new_draft(&c), 3, None).unwrap();
    mtp.prefill(&prompt).unwrap();
    assert_eq!(mtp.generate(17).unwrap(), ordinary);
    assert_eq!(mtp.target, gpu_state);
    let draft_attention = &mtp.draft.attention[c.layers];
    assert!(draft_attention.keys.len() > c.index_top_k + c.index_block);
    assert!(draft_attention.pooled.is_empty() && draft_attention.raw_tail.is_empty());
    let stats = mtp.backend.operators.stats();
    assert!(stats.matmuls > 0);
    assert!(stats.attention > 0 && stats.gdn > 0);
    assert!(mtp.stats.proposed > 0);
}

#[test]
#[ignore = "requires a real CUDA device and Q4E_CUDA_PROXY_LIB"]
fn raw_gguf_format_expert_strides_and_variable_m() {
    let gpu = proxy();
    for kind in [2, 3, 6, 7, 8, 12, 13, 14, 20, 42] {
        let (block, bytes) = q4e_core::gguf::type_layout(kind).unwrap();
        let k = 256usize;
        let n = 7;
        let experts = 3;
        let mut raw = vec![0u8; n * k * experts / block as usize * bytes as usize];
        for (i, b) in raw.iter_mut().enumerate() {
            *b = (i.wrapping_mul(73).wrapping_add(19) % 256) as u8;
        }
        for b in raw.chunks_exact_mut(bytes as usize) {
            let d = f16::from_f32(0.00390625).to_bits().to_le_bytes();
            if kind == 14 {
                b[208..210].copy_from_slice(&d);
            } else {
                b[..2].copy_from_slice(&d);
                if [3, 7, 12, 13].contains(&kind) {
                    b[2..4].copy_from_slice(&d);
                }
            }
        }
        let t = Tensor::encoded(vec![k, n, experts], kind, raw).unwrap();
        for m in [1, 4, 17] {
            let input: Vec<_> = (0..m * k)
                .map(|i| ((i * 17 % 101) as f32 - 50.) / 127.)
                .collect();
            for e in 0..experts {
                let out = gpu
                    .matmul(&format!("kind-{kind}"), &t, e, m, &input)
                    .unwrap();
                for row in 0..m {
                    close(
                        &out[row * n..(row + 1) * n],
                        &t.mm(&input[row * k..(row + 1) * k], e).unwrap(),
                        3e-4,
                    );
                }
            }
        }
    }
    let x = Tensor::owned(vec![8, 8], vec![1.; 64]).unwrap();
    gpu.matmul("immutable", &x, 0, 1, &[1.; 8]).unwrap();
    let y = Tensor::owned(vec![8, 8], vec![2.; 64]).unwrap();
    assert!(gpu.matmul("immutable", &y, 0, 1, &[1.; 8]).is_err());
}

#[test]
#[ignore = "requires a real CUDA device and Q4E_CUDA_PROXY_LIB"]
fn real_model_gdn_heads_and_selected_attention() {
    let gpu = proxy();
    let (hk, hv, d) = (16, 48, 128);
    let mut qkv: Vec<_> = (0..(2 * hk + hv) * d)
        .map(|i| ((i * 13 % 101) as f32 - 50.) / 200.)
        .collect();
    for h in 0..2 * hk {
        let norm = qkv[h * d..(h + 1) * d]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        for x in &mut qkv[h * d..(h + 1) * d] {
            *x /= norm;
        }
    }
    let decay = vec![0.9; hv];
    let beta = vec![0.6; hv];
    let p = DeltaStep {
        qkv: &qkv,
        decay: &decay,
        beta: &beta,
        key_heads: hk,
        value_heads: hv,
        dim: d,
    };
    let mut a = vec![0.02; hv * d * d];
    let mut b = a.clone();
    for _ in 0..5 {
        close(
            &gpu.delta(&p, &mut a).unwrap(),
            &scalar_delta(&p, &mut b).unwrap(),
            1e-5,
        );
    }
    close(&a, &b, 1e-5);
    let d = 256;
    let q: Vec<_> = (0..24 * d).map(|i| (i % 31) as f32 / 31.).collect();
    let k: Vec<Vec<_>> = (0..29)
        .map(|t| {
            (0..2 * d)
                .map(|i| ((i + t * 3) % 23) as f32 / 23.)
                .collect()
        })
        .collect();
    let v = k.clone();
    let selected = [0, 1, 2, 3, 16, 17, 18, 19, 28];
    let p = Attention {
        query: &q,
        keys: &k,
        values: &v,
        selected: &selected,
        heads: 24,
        kv_heads: 2,
        dim: d,
    };
    close(
        &gpu.attention(&p).unwrap(),
        &scalar_attention(&p).unwrap(),
        1e-5,
    );
}
