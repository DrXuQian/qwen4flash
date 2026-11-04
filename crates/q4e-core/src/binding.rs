//! Tensor contracts for the pinned target and its separate shared-Q8 MTP GGUF.
use crate::{
    gguf::{ModelIndex, Value},
    model::Config,
};
use anyhow::{Result, ensure};
use std::collections::BTreeMap;
type Shapes = BTreeMap<String, Vec<usize>>;
fn hc(s: &mut Shapes, prefix: &str, c: &Config, inject: bool) {
    s.insert(format!("{prefix}_norm.weight"), vec![c.width * c.hc]);
    s.insert(
        format!("{prefix}_down.weight"),
        vec![c.width * c.hc, c.hc_rank],
    );
    s.insert(
        format!("{prefix}_up.weight"),
        vec![c.hc_rank, c.width * c.hc],
    );
    if inject {
        s.insert(
            format!("{prefix}_inject.weight"),
            vec![c.width * c.hc, c.hc],
        );
    }
}
fn block(s: &mut Shapes, c: &Config, il: usize) {
    hc(s, &format!("blk.{il}.hc_attn"), c, true);
    hc(s, &format!("blk.{il}.hc_ffn"), c, true);
    let mut put = |name: &str, dims: Vec<usize>| {
        s.insert(format!("blk.{il}.{name}"), dims);
    };
    if c.is_attention(il) {
        for (name, k, n) in [
            ("attn_q", c.width, c.attention_heads * 2 * c.head_dim),
            ("attn_k", c.width, c.kv_heads * c.head_dim),
            ("attn_v", c.width, c.kv_heads * c.head_dim),
            ("attn_output", c.attention_heads * c.head_dim, c.width),
            ("indexer.q_proj", c.width, c.index_heads * c.index_dim),
            ("indexer.k_proj", c.width, c.index_dim),
        ] {
            put(&format!("{name}.weight"), vec![k, n]);
        }
        for (name, n) in [
            ("attn_q_norm", c.head_dim),
            ("attn_k_norm", c.head_dim),
            ("indexer.q_norm", c.index_dim),
            ("indexer.k_norm", c.index_dim),
        ] {
            put(&format!("{name}.weight"), vec![n]);
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
            put(&format!("{name}.weight"), vec![k, n]);
        }
        put("ssm_a", vec![c.value_heads]);
        put("ssm_dt.bias", vec![c.value_heads]);
        put("ssm_norm.weight", vec![c.gdn_dim]);
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
        put(&format!("{name}.weight"), vec![k, n, e]);
    }
}
fn mtp(s: &mut Shapes, c: &Config) {
    let il = c.layers;
    block(s, c, il);
    hc(s, &format!("blk.{il}.nextn.hc_head"), c, false);
    for (name, dims) in [
        ("eh_proj", vec![2 * c.width, c.width]),
        ("enorm", vec![c.width]),
        ("hnorm", vec![c.width * c.hc]),
    ] {
        s.insert(format!("blk.{il}.nextn.{name}.weight"), dims);
    }
}
fn check(index: &ModelIndex, shapes: &Shapes) -> Result<()> {
    index.validate()?;
    ensure!(
        index.tensors().count() == shapes.len(),
        "unexpected tensor inventory: expected {}, got {}",
        shapes.len(),
        index.tensors().count()
    );
    for (name, expected) in shapes {
        let (_, t) = index.tensor(name)?;
        let mut actual: Vec<_> = t.dimensions.iter().map(|&n| n as usize).collect();
        let mut expected = expected.clone();
        while actual.len() > 1 && actual.last() == Some(&1) {
            actual.pop();
        }
        while expected.len() > 1 && expected.last() == Some(&1) {
            expected.pop();
        }
        if (name.contains("_norm.") || name.contains("nextn.hnorm.")) && expected.len() == 1 {
            actual = vec![actual.iter().product()];
        }
        ensure!(
            actual == expected,
            "{name}: shape {actual:?}, expected {expected:?}"
        );
        ensure!(
            [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 20, 30].contains(&t.kind),
            "{name}: unsupported scalar/native preparation format {}",
            t.kind
        );
    }
    Ok(())
}
pub fn validate_target(index: &ModelIndex, c: &Config) -> Result<()> {
    let mut s = Shapes::new();
    s.insert("token_embd.weight".into(), vec![c.width, c.vocab]);
    s.insert("output.weight".into(), vec![c.width, c.vocab]);
    hc(&mut s, "output_hc", c, false);
    for il in 0..c.layers {
        block(&mut s, c, il);
    }
    if let Some(p) = &c.ple {
        let rows = p
            .offsets
            .iter()
            .zip(&p.vocabs)
            .map(|(&a, &b)| a + b)
            .max()
            .unwrap_or(0) as usize;
        let (_, table) = index.tensor("per_layer_token_embd.weight")?;
        ensure!(
            table.dimensions.len() == 2 && table.dimensions[1] >= rows as u64,
            "PLE table is too short for its hash ranges"
        );
        // The pinned GGUF pads the final vocabulary extent to 320001536 rows.
        s.insert(
            "per_layer_token_embd.weight".into(),
            vec![p.head_dim, table.dimensions[1] as usize],
        );
        for &il in &p.layers {
            for (name, dims) in [
                ("ple_key", vec![c.width, c.width * c.hc]),
                ("ple_value", vec![c.width, c.width]),
                ("ple_norm_key", vec![c.width * c.hc]),
                ("ple_norm_query", vec![c.width * c.hc]),
                ("ple_norm_conv", vec![c.width * c.hc]),
                ("ple_conv1d", vec![p.conv_kernel, c.width * c.hc]),
            ] {
                s.insert(format!("blk.{il}.{name}.weight"), dims);
            }
        }
    }
    if c.embedded_mtp_layers == 1 {
        mtp(&mut s, c);
    }
    check(index, &s)
}
pub fn validate_shared_mtp(index: &ModelIndex, target: &Config) -> Result<()> {
    ensure!(
        index.metadata().get("qwen4exp.nextn_shared_target_tensors") == Some(&Value::Bool(true)),
        "MTP must explicitly declare shared target tensors"
    );
    let c = Config::from_index(index)?;
    let mut expected = target.clone();
    expected.ple = None;
    expected.embedded_mtp_layers = 1;
    ensure!(c == expected, "MTP geometry does not match its target");
    let mut s = Shapes::new();
    mtp(&mut s, target);
    check(index, &s)
}
