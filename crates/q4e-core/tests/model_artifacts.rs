use q4e_core::{
    binding::{validate_shared_mtp, validate_target},
    gguf::{ModelIndex, Value},
    model::Config,
    plan::{Options, PreparedPlan},
};
fn target() -> ModelIndex {
    serde_json::from_str(include_str!("../../../fixtures/target-index.json")).unwrap()
}
fn mtp() -> ModelIndex {
    serde_json::from_str(include_str!("../../../fixtures/mtp-index.json")).unwrap()
}

#[test]
fn pinned_shards_and_shared_mtp_match_the_real_tensor_contract() {
    let t = target();
    let m = mtp();
    let c = Config::from_index(&t).unwrap();
    validate_target(&t, &c).unwrap();
    validate_shared_mtp(&m, &c).unwrap();
    assert_eq!((t.tensors().count(), m.tensors().count()), (1224, 32));
    assert!(m.tensor("token_embd.weight").is_err());
    assert!(m.tensor("output.weight").is_err());
    let p = PreparedPlan::build(&t, Some(&m), Options::default()).unwrap();
    assert_eq!(p.memory.ple_bytes, 28_800_138_240);
    assert_eq!(p.memory.expert_bytes, 77_017_907_200);
    assert_eq!(p.memory.mtp_weight_bytes, Some(2_775_621_632));
    assert_eq!(p.memory.host_planned_bytes, 42_221_911_040);
    assert!(p.memory.host_planned_bytes < 64 << 30);
    assert!(p.memory.planned_device_with_workspace < 96 << 30);
    assert!(!p.device_execution_admitted);
    assert!(!p.blocking.is_empty());
}
#[test]
fn shared_model_mismatch_is_rejected_before_any_payload_is_mapped() {
    let c = Config::from_index(&target()).unwrap();
    let mut m = mtp();
    m.shards[0]
        .header
        .metadata
        .insert("qwen4exp.hyper_connection.count".into(), Value::Unsigned(8));
    assert!(validate_shared_mtp(&m, &c).is_err());
    let mut m = mtp();
    m.shards[0]
        .header
        .metadata
        .remove("qwen4exp.nextn_shared_target_tensors");
    assert!(validate_shared_mtp(&m, &c).is_err());
}
#[test]
fn incomplete_shards_and_forged_index_bytes_fail_validation() {
    let mut t = target();
    t.shards.pop();
    assert!(t.validate().is_err());
    let mut t = target();
    t.shards[1].header.tensors[0].bytes += 32;
    assert!(t.validate().is_err());
}
#[test]
fn missing_mtp_and_insufficient_ram_remain_explicit() {
    let p = PreparedPlan::build(
        &target(),
        None,
        Options {
            host_budget_bytes: 32 << 30,
            ..Options::default()
        },
    )
    .unwrap();
    assert!(p.blocking.iter().any(|s| s.contains("MTP weights")));
    assert!(p.blocking.iter().any(|s| s.contains("host RAM")));
    assert!(!p.device_execution_admitted);
}
