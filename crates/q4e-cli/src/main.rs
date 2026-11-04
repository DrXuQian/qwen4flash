use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use q4e_core::{
    gguf::ModelIndex,
    model::Config,
    mtp::Session,
    plan::{Options, PreparedPlan},
    reference::{Reference, State, Weights, tiny_weights},
};
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Parser)]
#[command(
    version,
    about = "Qwen4exp PPU bring-up: GGUF inspection, placement, learned MTP reference and native ABI probes"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(clap::Args)]
struct CudaOptions {
    /// Explicit synchronous CUDA validation backend. State/elementwise work stays on the host.
    #[arg(long)]
    cuda_proxy_lib: Option<PathBuf>,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 88)]
    weight_budget_gib: u64,
}
#[derive(Subcommand)]
enum Command {
    /// Read GGUF descriptors across all shards, without mapping weight payloads.
    Inspect { model: PathBuf },
    /// Inspect the metadata-only JSON produced by inspect_remote_gguf.py.
    InspectIndex { index: PathBuf },
    /// Create a placement/operator plan. This does not claim native admission.
    Prepare {
        model: PathBuf,
        #[arg(long)]
        metadata_only: bool,
        #[arg(long)]
        mtp: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 32768)]
        context: usize,
        #[arg(long, default_value_t = 1)]
        concurrency: usize,
        #[arg(long, default_value_t = 3)]
        draft_tokens: usize,
        #[arg(long, default_value_t = 64)]
        ram_gib: u64,
        #[arg(long, default_value_t = 96)]
        vram_gib: u64,
    },
    /// Execute a miniature Qwen4exp with GDN, QSA, PLE, MoE and learned MTP.
    Demo {
        #[arg(long, default_value = "1,3,2,4,7")]
        tokens: String,
        #[arg(long, default_value_t = 16)]
        generate: usize,
        #[command(flatten)]
        cuda: CudaOptions,
    },
    /// Slow scalar model oracle. Input/output are token IDs, not a serving API.
    Reference {
        model: PathBuf,
        #[arg(long)]
        mtp: Option<PathBuf>,
        #[arg(long)]
        tokens: String,
        #[arg(long, default_value_t = 8)]
        generate: usize,
        #[arg(long, default_value_t = 128)]
        context: usize,
        #[command(flatten)]
        cuda: CudaOptions,
    },
    /// Probe an explicitly selected trusted SDK runtime and optional QUactlize DSO.
    Probe {
        #[arg(long)]
        runtime: PathBuf,
        #[arg(long)]
        quactlize: Option<PathBuf>,
    },
}
fn read_index(path: &std::path::Path) -> Result<ModelIndex> {
    let i: ModelIndex = serde_json::from_reader(std::fs::File::open(path)?)?;
    i.validate()?;
    Ok(i)
}
fn inspect(index: ModelIndex) -> Result<()> {
    index.validate()?;
    let config = Config::from_index(&index)?;
    let mut formats = BTreeMap::<String, u64>::new();
    let (mut total, mut ple, mut experts) = (0u64, 0u64, 0u64);
    for (_, t) in index.tensors() {
        total += t.bytes;
        if t.is_ple_table() {
            ple += t.bytes;
        }
        if t.name.contains("_exps.") {
            experts += t.bytes;
        }
        *formats
            .entry(q4e_core::gguf::type_name(t.kind))
            .or_default() += t.bytes;
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"config":config,"shards":index.shards.len(),"tensors":index.tensors().count(),"tensor_bytes":total,"ple_bytes":ple,"expert_bytes":experts,"other_bytes":total-ple-experts,"format_bytes":formats})
        )?
    );
    Ok(())
}
fn tokens(s: &str) -> Result<Vec<u32>> {
    let t = s
        .split(',')
        .map(|x| {
            x.trim()
                .parse::<u32>()
                .context("token IDs must be comma-separated integers")
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(!t.is_empty(), "empty token input");
    Ok(t)
}
fn generate(
    config: Config,
    weights: Weights,
    input: &str,
    count: usize,
    context: usize,
    cuda: &CudaOptions,
) -> Result<()> {
    let input = tokens(input)?;
    ensure!(
        input.len().checked_add(count).is_some_and(|n| n <= context),
        "requested tokens exceed prepared context"
    );
    let mut model = Reference::new(config.clone(), weights, context)?;
    if let Some(path) = &cuda.cuda_proxy_lib {
        let budget = cuda
            .weight_budget_gib
            .checked_mul(1 << 30)
            .context("weight budget overflow")?;
        model = model.with_operators(Box::new(q4e_cuda::CudaProxy::load(
            path,
            cuda.device,
            budget,
        )?));
    }
    let mut s = Session::new(
        model,
        State::new(&config),
        State::new_draft(&config),
        3,
        None,
    )?;
    s.prefill(&input)?;
    let out = s.generate(count)?;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"backend":s.backend.operators.name(),"mtp":"learned_greedy","tokens":out,"stats":s.stats,"operators":s.backend.operators.stats(),"host_state":true,"acceptance":s.stats.acceptance(),"native_ppu_execution":false})
        )?
    );
    Ok(())
}
fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Inspect { model } => inspect(ModelIndex::open(&model)?),
        Command::InspectIndex { index } => inspect(read_index(&index)?),
        Command::Prepare {
            model,
            metadata_only,
            mtp,
            out,
            context,
            concurrency,
            draft_tokens,
            ram_gib,
            vram_gib,
        } => {
            let index = if metadata_only {
                read_index(&model)?
            } else {
                ModelIndex::open(&model)?
            };
            let mtp = mtp
                .map(|p| {
                    if metadata_only {
                        read_index(&p)
                    } else {
                        ModelIndex::open(&p)
                    }
                })
                .transpose()?;
            let options = Options {
                context,
                concurrency,
                draft_tokens,
                host_budget_bytes: ram_gib
                    .checked_mul(1 << 30)
                    .context("RAM capacity overflow")?,
                device_budget_bytes: vram_gib
                    .checked_mul(1 << 30)
                    .context("VRAM capacity overflow")?,
                ..Options::default()
            };
            let plan = PreparedPlan::build(&index, mtp.as_ref(), options)?;
            if let Some(parent) = out.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent)?;
                }
            }
            std::fs::write(&out, serde_json::to_vec_pretty(&plan)?)?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &serde_json::json!({"plan":out,"memory":plan.memory,"blocking":plan.blocking,"device_execution_admitted":false})
                )?
            );
            Ok(())
        }
        Command::Demo {
            tokens,
            generate: count,
            cuda,
        } => {
            let c = Config::tiny();
            let w = tiny_weights(&c);
            generate(c, w, &tokens, count, 256, &cuda)
        }
        Command::Reference {
            model,
            mtp,
            tokens,
            generate: count,
            context,
            cuda,
        } => {
            let index = ModelIndex::open(&model)?;
            let c = Config::from_index(&index)?;
            q4e_core::binding::validate_target(&index, &c)?;
            ensure!(
                c.embedded_mtp_layers == 1 || mtp.is_some(),
                "learned MTP weights are required: pass --mtp with the shared-Q8 sidecar"
            );
            let mut w = Weights::open(&index)?;
            if let Some(path) = mtp {
                ensure!(
                    c.embedded_mtp_layers == 0,
                    "target already contains MTP weights"
                );
                w.attach_shared_mtp(&ModelIndex::open(&path)?, &c)?;
            }
            generate(c, w, &tokens, count, context, &cuda)
        }
        Command::Probe { runtime, quactlize } => {
            let (version, devices) = q4e_ppu::probe_runtime(&runtime)?;
            println!("runtime_version={version} devices={devices}");
            if let Some(path) = quactlize {
                let api = q4e_ppu::quactlize::LibraryApi::load(&path)?;
                for (q, n, k) in [(8, 2560, 2560), (12, 1280, 2560), (13, 1280, 2560)] {
                    let p = api.params(q, n, k, 0)?;
                    println!(
                        "qtype={q} N={n} K={k} packed_bytes={} arrangement={:?}",
                        p.sizes.packed_bytes()?,
                        p.arrangement
                    );
                }
            }
            Ok(())
        }
    }
}
