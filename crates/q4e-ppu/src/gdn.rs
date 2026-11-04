//! Native actlizeLA WY prefill binding. No Torch or Python runtime dependency.
use anyhow::{Context, Result, ensure};
use libloading::Library;
use std::{ffi::c_void, path::Path};
type Forward = unsafe extern "C" fn(
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const c_void,
    *const f32,
    *mut c_void,
    *mut f32,
    *mut c_void,
    *mut c_void,
    *mut c_void,
    *mut f32,
    i32,
    i32,
    i32,
    i32,
    bool,
    *mut c_void,
) -> i32;
pub struct GdnWy {
    _library: Library,
    forward: Forward,
}
#[derive(Debug)]
pub struct Workspace {
    pub w_bytes: usize,
    pub snapshots_bytes: usize,
    pub vn_bytes: usize,
    pub gates_bytes: usize,
}
pub struct Buffers {
    pub q: *const c_void,
    pub k: *const c_void,
    pub v: *const c_void,
    pub log_gate: *const c_void,
    pub beta: *const c_void,
    pub initial: *const f32,
    pub output: *mut c_void,
    pub final_state: *mut f32,
    pub w: *mut c_void,
    pub snapshots: *mut c_void,
    pub vn: *mut c_void,
    pub gates: *mut f32,
}
impl GdnWy {
    pub fn load(path: &Path) -> Result<Self> {
        let lib = unsafe { Library::new(path) }.context("load native libgdn_wy_ppu")?;
        let forward = unsafe { *lib.get(b"gdn_wy_forward_residual_full_chunk\0")? };
        Ok(Self {
            _library: lib,
            forward,
        })
    }
    pub fn workspace(batch: usize, tokens: usize, hv: usize) -> Result<Workspace> {
        ensure!(batch > 0 && tokens > 0 && hv > 0, "zero GDN dimension");
        let groups = batch
            .checked_mul(hv)
            .and_then(|n| n.checked_mul(tokens.div_ceil(64)))
            .context("GDN workspace overflow")?;
        let bytes = |n| groups.checked_mul(n).context("GDN workspace overflow");
        Ok(Workspace {
            w_bytes: bytes(128 * 128 * 2)?,
            snapshots_bytes: bytes(128 * 128 * 2)?,
            vn_bytes: bytes(64 * 128 * 2)?,
            gates_bytes: bytes(64 * 4)?,
        })
    }
    /// # Safety
    /// D=128. Contiguous Q/K/V/beta/output are BF16; gates use the explicit dtype,
    /// natural-log decay and provider scaling. Initial/final KxV state is F32.
    /// Buffers must match provider head order and have the queried workspace size.
    /// Qwen's `vh % Hk` assignment MUST be permuted/expanded for a provider using
    /// contiguous groups; this ABI does not silently reinterpret that assignment.
    /// Device pointers, library and workspaces must outlive enqueued stream work.
    #[allow(clippy::too_many_arguments)] // Match the native launch geometry and stream.
    pub unsafe fn enqueue(
        &self,
        b: &Buffers,
        batch: i32,
        tokens: i32,
        hk: i32,
        hv: i32,
        gate_f32: bool,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            batch > 0 && tokens > 0 && hk > 0 && hv > 0 && hv % hk == 0,
            "invalid GDN geometry"
        );
        let rc = unsafe {
            (self.forward)(
                b.q,
                b.k,
                b.v,
                b.log_gate,
                b.beta,
                b.initial,
                b.output,
                b.final_state,
                b.w,
                b.snapshots,
                b.vn,
                b.gates,
                batch,
                tokens,
                hk,
                hv,
                gate_f32,
                stream,
            )
        };
        ensure!(rc == 0, "actlizeLA GDN launch failed: {rc}");
        Ok(())
    }
}
