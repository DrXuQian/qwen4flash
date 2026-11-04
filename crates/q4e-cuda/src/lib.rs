//! Synchronous CUDA proxy for the SAME Rust graph and MTP state machine.
//! Host state and small elementwise work remain explicit. This is not a serving backend.
use anyhow::{Context, Result, ensure};
use libloading::Library;
use q4e_core::{
    ops::{Attention, DeltaStep, OperatorStats, Operators},
    reference::Tensor,
};
use std::{
    cell::RefCell,
    collections::HashMap,
    ffi::{CStr, c_char, c_void},
    path::Path,
    ptr,
};

type Create = unsafe extern "C" fn(i32, u64, *mut *mut c_void) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void);
type Error = unsafe extern "C" fn() -> *const c_char;
type Upload = unsafe extern "C" fn(
    *mut c_void,
    i32,
    i32,
    i32,
    i32,
    *const c_void,
    u64,
    *mut *mut c_void,
) -> i32;
type Matmul = unsafe extern "C" fn(*mut c_void, *mut c_void, i32, i32, *const f32, *mut f32) -> i32;
type Attend = unsafe extern "C" fn(
    *mut c_void,
    *const f32,
    *const f32,
    *const f32,
    i32,
    i32,
    i32,
    i32,
    *mut f32,
) -> i32;
type Delta = unsafe extern "C" fn(
    *mut c_void,
    *const f32,
    *const f32,
    *const f32,
    i32,
    i32,
    i32,
    *mut f32,
    *mut f32,
) -> i32;
type Stats = unsafe extern "C" fn(*mut c_void, *mut u64) -> i32;
struct Cached {
    handle: *mut c_void,
    identity: (usize, usize, u32, Vec<usize>),
    _owner: Tensor,
}

pub struct CudaProxy {
    _library: Library,
    context: *mut c_void,
    destroy: Destroy,
    error: Error,
    upload: Upload,
    matmul: Matmul,
    attend: Attend,
    delta: Delta,
    stats: Stats,
    cache: RefCell<HashMap<String, Cached>>,
    pub features: u32,
}
fn dim(n: usize) -> Result<i32> {
    i32::try_from(n).context("dimension exceeds CUDA proxy ABI")
}
impl CudaProxy {
    pub fn load(path: &Path, device: i32, budget: u64) -> Result<Self> {
        let lib = unsafe { Library::new(path) }
            .with_context(|| format!("load CUDA proxy {}", path.display()))?;
        unsafe {
            let abi: unsafe extern "C" fn() -> u32 = *lib.get(b"q4e_cuda_abi_v1\0")?;
            ensure!(abi() == 1, "CUDA proxy ABI mismatch");
            let query: unsafe extern "C" fn() -> u32 = *lib.get(b"q4e_cuda_features_v1\0")?;
            let features = query();
            ensure!(
                features & 7 == 7,
                "CUDA proxy is missing required GPU operators"
            );
            let create: Create = *lib.get(b"q4e_cuda_create_v1\0")?;
            let error: Error = *lib.get(b"q4e_cuda_error_v1\0")?;
            // Resolve every symbol before allocating the native context.
            let destroy: Destroy = *lib.get(b"q4e_cuda_destroy_v1\0")?;
            let upload: Upload = *lib.get(b"q4e_cuda_upload_v1\0")?;
            let matmul: Matmul = *lib.get(b"q4e_cuda_matmul_v1\0")?;
            let attend: Attend = *lib.get(b"q4e_cuda_attention_v1\0")?;
            let delta: Delta = *lib.get(b"q4e_cuda_delta_v1\0")?;
            let stats: Stats = *lib.get(b"q4e_cuda_stats_v1\0")?;
            let mut context = ptr::null_mut();
            let rc = create(device, budget, &mut context);
            ensure!(
                rc == 0 && !context.is_null(),
                "CUDA proxy create: {}",
                CStr::from_ptr(error()).to_string_lossy()
            );
            Ok(Self {
                _library: lib,
                context,
                destroy,
                error,
                upload,
                matmul,
                attend,
                delta,
                stats,
                cache: RefCell::new(HashMap::new()),
                features,
            })
        }
    }
    fn check(&self, rc: i32) -> Result<()> {
        ensure!(rc == 0, "CUDA proxy: {}", unsafe {
            CStr::from_ptr((self.error)()).to_string_lossy()
        });
        Ok(())
    }
    pub fn matmul(
        &self,
        name: &str,
        tensor: &Tensor,
        expert: usize,
        m: usize,
        input: &[f32],
    ) -> Result<Vec<f32>> {
        ensure!(
            !tensor.dims.is_empty() && tensor.dims.len() <= 3 && m > 0,
            "invalid CUDA tensor rank/M"
        );
        let k = tensor.dims[0];
        let n = tensor.dims.get(1).copied().unwrap_or(1);
        let experts = tensor.dims.get(2).copied().unwrap_or(1);
        ensure!(
            expert < experts && k.checked_mul(m) == Some(input.len()),
            "CUDA matmul input/expert mismatch"
        );
        let (kind, bytes) = tensor.storage();
        let identity = (
            bytes.as_ptr() as usize,
            bytes.len(),
            kind,
            tensor.dims.clone(),
        );
        let mut cache = self.cache.borrow_mut();
        if let Some(c) = cache.get(name) {
            ensure!(
                c.identity == identity,
                "immutable weight identity changed for {name}"
            );
        } else {
            let mut handle = ptr::null_mut();
            self.check(unsafe {
                (self.upload)(
                    self.context,
                    kind as i32,
                    dim(n)?,
                    dim(k)?,
                    dim(experts)?,
                    bytes.as_ptr().cast(),
                    bytes.len() as u64,
                    &mut handle,
                )
            })?;
            ensure!(!handle.is_null(), "null CUDA weight");
            cache.insert(
                name.into(),
                Cached {
                    handle,
                    identity,
                    _owner: tensor.clone(),
                },
            );
        }
        let mut output = vec![0.; n.checked_mul(m).context("output overflow")?];
        self.check(unsafe {
            (self.matmul)(
                self.context,
                cache[name].handle,
                dim(expert)?,
                dim(m)?,
                input.as_ptr(),
                output.as_mut_ptr(),
            )
        })?;
        ensure!(
            output.iter().all(|x| x.is_finite()),
            "nonfinite CUDA matmul output"
        );
        Ok(output)
    }
    pub fn counters(&self) -> Result<OperatorStats> {
        let mut counts = [0u64; 5];
        self.check(unsafe { (self.stats)(self.context, counts.as_mut_ptr()) })?;
        Ok(OperatorStats {
            matmuls: counts[0],
            attention: counts[1],
            gdn: counts[2],
            resident_weight_bytes: counts[3],
            weight_uploads: counts[4],
        })
    }
}
impl Drop for CudaProxy {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.context) }
    }
}
impl Operators for CudaProxy {
    fn name(&self) -> &str {
        "cuda_proxy"
    }
    fn matvec(&self, name: &str, t: &Tensor, e: usize, x: &[f32]) -> Result<Vec<f32>> {
        self.matmul(name, t, e, 1, x)
    }
    fn delta(&self, p: &DeltaStep<'_>, state: &mut [f32]) -> Result<Vec<f32>> {
        ensure!(
            p.dim > 0
                && p.dim <= 256
                && p.key_heads > 0
                && p.value_heads > 0
                && p.value_heads % p.key_heads == 0,
            "invalid proxy GDN geometry"
        );
        ensure!(
            p.qkv.len() == (2 * p.key_heads + p.value_heads) * p.dim
                && p.decay.len() == p.value_heads
                && p.beta.len() == p.value_heads
                && state.len() == p.value_heads * p.dim * p.dim,
            "GDN buffer extent mismatch"
        );
        let mut output = vec![0.; p.value_heads * p.dim];
        self.check(unsafe {
            (self.delta)(
                self.context,
                p.qkv.as_ptr(),
                p.decay.as_ptr(),
                p.beta.as_ptr(),
                dim(p.key_heads)?,
                dim(p.value_heads)?,
                dim(p.dim)?,
                state.as_mut_ptr(),
                output.as_mut_ptr(),
            )
        })?;
        ensure!(
            output.iter().chain(state.iter()).all(|x| x.is_finite()),
            "nonfinite proxy GDN result"
        );
        Ok(output)
    }
    fn attention(&self, p: &Attention<'_>) -> Result<Vec<f32>> {
        ensure!(
            p.dim > 0
                && p.dim <= 256
                && p.kv_heads > 0
                && p.heads % p.kv_heads == 0
                && p.query.len() == p.heads * p.dim
                && !p.selected.is_empty(),
            "invalid proxy attention geometry"
        );
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for &i in p.selected {
            ensure!(
                i < p.keys.len()
                    && i < p.values.len()
                    && p.keys[i].len() == p.kv_heads * p.dim
                    && p.values[i].len() == p.kv_heads * p.dim,
                "invalid selected KV row"
            );
            keys.extend_from_slice(&p.keys[i]);
            values.extend_from_slice(&p.values[i]);
        }
        let mut output = vec![0.; p.heads * p.dim];
        self.check(unsafe {
            (self.attend)(
                self.context,
                p.query.as_ptr(),
                keys.as_ptr(),
                values.as_ptr(),
                dim(p.selected.len())?,
                dim(p.heads)?,
                dim(p.kv_heads)?,
                dim(p.dim)?,
                output.as_mut_ptr(),
            )
        })?;
        ensure!(
            output.iter().all(|x| x.is_finite()),
            "nonfinite CUDA attention output"
        );
        Ok(output)
    }
    fn stats(&self) -> OperatorStats {
        self.counters().expect("live CUDA proxy context")
    }
}
