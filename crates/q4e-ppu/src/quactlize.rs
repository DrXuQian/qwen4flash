//! Binding to QUactlize's public dense/packing APIs, without ggml helpers.
use anyhow::{Context, Result, bail, ensure};
use libloading::Library;
use std::{
    ffi::{CStr, c_char, c_void},
    marker::PhantomData,
    path::Path,
    rc::Rc,
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Arrangement {
    pub version: i32,
    pub layout: i32,
    pub bits: i32,
    pub high_bits: i32,
    pub artifact_tile_k: i32,
    pub transport_tile_k: i32,
    pub group_size: i32,
    pub reserved: i32,
    pub mapping_id: u64,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Sizes {
    pub raw_bytes: u64,
    pub low_bytes: u64,
    pub high_bytes: u64,
    pub units_bytes: u64,
}
impl Sizes {
    pub fn packed_bytes(&self) -> Result<u64> {
        self.low_bytes
            .checked_add(self.high_bytes)
            .and_then(|n| n.checked_add(self.units_bytes))
            .context("packed size overflow")
    }
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub version: u32,
    pub size: u32,
    pub qtype: i32,
    pub n: i32,
    pub k: i32,
    pub m: i32,
    pub input_stride: i64,
    pub output_stride: i64,
    pub arrangement: Arrangement,
    pub sizes: Sizes,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PlanInfo {
    pub version: u32,
    pub size: u32,
    pub workspace_bytes: u64,
    pub route: i32,
    pub policy: i32,
    pub split: i32,
    pub compute_type: i32,
    pub config: [c_char; 192],
}
impl Default for PlanInfo {
    fn default() -> Self {
        Self {
            version: 1,
            size: std::mem::size_of::<Self>() as u32,
            workspace_bytes: 0,
            route: 0,
            policy: 0,
            split: 0,
            compute_type: 0,
            config: [0; 192],
        }
    }
}
#[repr(C)]
struct Options {
    version: u32,
    size: u32,
    compute_type: i32,
    reserved: i32,
}
#[derive(Clone, Copy)]
pub enum Compute {
    F16 = 0,
    Bf16 = 1,
}

type Canonical = unsafe extern "C" fn(i32, *mut Arrangement) -> i32;
type GetSizes = unsafe extern "C" fn(i32, i32, i32, i32, *const Arrangement, *mut Sizes) -> i32;
type Supported = unsafe extern "C" fn(*const Params) -> i32;
type Create = unsafe extern "C" fn(i32, *const Options, *mut *mut c_void) -> i32;
type Prepare = unsafe extern "C" fn(
    *mut c_void,
    *const Params,
    *const c_void,
    *const c_void,
    *const c_void,
    *mut *mut c_void,
    *mut PlanInfo,
) -> i32;
type Run =
    unsafe extern "C" fn(*mut c_void, *const f32, *mut f32, *mut c_void, u64, *mut c_void) -> i32;
type Destroy = unsafe extern "C" fn(*mut c_void);
type Error = unsafe extern "C" fn() -> *const c_char;
type Pack = unsafe extern "C" fn(
    *const u8,
    *mut u8,
    *mut u8,
    *mut u8,
    i32,
    i32,
    i32,
    i32,
    *const Arrangement,
    *mut c_void,
) -> i32;

pub struct LibraryApi {
    _library: Library,
    canonical: Canonical,
    sizes: GetSizes,
    supported: Supported,
    create: Create,
    prepare: Prepare,
    run: Run,
    destroy: Destroy,
    error: Error,
    pack: Pack,
    // Native device/context affinity is explicit; no implicit cross-thread sharing.
    _thread_bound: PhantomData<Rc<()>>,
}
impl LibraryApi {
    pub fn load(path: &Path) -> Result<Self> {
        let lib =
            unsafe { Library::new(path) }.with_context(|| format!("load {}", path.display()))?;
        unsafe {
            Ok(Self {
                canonical: *lib.get(b"quactlize_ppu_kpack_canonical_arrangement_v1\0")?,
                sizes: *lib.get(b"quactlize_ppu_kpack_sizes_for_arrangement_v1\0")?,
                supported: *lib.get(b"quactlize_ppu_dense_supported_v1\0")?,
                create: *lib.get(b"quactlize_ppu_dense_create_with_options_v1\0")?,
                prepare: *lib.get(b"quactlize_ppu_dense_prepare_v1\0")?,
                run: *lib.get(b"quactlize_ppu_dense_run_v1\0")?,
                destroy: *lib.get(b"quactlize_ppu_dense_destroy_v1\0")?,
                error: *lib.get(b"quactlize_ppu_dense_error_v1\0")?,
                pack: *lib
                    .get(b"quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2\0")?,
                _library: lib,
                _thread_bound: PhantomData,
            })
        }
    }
    fn check(&self, rc: i32, operation: &str) -> Result<()> {
        if rc != 0 {
            let message = unsafe {
                let p = (self.error)();
                if p.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(p).to_string_lossy().into_owned()
                }
            };
            bail!("QUactlize {operation} failed ({rc}): {message}");
        }
        Ok(())
    }
    /// M=0 admits the whole supported positive M domain; it does not create a launch.
    pub fn params(&self, qtype: i32, n: i32, k: i32, m: i32) -> Result<Params> {
        ensure!(n > 0 && k > 0 && m >= 0, "invalid dense dimensions");
        let mut a = Arrangement::default();
        self.check(
            unsafe { (self.canonical)(qtype, &mut a) },
            "canonical arrangement",
        )?;
        let mut sizes = Sizes::default();
        self.check(
            unsafe { (self.sizes)(n, k, 1, qtype, &a, &mut sizes) },
            "sizes",
        )?;
        let p = Params {
            version: 1,
            size: std::mem::size_of::<Params>() as u32,
            qtype,
            n,
            k,
            m,
            input_stride: k as i64,
            output_stride: n as i64,
            arrangement: a,
            sizes,
        };
        ensure!(
            unsafe { (self.supported)(&p) } == 1,
            "dense shape/format not admitted by installed QUactlize"
        );
        Ok(p)
    }
    /// # Safety
    /// Current device must equal `device`. The caller must retire every queued or
    /// captured operation before dropping this context or any referenced buffers.
    pub unsafe fn context(&self, device: i32, compute: Compute) -> Result<DenseContext<'_>> {
        let options = Options {
            version: 1,
            size: std::mem::size_of::<Options>() as u32,
            compute_type: compute as i32,
            reserved: 0,
        };
        let mut raw = std::ptr::null_mut();
        self.check(
            unsafe { (self.create)(device, &options, &mut raw) },
            "create",
        )?;
        ensure!(!raw.is_null(), "null dense context");
        Ok(DenseContext { api: self, raw })
    }
    /// # Safety
    /// All pointers must be nonoverlapping device spans of the sizes returned for
    /// this exact arrangement and expert count. High may be null only when empty.
    /// Keep buffers alive and order consumers after packing on their streams.
    #[allow(clippy::too_many_arguments)] // Keep the versioned packing ABI visible.
    pub unsafe fn pack(
        &self,
        p: &Params,
        experts: i32,
        raw: *const u8,
        low: *mut u8,
        high: *mut u8,
        units: *mut u8,
        stream: *mut c_void,
    ) -> Result<Sizes> {
        let mut sizes = Sizes::default();
        self.check(
            unsafe { (self.sizes)(p.n, p.k, experts, p.qtype, &p.arrangement, &mut sizes) },
            "expert sizes",
        )?;
        self.check(
            unsafe {
                (self.pack)(
                    raw,
                    low,
                    high,
                    units,
                    p.n,
                    p.k,
                    experts,
                    p.qtype,
                    &p.arrangement,
                    stream,
                )
            },
            "pack",
        )?;
        Ok(sizes)
    }
}
pub struct DenseContext<'a> {
    api: &'a LibraryApi,
    raw: *mut c_void,
}
impl DenseContext<'_> {
    /// # Safety
    /// Weight planes belong to the context's current device and stay immutable
    /// until the context AND all captured/queued uses are retired. Outside capture.
    pub unsafe fn prepare(
        &self,
        p: &Params,
        low: *const c_void,
        high: *const c_void,
        units: *const c_void,
    ) -> Result<DensePlan<'_>> {
        ensure!(p.m > 0, "M=0 is admission only");
        let mut raw = std::ptr::null_mut();
        let mut info = PlanInfo::default();
        self.api.check(
            unsafe { (self.api.prepare)(self.raw, p, low, high, units, &mut raw, &mut info) },
            "prepare",
        )?;
        ensure!(
            !raw.is_null()
                && info.version == 1
                && info.size == std::mem::size_of::<PlanInfo>() as u32,
            "invalid prepared plan ABI"
        );
        Ok(DensePlan {
            context: self,
            raw,
            info,
        })
    }
}
impl Drop for DenseContext<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.destroy)(self.raw) }
    }
}
pub struct DensePlan<'a> {
    context: &'a DenseContext<'a>,
    raw: *mut c_void,
    pub info: PlanInfo,
}
impl DensePlan<'_> {
    /// # Safety
    /// A/C are 16-byte-aligned F32 device spans of the prepared shape/strides.
    /// Workspace is private, 256-byte aligned, and at least the reported size.
    /// All buffers and this context must outlive queued work and captured graphs.
    /// Warm up the installed provider eagerly before capture/timing.
    pub unsafe fn enqueue(
        &self,
        input: *const f32,
        output: *mut f32,
        workspace: *mut c_void,
        bytes: u64,
        stream: *mut c_void,
    ) -> Result<()> {
        ensure!(
            bytes >= self.info.workspace_bytes,
            "insufficient dense workspace"
        );
        self.context.api.check(
            unsafe { (self.context.api.run)(self.raw, input, output, workspace, bytes, stream) },
            "run",
        )
    }
}

const _: () = {
    assert!(std::mem::size_of::<Arrangement>() == 40);
    assert!(std::mem::size_of::<Sizes>() == 32);
    assert!(std::mem::size_of::<Params>() == 112);
    assert!(std::mem::offset_of!(Params, arrangement) == 40);
    assert!(std::mem::offset_of!(Params, sizes) == 80);
    assert!(std::mem::size_of::<PlanInfo>() == 224);
    assert!(std::mem::offset_of!(PlanInfo, config) == 32);
};
