//! Narrow, versioned PPU operator ABIs. Preparation and enqueue are separate.
pub mod gdn;
pub mod quactlize;

use anyhow::{Context, Result, ensure};
use libloading::{Library, Symbol};
use std::path::Path;

/// Load an explicitly selected, trusted SDK runtime and ask it about devices.
/// This does not imply that the full model has a native execution backend.
pub fn probe_runtime(path: &Path) -> Result<(i32, i32)> {
    // Loading a native library executes its initialization routines.
    let lib = unsafe { Library::new(path) }.with_context(|| format!("load {}", path.display()))?;
    unsafe {
        let version: Symbol<unsafe extern "C" fn(*mut i32) -> i32> =
            lib.get(b"cudaRuntimeGetVersion\0")?;
        let count: Symbol<unsafe extern "C" fn(*mut i32) -> i32> =
            lib.get(b"cudaGetDeviceCount\0")?;
        let (mut v, mut n) = (0, 0);
        let rc = version(&mut v);
        ensure!(rc == 0, "runtime version query failed: {rc}");
        let rc = count(&mut n);
        ensure!(rc == 0, "device query failed: {rc}");
        Ok((v, n))
    }
}
