//! Bounded host staging and explicitly resident, still-quantized PLE storage.
use crate::gguf::{ModelIndex, Shard, TensorInfo, type_layout};
use anyhow::{Context, Result, ensure};
use memmap2::{Mmap, MmapOptions};
use std::{fs::File, os::fd::AsRawFd, os::unix::fs::FileExt};

pub struct ResidentPle {
    map: Mmap,
    pub rows: u64,
    pub row_bytes: usize,
    pub kind: u32,
}
impl ResidentPle {
    /// Requires an immutable model file and enough lockable physical RAM.
    /// Failure is explicit: a pageable mapping is not a performance-equivalent fallback.
    pub fn open(index: &ModelIndex, budget: u64) -> Result<Self> {
        let (s, t) = index.tensor("per_layer_token_embd.weight")?;
        ensure!(
            t.dimensions.len() == 2 && t.bytes <= budget,
            "PLE table exceeds resident budget or has invalid rank"
        );
        let f = File::open(&s.path)?;
        ensure!(
            f.metadata()?.len() == s.header.file_bytes,
            "model file changed"
        );
        let offset = s
            .header
            .data_offset
            .checked_add(t.offset)
            .context("PLE offset overflow")?;
        let len = usize::try_from(t.bytes).context("PLE exceeds address space")?;
        // File is an immutable artifact. The mapping preserves quantized bytes.
        let map = unsafe { MmapOptions::new().offset(offset).len(len).map(&f) }?;
        let rc = unsafe { libc::mlock(map.as_ptr().cast(), map.len()) };
        ensure!(
            rc == 0,
            "PLE could not be made resident: {} (check RAM and memlock limit)",
            std::io::Error::last_os_error()
        );
        // mlock faults the mapped pages in; no extra 26.8 GiB heap copy is made.
        let (block, bytes) = type_layout(t.kind)?;
        Ok(Self {
            map,
            rows: t.dimensions[1],
            row_bytes: usize::try_from(t.dimensions[0] / block * bytes)?,
            kind: t.kind,
        })
    }
    pub fn row(&self, row: u64) -> Result<&[u8]> {
        ensure!(row < self.rows, "PLE row out of range");
        let start = row as usize * self.row_bytes;
        Ok(&self.map[start..start + self.row_bytes])
    }
    pub fn bytes(&self) -> usize {
        self.map.len()
    }
}
impl Drop for ResidentPle {
    fn drop(&mut self) {
        unsafe {
            libc::munlock(self.map.as_ptr().cast(), self.map.len());
        }
    }
}

/// Read one tensor through a bounded, quant-block-aligned staging buffer.
/// The consumer must finish using each slice (including DMA) before returning.
/// This intentionally retains no CPU expert mirror. An async implementation
/// should own a fixed ring of equivalent buffers and release them through events.
pub fn stream_tensor(
    s: &Shard,
    t: &TensorInfo,
    capacity: usize,
    mut consume: impl FnMut(u64, &[u8]) -> Result<()>,
) -> Result<()> {
    let (_, block_bytes) = type_layout(t.kind)?;
    let quantum = block_bytes as usize;
    ensure!(
        capacity >= quantum && capacity <= 512 << 20,
        "staging capacity must be one block..512 MiB"
    );
    let chunk = capacity / quantum * quantum;
    let mut buf = vec![0u8; chunk.min(t.bytes as usize)];
    let f = File::open(&s.path)?;
    ensure!(
        f.metadata()?.len() == s.header.file_bytes,
        "model file changed"
    );
    let base = s
        .header
        .data_offset
        .checked_add(t.offset)
        .context("tensor offset overflow")?;
    let mut offset = 0;
    while offset < t.bytes {
        let n = (t.bytes - offset).min(buf.len() as u64) as usize;
        f.read_exact_at(&mut buf[..n], base + offset)?;
        consume(offset, &buf[..n])?;
        // A hint for completed source pages. Does not discard modified data or
        // locked PLE pages; the OS may keep a cache when RAM is otherwise free.
        unsafe {
            libc::posix_fadvise(
                f.as_raw_fd(),
                (base + offset) as libc::off_t,
                n as libc::off_t,
                libc::POSIX_FADV_DONTNEED,
            );
        }
        offset += n as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::Header;
    use std::collections::BTreeMap;
    use std::io::Write;
    #[test]
    fn streamed_payload_is_complete_and_bounded() {
        let path = std::env::temp_dir().join(format!("q4e-staging-{}", std::process::id()));
        let payload: Vec<_> = (0..34 * 19).map(|i| (i % 251) as u8).collect();
        let mut f = File::create(&path).unwrap();
        f.write_all(&[0; 32]).unwrap();
        f.write_all(&payload).unwrap();
        drop(f);
        let t = TensorInfo {
            name: "weight".into(),
            dimensions: vec![32, 19],
            kind: 8,
            offset: 0,
            bytes: payload.len() as u64,
        };
        let s = Shard {
            path: path.clone(),
            header: Header {
                version: 3,
                metadata: BTreeMap::new(),
                tensors: vec![t.clone()],
                data_offset: 32,
                file_bytes: 32 + t.bytes,
            },
        };
        let mut got = Vec::new();
        stream_tensor(&s, &t, 101, |offset, b| {
            assert_eq!(offset, got.len() as u64);
            assert!(b.len() <= 101 && b.len() % 34 == 0);
            got.extend_from_slice(b);
            Ok(())
        })
        .unwrap();
        assert_eq!(got, payload);
        std::fs::remove_file(path).unwrap();
    }
}
