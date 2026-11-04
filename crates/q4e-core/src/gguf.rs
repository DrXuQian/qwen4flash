//! GGUF v2/v3 metadata and split-file intake. Tensor payloads are never read by inspection.
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek},
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Value {
    Unsigned(u64),
    Signed(i64),
    Float(f64),
    Bool(bool),
    Text(String),
    Array(Vec<Value>),
}
impl Value {
    pub fn u64(&self) -> Result<u64> {
        match self {
            Self::Unsigned(x) => Ok(*x),
            Self::Signed(x) if *x >= 0 => Ok(*x as u64),
            _ => bail!("expected unsigned metadata, got {self:?}"),
        }
    }
    pub fn f64(&self) -> Result<f64> {
        match self {
            Self::Float(x) => Ok(*x),
            Self::Unsigned(x) => Ok(*x as f64),
            Self::Signed(x) => Ok(*x as f64),
            _ => bail!("expected numeric metadata"),
        }
    }
    pub fn text(&self) -> Result<&str> {
        if let Self::Text(x) = self {
            Ok(x)
        } else {
            bail!("expected string metadata")
        }
    }
    pub fn array(&self) -> Result<&[Value]> {
        if let Self::Array(x) = self {
            Ok(x)
        } else {
            bail!("expected array metadata")
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub kind: u32,
    pub offset: u64,
    pub bytes: u64,
}
impl TensorInfo {
    pub fn elements(&self) -> Result<u64> {
        self.dimensions.iter().try_fold(1u64, |a, b| {
            a.checked_mul(*b).context("tensor element overflow")
        })
    }
    pub fn is_ple_table(&self) -> bool {
        self.name == "per_layer_token_embd.weight"
    }
}

// GGML block sizes. Unknown formats are rejected, not guessed from adjacent offsets.
pub fn type_layout(kind: u32) -> Result<(u64, u64)> {
    Ok(match kind {
        0 => (1, 4),
        1 => (1, 2),
        2 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        9 => (32, 40),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        15 => (256, 292),
        16 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        19 => (256, 50),
        20 => (32, 18),
        21 => (256, 110),
        22 => (256, 82),
        23 => (256, 136),
        24 => (1, 1),
        25 => (1, 2),
        26 => (1, 4),
        27 => (1, 8),
        28 => (1, 8),
        29 => (256, 56),
        30 => (1, 2),
        42 => (64, 18),
        _ => bail!("unsupported GGML type {kind}; add its exact storage contract before loading"),
    })
}
pub fn type_name(k: u32) -> String {
    match k {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        20 => "IQ4_NL",
        30 => "BF16",
        42 => "Q2_0",
        _ => return format!("GGML_{k}"),
    }
    .into()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Header {
    pub version: u32,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64,
    pub file_bytes: u64,
}
impl Header {
    pub fn read(path: &Path) -> Result<Self> {
        let mut f = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let size = f.metadata()?.len();
        Self::parse(&mut f, size)
    }
    pub fn parse<R: Read + Seek>(r: &mut R, file_bytes: u64) -> Result<Self> {
        ensure!(
            read_array::<4, _>(r)? == *b"GGUF",
            "not a little-endian GGUF file"
        );
        let version = u32::from_le_bytes(read_array(r)?);
        ensure!(
            version == 2 || version == 3,
            "GGUF version {version} is unsupported"
        );
        let n = u64::from_le_bytes(read_array(r)?);
        let n_meta = u64::from_le_bytes(read_array(r)?);
        ensure!(
            n <= 1_000_000 && n_meta <= 100_000,
            "unreasonable GGUF descriptor counts"
        );
        let mut metadata = BTreeMap::new();
        let mut budget = 128usize << 20;
        for _ in 0..n_meta {
            let key = read_string(r, &mut budget)?;
            let kind = u32::from_le_bytes(read_array(r)?);
            let v = read_value(r, kind, &mut budget, 0)?;
            ensure!(
                metadata.insert(key.clone(), v).is_none(),
                "duplicate metadata key {key}"
            );
        }
        let mut tensors = Vec::new();
        let mut names = std::collections::BTreeSet::new();
        for _ in 0..n {
            let name = read_string(r, &mut budget)?;
            ensure!(names.insert(name.clone()), "duplicate tensor {name}");
            let nd = u32::from_le_bytes(read_array(r)?);
            ensure!((1..=4).contains(&nd), "invalid rank of {name}");
            let mut dimensions = Vec::new();
            for _ in 0..nd {
                let d = u64::from_le_bytes(read_array(r)?);
                ensure!(d > 0, "zero dimension of {name}");
                dimensions.push(d);
            }
            let kind = u32::from_le_bytes(read_array(r)?);
            let offset = u64::from_le_bytes(read_array(r)?);
            let (block, bytes) = type_layout(kind)?;
            ensure!(
                dimensions[0] % block == 0,
                "{name}: row width {} is not divisible by quant block {block}",
                dimensions[0]
            );
            let elements = dimensions.iter().try_fold(1u64, |a, b| {
                a.checked_mul(*b).context("tensor extent overflow")
            })?;
            let bytes = (elements / block)
                .checked_mul(bytes)
                .context("tensor byte overflow")?;
            tensors.push(TensorInfo {
                name,
                dimensions,
                kind,
                offset,
                bytes,
            });
        }
        let align = metadata
            .get("general.alignment")
            .map(Value::u64)
            .transpose()?
            .unwrap_or(32);
        ensure!(
            align.is_power_of_two() && align <= 1 << 20,
            "invalid GGUF alignment {align}"
        );
        let cursor = r.stream_position()?;
        let data_offset = cursor.checked_add(align - 1).context("header overflow")? & !(align - 1);
        ensure!(data_offset <= file_bytes, "truncated GGUF header");
        let mut spans = Vec::new();
        for t in &tensors {
            ensure!(t.offset % align == 0, "{}: unaligned tensor offset", t.name);
            let end = data_offset
                .checked_add(t.offset)
                .and_then(|x| x.checked_add(t.bytes))
                .context("tensor offset overflow")?;
            ensure!(end <= file_bytes, "{}: payload exceeds file size", t.name);
            spans.push((t.offset, t.offset + t.bytes, &t.name));
        }
        spans.sort_by_key(|x| x.0);
        for w in spans.windows(2) {
            ensure!(
                w[0].1 <= w[1].0,
                "overlapping tensor payloads {} and {}",
                w[0].2,
                w[1].2
            );
        }
        Ok(Self {
            version,
            metadata,
            tensors,
            data_offset,
            file_bytes,
        })
    }
    pub fn required(&self, key: &str) -> Result<&Value> {
        self.metadata
            .get(key)
            .with_context(|| format!("missing GGUF metadata {key}"))
    }
}

fn read_array<const N: usize, R: Read>(r: &mut R) -> Result<[u8; N]> {
    let mut a = [0; N];
    r.read_exact(&mut a).context("truncated GGUF descriptor")?;
    Ok(a)
}
fn charge(budget: &mut usize, n: usize) -> Result<()> {
    *budget = budget
        .checked_sub(n)
        .context("GGUF metadata exceeds 128 MiB budget")?;
    Ok(())
}
fn read_string<R: Read>(r: &mut R, budget: &mut usize) -> Result<String> {
    let n = u64::from_le_bytes(read_array(r)?);
    ensure!(n <= 16 << 20, "GGUF string exceeds 16 MiB");
    charge(budget, n as usize)?;
    let mut b = vec![0; n as usize];
    r.read_exact(&mut b)?;
    String::from_utf8(b).context("GGUF string is not UTF-8")
}
fn read_value<R: Read>(r: &mut R, k: u32, b: &mut usize, depth: usize) -> Result<Value> {
    charge(b, 8)?;
    Ok(match k {
        0 => Value::Unsigned(u8::from_le_bytes(read_array(r)?) as u64),
        1 => Value::Signed(i8::from_le_bytes(read_array(r)?) as i64),
        2 => Value::Unsigned(u16::from_le_bytes(read_array(r)?) as u64),
        3 => Value::Signed(i16::from_le_bytes(read_array(r)?) as i64),
        4 => Value::Unsigned(u32::from_le_bytes(read_array(r)?) as u64),
        5 => Value::Signed(i32::from_le_bytes(read_array(r)?) as i64),
        6 => Value::Float(f32::from_le_bytes(read_array(r)?) as f64),
        7 => {
            let x = u8::from_le_bytes(read_array(r)?);
            ensure!(x <= 1, "invalid GGUF bool");
            Value::Bool(x == 1)
        }
        8 => Value::Text(read_string(r, b)?),
        9 => {
            ensure!(depth == 0, "nested GGUF arrays are unsupported");
            let et = u32::from_le_bytes(read_array(r)?);
            ensure!(et != 9, "nested GGUF arrays are invalid");
            let n = u64::from_le_bytes(read_array(r)?);
            ensure!(n <= 2_000_000, "GGUF array exceeds element limit");
            let mut v = Vec::with_capacity(n as usize);
            for _ in 0..n {
                v.push(read_value(r, et, b, depth + 1)?);
            }
            Value::Array(v)
        }
        10 => Value::Unsigned(u64::from_le_bytes(read_array(r)?)),
        11 => Value::Signed(i64::from_le_bytes(read_array(r)?)),
        12 => Value::Float(f64::from_le_bytes(read_array(r)?)),
        _ => bail!("unknown GGUF metadata type {k}"),
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Shard {
    pub path: PathBuf,
    pub header: Header,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelIndex {
    pub shards: Vec<Shard>,
}
impl ModelIndex {
    pub fn open(first: &Path) -> Result<Self> {
        let header = Header::read(first)?;
        let count = header
            .metadata
            .get("split.count")
            .map(Value::u64)
            .transpose()?
            .unwrap_or(1);
        ensure!((1..=1024).contains(&count), "invalid shard count");
        if count == 1 {
            let index = Self {
                shards: vec![Shard {
                    path: first.into(),
                    header,
                }],
            };
            index.validate()?;
            return Ok(index);
        }
        ensure!(
            header.required("split.no")?.u64()? == 0,
            "pass the first GGUF shard"
        );
        let name = first
            .file_name()
            .context("missing filename")?
            .to_str()
            .context("non UTF-8 split filename")?;
        let suffix = format!("-00001-of-{count:05}.gguf");
        let stem = name
            .strip_suffix(&suffix)
            .context("split GGUF filename does not match metadata")?;
        let mut shards = vec![Shard {
            path: first.into(),
            header,
        }];
        for i in 1..count {
            let path = first.with_file_name(format!("{stem}-{:05}-of-{count:05}.gguf", i + 1));
            let header = Header::read(&path)?;
            ensure!(
                header.required("split.no")?.u64()? == i
                    && header.required("split.count")?.u64()? == count,
                "split identity mismatch in {}",
                path.display()
            );
            shards.push(Shard { path, header });
        }
        let index = Self { shards };
        index.validate()?;
        Ok(index)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.shards.is_empty(), "empty model index");
        let count = self
            .metadata()
            .get("split.count")
            .map(Value::u64)
            .transpose()?
            .unwrap_or(1);
        ensure!(
            count == self.shards.len() as u64,
            "incomplete shard inventory"
        );
        let mut names = std::collections::BTreeSet::new();
        for (i, s) in self.shards.iter().enumerate() {
            ensure!(
                (2..=3).contains(&s.header.version) && s.header.data_offset <= s.header.file_bytes,
                "invalid indexed GGUF header"
            );
            if count > 1 {
                ensure!(
                    s.header.required("split.no")?.u64()? == i as u64
                        && s.header.required("split.count")?.u64()? == count,
                    "split identity mismatch"
                );
            }
            let align = s
                .header
                .metadata
                .get("general.alignment")
                .map(Value::u64)
                .transpose()?
                .unwrap_or(32);
            ensure!(
                align.is_power_of_two() && align <= 1 << 20,
                "invalid indexed alignment"
            );
            let mut spans = Vec::new();
            for t in &s.header.tensors {
                ensure!(
                    (1..=4).contains(&t.dimensions.len()) && t.dimensions.iter().all(|&n| n > 0),
                    "invalid tensor rank/dimensions: {}",
                    t.name
                );
                let (block, size) = type_layout(t.kind)?;
                let elements = t.elements()?;
                ensure!(
                    t.dimensions[0] % block == 0
                        && elements
                            .checked_div(block)
                            .and_then(|n| n.checked_mul(size))
                            == Some(t.bytes),
                    "invalid tensor byte size: {}",
                    t.name
                );
                let end = t
                    .offset
                    .checked_add(t.bytes)
                    .context("tensor offset overflow")?;
                ensure!(
                    t.offset % align == 0
                        && s.header
                            .data_offset
                            .checked_add(end)
                            .is_some_and(|n| n <= s.header.file_bytes),
                    "invalid tensor span: {}",
                    t.name
                );
                spans.push((t.offset, end));
                ensure!(
                    names.insert(&t.name),
                    "duplicate tensor across shards: {}",
                    t.name
                );
            }
            spans.sort_unstable();
            ensure!(
                spans.windows(2).all(|w| w[0].1 <= w[1].0),
                "overlapping indexed tensors"
            );
        }
        if let Some(total) = self.metadata().get("split.tensors.count") {
            ensure!(
                total.u64()? == names.len() as u64,
                "incomplete tensor inventory"
            );
        }
        Ok(())
    }
    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.shards[0].header.metadata
    }
    pub fn tensors(&self) -> impl Iterator<Item = (&Shard, &TensorInfo)> {
        self.shards
            .iter()
            .flat_map(|s| s.header.tensors.iter().map(move |t| (s, t)))
    }
    pub fn tensor(&self, name: &str) -> Result<(&Shard, &TensorInfo)> {
        self.tensors()
            .find(|(_, t)| t.name == name)
            .with_context(|| format!("missing tensor {name}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn header() -> Vec<u8> {
        let mut b = b"GGUF".to_vec();
        b.extend(3u32.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        b.resize(32, 0);
        b
    }
    #[test]
    fn empty_metadata_shard_is_legal() {
        let b = header();
        assert!(Header::parse(&mut std::io::Cursor::new(b), 32).is_ok());
    }
    #[test]
    fn rejects_truncation_and_old_version() {
        let mut b = header();
        b[4] = 1;
        assert!(Header::parse(&mut std::io::Cursor::new(&b), 32).is_err());
        let b = header();
        assert!(Header::parse(&mut std::io::Cursor::new(&b[..20]), 20).is_err());
    }
    #[test]
    fn q5_1_is_not_q5_k() {
        assert_eq!(type_layout(7).unwrap(), (32, 24));
        assert_eq!(type_layout(13).unwrap(), (256, 176));
        assert!(type_layout(999).is_err());
    }
}
