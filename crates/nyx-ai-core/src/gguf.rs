//! Read a GGUF file's metadata without touching its tensors.

use serde::{Deserialize, Serialize};
use std::io::{BufReader, Read};
use std::path::Path;

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct Meta {
    pub version: u32,
    pub architecture: Option<String>,
    pub name: Option<String>,
    /// `general.type`: `model`, `mmproj`, `adapter`, …
    pub kind: Option<String>,
    pub file_type: Option<u32>,
    pub size_label: Option<String>,
    pub license: Option<String>,
    pub context_length: Option<u64>,
    pub embedding_length: Option<u64>,
    /// 0 none, 1 mean, 2 cls, 3 last, 4 rank.
    pub pooling_type: Option<u32>,
    pub causal_attention: Option<bool>,
    pub chat_template: Option<String>,
    pub split_count: Option<u32>,
    pub vision_encoder: bool,
    pub audio_encoder: bool,
}

const MAX_STRING: u64 = 16 << 20;
const MAX_KV: u64 = 1 << 20;

#[derive(Debug)]
pub struct Invalid(pub String);

impl std::fmt::Display for Invalid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Invalid {}

fn bad(what: impl Into<String>) -> Invalid {
    Invalid(what.into())
}

/// Whether a file starts like a GGUF file, for a quick check before loading.
pub fn looks_like_gguf(path: &Path) -> bool {
    let mut magic = [0u8; 8];
    std::fs::File::open(path).and_then(|mut f| f.read_exact(&mut magic)).is_ok()
        && &magic[..4] == b"GGUF"
        && matches!(u32::from_le_bytes([magic[4], magic[5], magic[6], magic[7]]), 2 | 3)
}

pub fn read(path: &Path) -> Result<Meta, Invalid> {
    let file = std::fs::File::open(path).map_err(|e| bad(format!("can't open: {e}")))?;
    parse(&mut BufReader::with_capacity(1 << 16, file))
}

struct Reader<'a, R: Read> {
    input: &'a mut R,
}

impl<R: Read> Reader<'_, R> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N], Invalid> {
        let mut buf = [0u8; N];
        self.input.read_exact(&mut buf).map_err(|_| bad("truncated header"))?;
        Ok(buf)
    }

    fn u32(&mut self) -> Result<u32, Invalid> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }

    fn u64(&mut self) -> Result<u64, Invalid> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }

    fn string(&mut self) -> Result<String, Invalid> {
        let len = self.u64()?;
        if len > MAX_STRING {
            return Err(bad("string too long"));
        }
        let mut buf = vec![0u8; len as usize];
        self.input.read_exact(&mut buf).map_err(|_| bad("truncated string"))?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    fn skip(&mut self, n: u64) -> Result<(), Invalid> {
        let copied = std::io::copy(&mut (&mut *self.input).take(n), &mut std::io::sink()).map_err(|_| bad("truncated value"))?;
        if copied == n { Ok(()) } else { Err(bad("truncated value")) }
    }

    fn value(&mut self, kind: u32) -> Result<Value, Invalid> {
        Ok(match kind {
            0 | 1 => Value::Int(self.bytes::<1>()?[0] as i64),
            2 => Value::Int(u16::from_le_bytes(self.bytes()?) as i64),
            3 => Value::Int(i16::from_le_bytes(self.bytes()?) as i64),
            4 => Value::Int(self.u32()? as i64),
            5 => Value::Int(i32::from_le_bytes(self.bytes()?) as i64),
            6 => Value::Float(f32::from_le_bytes(self.bytes()?) as f64),
            7 => Value::Bool(self.bytes::<1>()?[0] != 0),
            8 => Value::Text(self.string()?),
            9 => {
                let item = self.u32()?;
                let count = self.u64()?;
                match fixed_size(item) {
                    Some(size) => self.skip(size.checked_mul(count).ok_or_else(|| bad("array too long"))?)?,
                    None if item == 8 => {
                        for _ in 0..count {
                            let len = self.u64()?;
                            if len > MAX_STRING {
                                return Err(bad("string too long"));
                            }
                            self.skip(len)?;
                        }
                    }
                    None => {
                        for _ in 0..count {
                            self.value(item)?;
                        }
                    }
                }
                Value::Array
            }
            10 => Value::Int(self.u64()? as i64),
            11 => Value::Int(i64::from_le_bytes(self.bytes()?)),
            12 => Value::Float(f64::from_le_bytes(self.bytes()?)),
            other => return Err(bad(format!("unknown value type {other}"))),
        })
    }
}

fn fixed_size(kind: u32) -> Option<u64> {
    match kind {
        0 | 1 | 7 => Some(1),
        2 | 3 => Some(2),
        4..=6 => Some(4),
        10..=12 => Some(8),
        _ => None,
    }
}

enum Value {
    Int(i64),
    Float(#[allow(dead_code)] f64),
    Bool(bool),
    Text(String),
    Array,
}

pub fn parse<R: Read>(input: &mut R) -> Result<Meta, Invalid> {
    let mut r = Reader { input };
    if &r.bytes::<4>()? != b"GGUF" {
        return Err(bad("not a GGUF file"));
    }
    let version = r.u32()?;
    if !matches!(version, 2 | 3) {
        return Err(bad(format!("unsupported GGUF version {version}")));
    }
    let _tensors = r.u64()?;
    let count = r.u64()?;
    if count > MAX_KV {
        return Err(bad("too many metadata keys"));
    }
    let mut meta = Meta { version, ..Default::default() };
    let mut by_arch: Vec<(String, Value)> = Vec::new();
    for _ in 0..count {
        let key = r.string()?;
        let kind = r.u32()?;
        let value = r.value(kind)?;
        let text = |v: &Value| match v {
            Value::Text(s) => Some(s.clone()),
            _ => None,
        };
        let int = |v: &Value| match v {
            Value::Int(n) => Some(*n),
            _ => None,
        };
        match key.as_str() {
            "general.architecture" => meta.architecture = text(&value),
            "general.name" => meta.name = text(&value),
            "general.type" => meta.kind = text(&value),
            "general.file_type" => meta.file_type = int(&value).map(|n| n as u32),
            "general.size_label" => meta.size_label = text(&value),
            "general.license" => meta.license = text(&value),
            "tokenizer.chat_template" => meta.chat_template = text(&value),
            "split.count" => meta.split_count = int(&value).map(|n| n as u32),
            "clip.has_vision_encoder" => meta.vision_encoder = matches!(value, Value::Bool(true)),
            "clip.has_audio_encoder" => meta.audio_encoder = matches!(value, Value::Bool(true)),
            _ if key.ends_with(".context_length")
                || key.ends_with(".embedding_length")
                || key.ends_with(".pooling_type")
                || key.ends_with(".causal_attention") =>
            {
                by_arch.push((key, value));
            }
            _ => {}
        }
    }
    if let Some(arch) = meta.architecture.clone() {
        for (key, value) in by_arch {
            let Some(field) = key.strip_prefix(&arch).and_then(|k| k.strip_prefix('.')) else { continue };
            match (field, value) {
                ("context_length", Value::Int(n)) => meta.context_length = Some(n as u64),
                ("embedding_length", Value::Int(n)) => meta.embedding_length = Some(n as u64),
                ("pooling_type", Value::Int(n)) => meta.pooling_type = Some(n as u32),
                ("causal_attention", Value::Bool(b)) => meta.causal_attention = Some(b),
                _ => {}
            }
        }
    }
    Ok(meta)
}

impl Meta {
    /// A multimodal projector, not a model of its own.
    pub fn projector(&self) -> bool {
        self.architecture.as_deref() == Some("clip") || self.kind.as_deref() == Some("mmproj")
    }

    /// Pools token states into one vector (mean, cls or last).
    pub fn embedding(&self) -> bool {
        matches!(self.pooling_type, Some(1..=3)) || (self.causal_attention == Some(false) && self.chat_template.is_none())
    }

    pub fn chat(&self) -> bool {
        self.chat_template.is_some() && !self.embedding() && !self.projector()
    }

    /// The template takes a tool list.
    pub fn tools(&self) -> bool {
        self.chat_template.as_deref().is_some_and(|t| t.contains("tools"))
    }

    /// The template has a reasoning section or switch.
    pub fn thinking(&self) -> bool {
        self.chat_template.as_deref().is_some_and(|t| {
            ["enable_thinking", "<think>", "reasoning_content", "<|channel>thought", "thinking"].iter().any(|m| t.contains(m))
        })
    }

    /// `general.file_type` as llama.cpp names it.
    pub fn quantization(&self) -> Option<&'static str> {
        Some(match self.file_type? {
            0 => "F32",
            1 => "F16",
            2 => "Q4_0",
            3 => "Q4_1",
            7 => "Q8_0",
            8 => "Q5_0",
            9 => "Q5_1",
            10 => "Q2_K",
            11 => "Q3_K_S",
            12 => "Q3_K_M",
            13 => "Q3_K_L",
            14 => "Q4_K_S",
            15 => "Q4_K_M",
            16 => "Q5_K_S",
            17 => "Q5_K_M",
            18 => "Q6_K",
            19 => "IQ2_XXS",
            20 => "IQ2_XS",
            21 => "Q2_K_S",
            22 => "IQ3_XS",
            23 => "IQ3_XXS",
            24 => "IQ1_S",
            25 => "IQ4_NL",
            26 => "IQ3_S",
            27 => "IQ3_M",
            28 => "IQ2_S",
            29 => "IQ2_M",
            30 => "IQ4_XS",
            31 => "IQ1_M",
            32 => "BF16",
            36 => "TQ1_0",
            37 => "TQ2_0",
            38 => "MXFP4",
            _ => return None,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A small GGUF header for tests.
    pub fn build(pairs: &[(&str, TestValue)]) -> Vec<u8> {
        let mut out = b"GGUF".to_vec();
        out.extend(3u32.to_le_bytes());
        out.extend(0u64.to_le_bytes());
        out.extend((pairs.len() as u64).to_le_bytes());
        let string = |out: &mut Vec<u8>, s: &str| {
            out.extend((s.len() as u64).to_le_bytes());
            out.extend(s.as_bytes());
        };
        for (key, value) in pairs {
            string(&mut out, key);
            match value {
                TestValue::U32(n) => {
                    out.extend(4u32.to_le_bytes());
                    out.extend(n.to_le_bytes());
                }
                TestValue::Str(s) => {
                    out.extend(8u32.to_le_bytes());
                    string(&mut out, s);
                }
                TestValue::Bool(b) => {
                    out.extend(7u32.to_le_bytes());
                    out.push(*b as u8);
                }
                TestValue::Strings(items) => {
                    out.extend(9u32.to_le_bytes());
                    out.extend(8u32.to_le_bytes());
                    out.extend((items.len() as u64).to_le_bytes());
                    for s in *items {
                        string(&mut out, s);
                    }
                }
            }
        }
        out
    }

    pub enum TestValue {
        U32(u32),
        Str(&'static str),
        Bool(bool),
        Strings(&'static [&'static str]),
    }

    pub fn chat_model() -> Vec<u8> {
        build(&[
            ("general.architecture", TestValue::Str("gemma4")),
            ("general.name", TestValue::Str("Gemma 4")),
            ("general.file_type", TestValue::U32(15)),
            ("tokenizer.ggml.tokens", TestValue::Strings(&["a", "b", "c"])),
            ("gemma4.context_length", TestValue::U32(131072)),
            ("tokenizer.chat_template", TestValue::Str("{% if tools %}{{ enable_thinking }}{% endif %}")),
        ])
    }

    #[test]
    fn reads_chat_model() {
        let meta = parse(&mut chat_model().as_slice()).unwrap();
        assert_eq!(meta.architecture.as_deref(), Some("gemma4"));
        assert_eq!(meta.context_length, Some(131072));
        assert_eq!(meta.quantization(), Some("Q4_K_M"));
        assert!(meta.chat() && meta.tools() && meta.thinking());
        assert!(!meta.embedding());
    }

    #[test]
    fn reads_embedding_and_projector() {
        let embed = build(&[
            ("general.architecture", TestValue::Str("gemma-embedding")),
            ("gemma-embedding.pooling_type", TestValue::U32(1)),
            ("gemma-embedding.embedding_length", TestValue::U32(768)),
        ]);
        let meta = parse(&mut embed.as_slice()).unwrap();
        assert!(meta.embedding() && !meta.chat());
        assert_eq!(meta.embedding_length, Some(768));
        let clip = build(&[("general.architecture", TestValue::Str("clip")), ("clip.has_vision_encoder", TestValue::Bool(true))]);
        let meta = parse(&mut clip.as_slice()).unwrap();
        assert!(meta.projector() && meta.vision_encoder);
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse(&mut &b"GGML\0\0\0\0"[..]).is_err());
        let mut truncated = chat_model();
        truncated.truncate(40);
        assert!(parse(&mut truncated.as_slice()).is_err());
    }
}
