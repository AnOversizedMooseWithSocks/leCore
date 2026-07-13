use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{Error, Result};

pub const DEFAULT_PROVIDER: &str = "holographic-hash-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EncoderSpec {
    #[serde(default = "default_kind")]
    pub kind: String,
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default = "default_dimensions")]
    pub dimensions: usize,
    #[serde(default)]
    pub seed: u64,
}

impl EncoderSpec {
    pub fn validate(&self) -> Result<()> {
        if self.kind != "text" {
            return Err(Error::UnsupportedEncoder(self.kind.clone()));
        }
        if self.provider != DEFAULT_PROVIDER && self.provider != "builtin-hash-v1" {
            return Err(Error::UnsupportedEncoder(self.provider.clone()));
        }
        if self.dimensions == 0 {
            return Err(Error::VectorDimensionMismatch {
                field: "encoder.dimensions".to_string(),
                expected: 1,
                actual: 0,
            });
        }
        Ok(())
    }
}

pub fn encode_value(spec: &EncoderSpec, value: &Value) -> Result<Vec<f64>> {
    spec.validate()?;
    Ok(encode_text(spec, &semantic_text(value)))
}

pub fn encode_text(spec: &EncoderSpec, text: &str) -> Vec<f64> {
    let mut vector = vec![0.0; spec.dimensions];
    let tokens = tokenize(text);

    if tokens.is_empty() {
        accumulate_feature(&mut vector, spec.seed, "empty", 1.0);
    }

    for token in &tokens {
        accumulate_feature(&mut vector, spec.seed, &format!("tok:{token}"), 1.0);
        for ngram in char_ngrams(token, 3, 5) {
            accumulate_feature(&mut vector, spec.seed, &format!("chr:{ngram}"), 0.35);
        }
    }

    for pair in tokens.windows(2) {
        accumulate_feature(
            &mut vector,
            spec.seed,
            &format!("pair:{}:{}", pair[0], pair[1]),
            0.75,
        );
    }

    normalize(&mut vector);
    vector
}

fn semantic_text(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => value.clone(),
        Value::Array(values) => values
            .iter()
            .map(semantic_text)
            .collect::<Vec<_>>()
            .join(" "),
        Value::Object(values) => values
            .iter()
            .map(|(key, value)| format!("{key} {}", semantic_text(value)))
            .collect::<Vec<_>>()
            .join(" "),
    }
}

fn tokenize(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();

    for ch in text.chars().flat_map(char::to_lowercase) {
        if ch.is_ascii_alphanumeric() {
            current.push(ch);
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

fn char_ngrams(token: &str, min: usize, max: usize) -> Vec<String> {
    let chars = token.chars().collect::<Vec<_>>();
    let mut ngrams = Vec::new();
    for width in min..=max {
        if chars.len() < width {
            continue;
        }
        for start in 0..=chars.len() - width {
            ngrams.push(chars[start..start + width].iter().collect());
        }
    }
    ngrams
}

fn accumulate_feature(vector: &mut [f64], seed: u64, feature: &str, weight: f64) {
    let feature_hash = hash_bytes(feature.as_bytes());
    for (dimension, value) in vector.iter_mut().enumerate() {
        let mut hash = seed ^ feature_hash ^ (dimension as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        hash = splitmix64(hash);
        let signed = if hash & 1 == 0 { -1.0 } else { 1.0 };
        let magnitude = (((hash >> 11) as f64) / ((1_u64 << 53) as f64)).max(0.05);
        *value += signed * magnitude * weight;
    }
}

fn normalize(vector: &mut [f64]) {
    let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm == 0.0 {
        return;
    }
    for value in vector {
        *value /= norm;
    }
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn default_kind() -> String {
    "text".to_string()
}

fn default_provider() -> String {
    DEFAULT_PROVIDER.to_string()
}

fn default_dimensions() -> usize {
    384
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_encoder_is_deterministic_and_normalized() {
        let spec = EncoderSpec {
            kind: "text".to_string(),
            provider: DEFAULT_PROVIDER.to_string(),
            dimensions: 64,
            seed: 42,
        };
        let first = encode_text(&spec, "memory of a quiet blue room");
        let second = encode_text(&spec, "memory of a quiet blue room");
        assert_eq!(first, second);

        let norm = first.iter().map(|value| value * value).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 0.0000001);
    }
}
