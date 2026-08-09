use serde::{Deserialize, Serialize};

use crate::{Error, Result};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NeuralSpace {
    pub encoder: String,
    pub dimensions: usize,
    #[serde(default = "default_max_prototypes")]
    pub max_prototypes: usize,
    #[serde(default)]
    pub prototypes: Vec<Prototype>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Prototype {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub vector: Vec<f64>,
    pub weight: f64,
}

impl NeuralSpace {
    pub fn new(encoder: &str, dimensions: usize, max_prototypes: usize) -> Self {
        Self {
            encoder: encoder.to_string(),
            dimensions,
            max_prototypes: max_prototypes.max(1),
            prototypes: Vec::new(),
        }
    }

    pub fn validate_vector(&self, vector: &[f64]) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(Error::VectorDimensionMismatch {
                field: "neuralSpace.vector".to_string(),
                expected: self.dimensions,
                actual: vector.len(),
            });
        }
        Ok(())
    }

    pub fn learn(&mut self, vector: &[f64], label: Option<String>, weight: f64) -> Result<()> {
        self.validate_vector(vector)?;
        let mut vector = vector.to_vec();
        normalize(&mut vector);
        let weight = if weight == 0.0 { 1.0 } else { weight };

        if let Some(position) = self.matching_prototype(&vector, label.as_deref()) {
            update_prototype(&mut self.prototypes[position], &vector, weight);
        } else if weight > 0.0 && self.prototypes.len() < self.max_prototypes {
            self.prototypes.push(Prototype {
                label,
                vector,
                weight,
            });
        } else if let Some(position) = self.nearest_prototype(&vector, label.as_deref()) {
            update_prototype(&mut self.prototypes[position], &vector, weight);
        }

        self.prototypes
            .sort_by(|left, right| right.weight.total_cmp(&left.weight));
        Ok(())
    }

    pub fn adapt_query(&self, query: &[f64], label: Option<&str>) -> Result<Vec<f64>> {
        self.validate_vector(query)?;
        let mut adapted = query.to_vec();
        normalize(&mut adapted);

        let mut matches = self
            .prototypes
            .iter()
            .filter(|prototype| label.is_none_or(|label| prototype.label.as_deref() == Some(label)))
            .map(|prototype| (cosine(&adapted, &prototype.vector), prototype))
            .filter(|(score, prototype)| *score > 0.0 && prototype.weight > 0.0)
            .collect::<Vec<_>>();
        matches.sort_by(|left, right| right.0.total_cmp(&left.0));

        for (rank, (score, prototype)) in matches.into_iter().take(3).enumerate() {
            let blend = (0.30 / (rank as f64 + 1.0)) * score.min(1.0);
            for (value, proto_value) in adapted.iter_mut().zip(&prototype.vector) {
                *value = (*value * (1.0 - blend)) + (proto_value * blend);
            }
        }

        normalize(&mut adapted);
        Ok(adapted)
    }

    fn matching_prototype(&self, vector: &[f64], label: Option<&str>) -> Option<usize> {
        if let Some(label) = label {
            return self
                .prototypes
                .iter()
                .position(|prototype| prototype.label.as_deref() == Some(label));
        }

        self.nearest_prototype(vector, None)
            .filter(|position| cosine(vector, &self.prototypes[*position].vector) >= 0.92)
    }

    fn nearest_prototype(&self, vector: &[f64], label: Option<&str>) -> Option<usize> {
        self.prototypes
            .iter()
            .enumerate()
            .filter(|(_, prototype)| {
                label.is_none_or(|label| prototype.label.as_deref() == Some(label))
            })
            .max_by(|(_, left), (_, right)| {
                cosine(vector, &left.vector).total_cmp(&cosine(vector, &right.vector))
            })
            .map(|(position, _)| position)
    }
}

fn update_prototype(prototype: &mut Prototype, vector: &[f64], weight: f64) {
    let total = (prototype.weight.abs() + weight.abs()).max(1.0);
    let direction = if weight >= 0.0 { 1.0 } else { -1.0 };
    let rate = (weight.abs() / total).clamp(0.05, 0.50);

    for (value, learned) in prototype.vector.iter_mut().zip(vector) {
        *value = (*value * (1.0 - rate)) + (learned * rate * direction);
    }
    normalize(&mut prototype.vector);
    prototype.weight = (prototype.weight + weight).max(0.0);
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

fn cosine(left: &[f64], right: &[f64]) -> f64 {
    let mut dot = 0.0;
    let mut left_norm = 0.0;
    let mut right_norm = 0.0;
    for (left, right) in left.iter().zip(right) {
        dot += left * right;
        left_norm += left * left;
        right_norm += right * right;
    }
    if left_norm == 0.0 || right_norm == 0.0 {
        return 0.0;
    }
    dot / (left_norm.sqrt() * right_norm.sqrt())
}

fn default_max_prototypes() -> usize {
    256
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_and_adapts_toward_matching_prototypes() {
        let mut space = NeuralSpace::new("local_text", 3, 8);
        space
            .learn(&[1.0, 0.0, 0.0], Some("east".to_string()), 1.0)
            .unwrap();
        let adapted = space.adapt_query(&[0.8, 0.2, 0.0], Some("east")).unwrap();

        assert!(adapted[0] > 0.8);
    }
}
