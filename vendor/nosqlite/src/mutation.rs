use serde_json::{json, Map, Value};

use crate::{
    query::{compile_filter, CompiledFilter},
    Document,
};

pub struct CompiledUpdates<'a> {
    set: Option<&'a Map<String, Value>>,
    unset: Vec<&'a str>,
    inc: Option<&'a Map<String, Value>>,
    single_set: Option<(&'a str, &'a Value)>,
}

impl<'a> CompiledUpdates<'a> {
    pub fn new(updates: &'a Document) -> Self {
        let set = updates.get("$set").and_then(Value::as_object);
        let unset = updates
            .get("$unset")
            .and_then(Value::as_array)
            .into_iter()
            .flat_map(|fields| fields.iter().filter_map(Value::as_str))
            .collect::<Vec<_>>();
        let inc = updates.get("$inc").and_then(Value::as_object);
        let single_set = match (set, unset.is_empty(), inc) {
            (Some(set), true, None) if set.len() == 1 => {
                set.iter().next().map(|(key, value)| (key.as_str(), value))
            }
            _ => None,
        };
        Self {
            set,
            unset,
            inc,
            single_set,
        }
    }

    pub fn fields(&self) -> impl Iterator<Item = &str> {
        let set = self
            .set
            .into_iter()
            .flat_map(|fields| fields.keys().map(String::as_str));
        let inc = self
            .inc
            .into_iter()
            .flat_map(|fields| fields.keys().map(String::as_str));

        set.chain(inc).chain(self.unset.iter().copied())
    }

    pub fn would_change(&self, document: &Document) -> bool {
        if let Some((key, value)) = self.single_set {
            return document.get(key) != Some(value);
        }

        if let Some(set) = self.set {
            if set
                .iter()
                .any(|(key, value)| document.get(key) != Some(value))
            {
                return true;
            }
        }

        if self
            .unset
            .iter()
            .copied()
            .any(|field| document.contains_key(field))
        {
            return true;
        }

        if let Some(inc) = self.inc {
            return inc.iter().any(|(key, value)| {
                let delta = value.as_f64().unwrap_or(0.0);
                let actual = document.get(key);
                let current = actual.and_then(Value::as_f64).unwrap_or(0.0);
                !number_value_equals_f64(actual, current + delta)
            });
        }

        false
    }
}

pub fn apply_updates(document: &mut Document, updates: &Document) -> bool {
    apply_compiled_updates(document, &CompiledUpdates::new(updates))
}

pub fn apply_compiled_updates(document: &mut Document, updates: &CompiledUpdates<'_>) -> bool {
    if let Some((key, value)) = updates.single_set {
        let changed = document.get(key) != Some(value);
        set_document_value(document, key, value);
        return changed;
    }

    let mut changed = false;

    if let Some(set) = updates.set {
        for (key, value) in set {
            changed |= document.get(key) != Some(value);
            set_document_value(document, key, value);
        }
    }

    if !updates.unset.is_empty() {
        for field in updates.unset.iter().copied() {
            changed |= document.remove(field).is_some();
        }
    }

    if let Some(inc) = updates.inc {
        for (key, value) in inc {
            let delta = value.as_f64().unwrap_or(0.0);
            let current = document.get(key).and_then(Value::as_f64).unwrap_or(0.0);
            let next = current + delta;
            let next = if next.fract() == 0.0 {
                json!(next as i64)
            } else {
                json!(next)
            };
            changed |= document.get(key) != Some(&next);
            set_document_value(document, key, &next);
        }
    }

    changed
}

fn number_value_equals_f64(value: Option<&Value>, expected: f64) -> bool {
    let Some(Value::Number(number)) = value else {
        return false;
    };

    if expected.fract() == 0.0 {
        let expected = expected as i64;
        return number.as_i64() == Some(expected)
            || (expected >= 0
                && number
                    .as_u64()
                    .is_some_and(|actual| actual == expected as u64));
    }

    number
        .as_f64()
        .is_some_and(|actual| actual.to_bits() == expected.to_bits())
}

fn set_document_value(document: &mut Document, key: &str, value: &Value) {
    if let Some(existing) = document.get_mut(key) {
        *existing = value.clone();
    } else {
        document.insert(key.to_string(), value.clone());
    }
}

pub fn apply_update_many(documents: &mut [Document], filter: &Document, updates: &Document) {
    let filter = compile_filter(filter);
    let updates = CompiledUpdates::new(updates);
    for document in documents {
        if filter.matches(document) {
            apply_compiled_updates(document, &updates);
        }
    }
}

pub fn apply_delete_many(documents: &mut Vec<Document>, filter: &Document) {
    let filter = compile_filter(filter);
    apply_delete_many_compiled(documents, &filter);
}

pub fn apply_delete_many_compiled(documents: &mut Vec<Document>, filter: &CompiledFilter<'_>) {
    documents.retain(|document| !filter.matches(document));
}

pub fn apply_cleanup_expired(documents: &mut Vec<Document>, ttl_field: &str, now: i64) {
    documents.retain(|document| {
        document
            .get(ttl_field)
            .and_then(Value::as_i64)
            .is_none_or(|ttl| ttl > now)
    });
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn increment_preflight_compares_numbers_without_allocating_values() {
        assert!(number_value_equals_f64(Some(&json!(3)), 3.0));
        assert!(number_value_equals_f64(Some(&json!(3.5)), 3.5));
        assert!(!number_value_equals_f64(Some(&json!(3)), 4.0));
        assert!(!number_value_equals_f64(Some(&json!("3")), 3.0));
    }
}
