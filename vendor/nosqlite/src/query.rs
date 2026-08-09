use serde_json::Value;

use crate::Document;

pub struct CompiledFilter<'a> {
    fields: Vec<CompiledField<'a>>,
}

struct CompiledField<'a> {
    key: &'a str,
    predicate: FieldPredicate<'a>,
}

enum FieldPredicate<'a> {
    Equality(&'a Value),
    Operators(Vec<FieldOperator<'a>>),
}

enum FieldOperator<'a> {
    Eq(&'a Value),
    Ne(&'a Value),
    Gt(&'a Value),
    Gte(&'a Value),
    Lt(&'a Value),
    Lte(&'a Value),
    Prefix(&'a Value),
    Exists(&'a Value),
    Unknown,
}

pub fn matches_filter(document: &Document, filter: &Document) -> bool {
    compile_filter(filter).matches(document)
}

pub fn compile_filter(filter: &Document) -> CompiledFilter<'_> {
    let fields = filter
        .iter()
        .map(|(key, expected)| CompiledField {
            key,
            predicate: compile_field_predicate(expected),
        })
        .collect();
    CompiledFilter { fields }
}

pub fn compile_filter_excluding<'a>(
    filter: &'a Document,
    excluded_field: Option<&str>,
) -> CompiledFilter<'a> {
    let fields = filter
        .iter()
        .filter(|(key, _)| excluded_field != Some(key.as_str()))
        .map(|(key, expected)| CompiledField {
            key,
            predicate: compile_field_predicate(expected),
        })
        .collect();
    CompiledFilter { fields }
}

impl CompiledFilter<'_> {
    pub fn matches(&self, document: &Document) -> bool {
        if self.fields.is_empty() {
            return true;
        }
        if self.fields.len() == 1 {
            let field = &self.fields[0];
            if let FieldPredicate::Equality(expected) = field.predicate {
                return document.get(field.key) == Some(expected);
            }
        }

        self.fields
            .iter()
            .all(|field| field.predicate.matches(document.get(field.key)))
    }
}

impl FieldPredicate<'_> {
    fn matches(&self, actual: Option<&Value>) -> bool {
        match self {
            FieldPredicate::Equality(expected) => actual == Some(*expected),
            FieldPredicate::Operators(operators) => {
                operators.iter().all(|operator| operator.matches(actual))
            }
        }
    }
}

impl FieldOperator<'_> {
    fn matches(&self, actual: Option<&Value>) -> bool {
        match self {
            FieldOperator::Eq(expected) => actual == Some(*expected),
            FieldOperator::Ne(expected) => actual != Some(*expected),
            FieldOperator::Gt(expected) => {
                compare_numbers(actual, expected, |left, right| left > right)
            }
            FieldOperator::Gte(expected) => {
                compare_numbers(actual, expected, |left, right| left >= right)
            }
            FieldOperator::Lt(expected) => {
                compare_numbers(actual, expected, |left, right| left < right)
            }
            FieldOperator::Lte(expected) => {
                compare_numbers(actual, expected, |left, right| left <= right)
            }
            FieldOperator::Prefix(expected) => actual
                .and_then(Value::as_str)
                .zip(expected.as_str())
                .is_some_and(|(actual, expected)| actual.starts_with(expected)),
            FieldOperator::Exists(expected) => expected
                .as_bool()
                .is_some_and(|should_exist| actual.is_some() == should_exist),
            FieldOperator::Unknown => false,
        }
    }
}

fn compile_field_predicate(expected: &Value) -> FieldPredicate<'_> {
    match expected {
        Value::Object(operator) if operator.len() == 1 => {
            if let Some(expected) = operator.get("$eq") {
                FieldPredicate::Equality(expected)
            } else {
                operator
                    .iter()
                    .next()
                    .filter(|(op, _)| op.starts_with('$'))
                    .map(|(op, value)| FieldPredicate::Operators(vec![compile_operator(op, value)]))
                    .unwrap_or(FieldPredicate::Equality(expected))
            }
        }
        Value::Object(operator) if operator.keys().any(|key| key.starts_with('$')) => {
            FieldPredicate::Operators(
                operator
                    .iter()
                    .map(|(op, value)| compile_operator(op.as_str(), value))
                    .collect(),
            )
        }
        _ => FieldPredicate::Equality(expected),
    }
}

fn compile_operator<'a>(op: &str, expected: &'a Value) -> FieldOperator<'a> {
    match op {
        "$eq" => FieldOperator::Eq(expected),
        "$ne" => FieldOperator::Ne(expected),
        "$gt" => FieldOperator::Gt(expected),
        "$gte" => FieldOperator::Gte(expected),
        "$lt" => FieldOperator::Lt(expected),
        "$lte" => FieldOperator::Lte(expected),
        "$prefix" => FieldOperator::Prefix(expected),
        "$exists" => FieldOperator::Exists(expected),
        _ => FieldOperator::Unknown,
    }
}

#[cfg(test)]
fn matches_filter_interpreted(document: &Document, filter: &Document) -> bool {
    if filter.len() == 1 {
        let (key, expected) = filter.iter().next().expect("filter has one entry");
        if let Some(expected) = equality_expected(expected) {
            return document.get(key) == Some(expected);
        }
    }

    filter
        .iter()
        .all(|(key, expected)| matches_field(document.get(key), expected))
}

#[cfg(test)]
fn equality_expected(expected: &Value) -> Option<&Value> {
    match expected {
        Value::Object(operator) if operator.len() == 1 => operator.get("$eq"),
        Value::Object(operator) if operator.keys().any(|key| key.starts_with('$')) => None,
        _ => Some(expected),
    }
}

#[cfg(test)]
fn matches_field(actual: Option<&Value>, expected: &Value) -> bool {
    match expected {
        Value::Object(operator) if operator.keys().any(|key| key.starts_with('$')) => operator
            .iter()
            .all(|(op, value)| matches_operator(actual, op.as_str(), value)),
        _ => actual == Some(expected),
    }
}

#[cfg(test)]
fn matches_operator(actual: Option<&Value>, op: &str, expected: &Value) -> bool {
    match op {
        "$eq" => actual == Some(expected),
        "$ne" => actual != Some(expected),
        "$gt" => compare_numbers(actual, expected, |left, right| left > right),
        "$gte" => compare_numbers(actual, expected, |left, right| left >= right),
        "$lt" => compare_numbers(actual, expected, |left, right| left < right),
        "$lte" => compare_numbers(actual, expected, |left, right| left <= right),
        "$prefix" => actual
            .and_then(Value::as_str)
            .zip(expected.as_str())
            .is_some_and(|(actual, expected)| actual.starts_with(expected)),
        "$exists" => expected
            .as_bool()
            .is_some_and(|should_exist| actual.is_some() == should_exist),
        _ => false,
    }
}

fn compare_numbers(
    actual: Option<&Value>,
    expected: &Value,
    predicate: impl FnOnce(f64, f64) -> bool,
) -> bool {
    let Some(actual) = actual.and_then(Value::as_f64) else {
        return false;
    };
    let Some(expected) = expected.as_f64() else {
        return false;
    };

    predicate(actual, expected)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn compiled_filter_matches_interpreted_filter() {
        let document = json!({
            "_id": "a",
            "pk": "room",
            "sk": "chat#001",
            "count": 7,
            "active": true,
            "nested": { "x": 1 }
        })
        .as_object()
        .expect("object")
        .clone();
        let filters = [
            json!({}),
            json!({ "pk": "room" }),
            json!({ "pk": { "$eq": "room" } }),
            json!({ "pk": { "$ne": "other" } }),
            json!({ "count": { "$gt": 6, "$lte": 7 } }),
            json!({ "sk": { "$prefix": "chat#" } }),
            json!({ "missing": { "$exists": false } }),
            json!({ "active": true, "pk": "room" }),
            json!({ "nested": { "x": 1 } }),
            json!({ "pk": { "$unknown": "room" } }),
        ];

        for filter in filters {
            let filter = filter.as_object().expect("filter object");
            assert_eq!(
                compile_filter(filter).matches(&document),
                matches_filter_interpreted(&document, filter)
            );
        }
    }
}
