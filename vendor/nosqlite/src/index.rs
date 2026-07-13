use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{mutation::CompiledUpdates, Document, Error, Result};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum ValueKey {
    Null,
    Bool(bool),
    String(String),
    I64(i64),
    U64(u64),
    F64(u64),
    Json(String),
}

impl ValueKey {
    fn new(value: &Value) -> Self {
        match value {
            Value::Null => Self::Null,
            Value::Bool(value) => Self::Bool(*value),
            Value::String(value) => Self::String(value.clone()),
            Value::Number(value) => {
                if let Some(value) = value.as_i64() {
                    Self::I64(value)
                } else if let Some(value) = value.as_u64() {
                    Self::U64(value)
                } else if let Some(value) = value.as_f64() {
                    Self::F64(value.to_bits())
                } else {
                    Self::Json(value.to_string())
                }
            }
            Value::Array(_) | Value::Object(_) => {
                Self::Json(serde_json::to_string(value).expect("serde_json::Value serializes"))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IndexKind {
    Exact,
    Neural,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexSpec {
    pub name: String,
    pub field: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fields: Option<Vec<String>>,
    pub kind: IndexKind,
    pub unique: bool,
    pub dimensions: Option<usize>,
}

#[derive(Debug)]
pub struct CollectionState {
    pub documents: Vec<Document>,
    indexes: IndexSet,
}

pub struct IndexedCandidates<'a> {
    pub positions: Vec<usize>,
    pub covered: bool,
    pub residual_filter_excluded_field: Option<&'a str>,
}

impl CollectionState {
    pub fn new(documents: Vec<Document>) -> Result<Self> {
        let mut state = Self {
            documents,
            indexes: IndexSet::default(),
        };
        state.indexes.add_primary();
        state.rebuild_indexes()?;
        Ok(state)
    }

    pub fn empty() -> Self {
        Self {
            documents: Vec::new(),
            indexes: IndexSet::with_primary(),
        }
    }

    pub fn create_indexes<I>(&mut self, specs: I) -> Result<()>
    where
        I: IntoIterator<Item = IndexSpec>,
    {
        self.indexes
            .create_and_append_indexed(specs, &self.documents)
    }

    pub fn index_specs(&self) -> Vec<IndexSpec> {
        self.indexes.specs()
    }

    pub fn rebuild_indexes(&mut self) -> Result<()> {
        self.indexes.rebuild(&self.documents)
    }

    pub fn index_document(&mut self, position: usize) -> Result<()> {
        let document = self
            .documents
            .get(position)
            .ok_or(Error::MissingField("document"))?;
        self.indexes.index_document(position, document)
    }

    pub fn update_documents<'a, I>(&mut self, changes: I) -> Result<()>
    where
        I: IntoIterator<Item = (usize, &'a Document, &'a Document)>,
    {
        let changes = changes.into_iter().collect::<Vec<_>>();
        let mut indexes = self.indexes.clone();
        indexes.update_documents(changes)?;
        self.indexes = indexes;
        Ok(())
    }

    pub fn update_documents_in_place<'a, I>(&mut self, changes: I) -> Result<()>
    where
        I: IntoIterator<Item = (usize, &'a Document)>,
    {
        let documents = &self.documents;
        let indexes = &mut self.indexes;
        for (position, old_document) in changes {
            let Some(new_document) = documents.get(position) else {
                continue;
            };
            indexes.update_document_in_place(position, old_document, new_document)?;
        }
        Ok(())
    }

    pub fn updates_affect_indexes(&self, updates: &CompiledUpdates<'_>) -> bool {
        self.indexes.updates_affect_indexes(updates)
    }

    pub fn updates_can_update_indexes_in_place(&self, updates: &CompiledUpdates<'_>) -> bool {
        self.indexes.updates_can_update_in_place(updates)
    }

    pub fn indexed_candidates_with_coverage<'a>(
        &'a self,
        filter: &Document,
    ) -> Option<IndexedCandidates<'a>> {
        self.indexes.exact_candidates_with_coverage(filter)
    }

    pub fn indexed_candidates_with_residual_hint<'a>(
        &'a self,
        filter: &Document,
    ) -> Option<IndexedCandidates<'a>> {
        self.indexes.exact_candidates_with_residual_hint(filter)
    }

    pub fn indexed_candidate_count(&self, filter: &Document) -> Option<usize> {
        self.indexes.exact_candidate_count(filter)
    }

    pub fn indexed_page_candidates(
        &self,
        filter: &Document,
        end: usize,
    ) -> Option<(Vec<usize>, bool)> {
        self.indexes.exact_page_candidates(filter, end)
    }

    pub fn ordered_page_candidates(
        &self,
        filter: &Document,
        sort_field: &str,
        ascending: bool,
        end: usize,
    ) -> Option<(Vec<usize>, bool, bool)> {
        self.indexes
            .ordered_page_candidates(filter, sort_field, ascending, end)
    }

    pub fn has_exact_index_fields(&self, fields: &[String]) -> bool {
        self.indexes.has_exact_fields(fields)
    }

    pub fn internal_exact_index_count(&self) -> usize {
        self.indexes.internal_exact_count()
    }

    pub fn create_internal_exact_index(&mut self, fields: Vec<String>) -> Result<()> {
        self.indexes
            .create_internal_exact_indexed(fields, &self.documents)
    }

    pub fn delete_positions(&mut self, positions: &[usize]) {
        if positions.is_empty() {
            return;
        }
        let remap = PositionRemap::new(self.documents.len(), positions);
        if positions.len() <= 8 {
            for position in positions.iter().rev() {
                if *position < self.documents.len() {
                    self.documents.remove(*position);
                }
            }
            self.indexes.remap_after_delete(&remap);
            return;
        }

        let mut deleted = positions.iter().copied().peekable();
        let mut position = 0;
        self.documents.retain(|_| {
            while deleted
                .peek()
                .is_some_and(|deleted_position| *deleted_position < position)
            {
                deleted.next();
            }
            let remove = deleted
                .peek()
                .is_some_and(|deleted_position| *deleted_position == position);
            if remove {
                deleted.next();
            }
            position += 1;
            !remove
        });
        self.indexes.remap_after_delete(&remap);
    }

    pub fn neural_candidates(
        &self,
        index_name: Option<&str>,
        field: Option<&str>,
        query: &[f64],
        limit: Option<usize>,
    ) -> Result<Vec<(usize, f64)>> {
        self.indexes
            .neural_candidates(index_name, field, query, limit)
    }
}

#[derive(Debug, Default, Clone)]
struct IndexSet {
    exact: Vec<ExactIndex>,
    neural: Vec<NeuralIndex>,
}

impl IndexSet {
    fn with_primary() -> Self {
        let mut indexes = Self::default();
        indexes.add_primary();
        indexes
    }

    fn add_primary(&mut self) {
        if self.exact.iter().any(|index| index.name == "_id_") {
            return;
        }
        self.exact.push(ExactIndex::new("_id_", "_id", true));
    }

    fn create(&mut self, spec: IndexSpec) -> Result<()> {
        if self.exact.iter().any(|index| index.name == spec.name)
            || self.neural.iter().any(|index| index.name == spec.name)
        {
            return Err(Error::IndexExists(spec.name));
        }

        match spec.kind {
            IndexKind::Exact => {
                if let Some(fields) = spec.fields {
                    self.exact
                        .push(ExactIndex::new_compound(&spec.name, fields, spec.unique));
                } else {
                    self.exact
                        .push(ExactIndex::new(&spec.name, &spec.field, spec.unique));
                }
            }
            IndexKind::Neural => {
                let dimensions = spec.dimensions.ok_or(Error::MissingField("dimensions"))?;
                self.neural
                    .push(NeuralIndex::new(&spec.name, &spec.field, dimensions));
            }
        }

        Ok(())
    }

    fn create_and_append_indexed<I>(&mut self, specs: I, documents: &[Document]) -> Result<()>
    where
        I: IntoIterator<Item = IndexSpec>,
    {
        let mut additions = IndexSet::default();
        for spec in specs {
            if self.has_name(&spec.name) {
                return Err(Error::IndexExists(spec.name));
            }
            additions.create(spec)?;
        }
        additions.rebuild(documents)?;
        self.exact.extend(additions.exact);
        self.neural.extend(additions.neural);
        Ok(())
    }

    fn has_name(&self, name: &str) -> bool {
        self.exact.iter().any(|index| index.name == name)
            || self.neural.iter().any(|index| index.name == name)
    }

    fn rebuild(&mut self, documents: &[Document]) -> Result<()> {
        for index in &mut self.exact {
            index.clear();
        }
        for index in &mut self.neural {
            index.clear();
        }

        for (position, document) in documents.iter().enumerate() {
            self.index_document(position, document)?;
        }

        Ok(())
    }

    fn index_document(&mut self, position: usize, document: &Document) -> Result<()> {
        for index in &mut self.exact {
            index.index_document(position, document)?;
        }
        for index in &mut self.neural {
            index.index_document(position, document)?;
        }
        Ok(())
    }

    fn update_documents<'a, I>(&mut self, changes: I) -> Result<()>
    where
        I: IntoIterator<Item = (usize, &'a Document, &'a Document)>,
    {
        let changes = changes.into_iter().collect::<Vec<_>>();
        for (position, old_document, _) in &changes {
            for index in &mut self.exact {
                index.remove_document(*position, old_document);
            }
            for index in &mut self.neural {
                index.remove_document(*position, old_document);
            }
        }

        for (position, _, new_document) in changes {
            for index in &mut self.exact {
                index.index_document(position, new_document)?;
            }
            for index in &mut self.neural {
                index.index_document(position, new_document)?;
            }
        }
        Ok(())
    }

    fn update_document_in_place(
        &mut self,
        position: usize,
        old_document: &Document,
        new_document: &Document,
    ) -> Result<()> {
        for index in &mut self.exact {
            index.update_document_in_place(position, old_document, new_document)?;
        }
        Ok(())
    }

    fn remap_after_delete(&mut self, remap: &PositionRemap<'_>) {
        for index in &mut self.exact {
            index.remap_after_delete(remap);
        }
        for index in &mut self.neural {
            index.remap_after_delete(remap);
        }
    }

    fn specs(&self) -> Vec<IndexSpec> {
        self.exact
            .iter()
            .filter(|index| index.name != "_id_" && !index.name.starts_with("__auto_"))
            .map(ExactIndex::spec)
            .chain(self.neural.iter().map(NeuralIndex::spec))
            .collect()
    }

    fn exact_candidates_with_coverage<'a>(
        &'a self,
        filter: &Document,
    ) -> Option<IndexedCandidates<'a>> {
        let index = self.best_exact_index(filter)?;
        let positions = index.lookup_filter(filter)?;
        let covered = page_filter_fields_covered(filter, &index.fields);
        Some(IndexedCandidates {
            positions,
            covered,
            residual_filter_excluded_field: None,
        })
    }

    fn exact_candidates_with_residual_hint<'a>(
        &'a self,
        filter: &Document,
    ) -> Option<IndexedCandidates<'a>> {
        let index = self.best_exact_index(filter)?;
        let positions = index.lookup_filter(filter)?;
        let covered = page_filter_fields_covered(filter, &index.fields);
        let residual_filter_excluded_field = (!covered)
            .then(|| index.single_equality_field(filter))
            .flatten();
        Some(IndexedCandidates {
            positions,
            covered,
            residual_filter_excluded_field,
        })
    }

    fn best_exact_index(&self, filter: &Document) -> Option<&ExactIndex> {
        let mut best: Option<(usize, usize)> = None;

        for (index_position, index) in self.exact.iter().enumerate() {
            let Some(candidate_len) = index.lookup_filter_len(filter) else {
                continue;
            };
            if best.is_none_or(|(_, existing_len)| candidate_len < existing_len) {
                best = Some((index_position, candidate_len));
                if candidate_len == 0 {
                    break;
                }
            }
        }

        let (index_position, _) = best?;
        self.exact.get(index_position)
    }

    fn exact_candidate_count(&self, filter: &Document) -> Option<usize> {
        let mut best: Option<usize> = None;

        for index in &self.exact {
            let Some(candidate_len) = index.lookup_covered_filter_len(filter) else {
                continue;
            };
            if best.is_none_or(|existing_len| candidate_len < existing_len) {
                best = Some(candidate_len);
                if candidate_len == 0 {
                    break;
                }
            }
        }

        best
    }

    fn exact_page_candidates(&self, filter: &Document, end: usize) -> Option<(Vec<usize>, bool)> {
        for index in &self.exact {
            if let Some(candidates) = index.lookup_filter_page(filter, end) {
                return Some(candidates);
            }
        }
        None
    }

    fn ordered_page_candidates(
        &self,
        filter: &Document,
        sort_field: &str,
        ascending: bool,
        end: usize,
    ) -> Option<(Vec<usize>, bool, bool)> {
        for index in &self.exact {
            if let Some(candidates) =
                index.ordered_page_candidates(filter, sort_field, ascending, end)
            {
                return Some(candidates);
            }
        }
        None
    }

    fn has_exact_fields(&self, fields: &[String]) -> bool {
        self.exact.iter().any(|index| index.fields == fields)
    }

    fn internal_exact_count(&self) -> usize {
        self.exact
            .iter()
            .filter(|index| index.name.starts_with("__auto_"))
            .count()
    }

    fn create_internal_exact_indexed(
        &mut self,
        fields: Vec<String>,
        documents: &[Document],
    ) -> Result<()> {
        if fields.is_empty() || self.has_exact_fields(&fields) {
            return Ok(());
        }

        let name = auto_index_name(&fields);
        if self.exact.iter().any(|index| index.name == name) {
            return Ok(());
        }

        let mut index = if fields.len() == 1 {
            ExactIndex::new(&name, &fields[0], false)
        } else {
            ExactIndex::new_compound(&name, fields, false)
        };
        index.rebuild(documents)?;
        self.exact.push(index);
        Ok(())
    }

    fn updates_affect_indexes(&self, updates: &CompiledUpdates<'_>) -> bool {
        updates.fields().any(|field| {
            self.exact
                .iter()
                .any(|index| index.fields.iter().any(|indexed| indexed == field))
                || self.neural.iter().any(|index| index.field == field)
        })
    }

    fn updates_can_update_in_place(&self, updates: &CompiledUpdates<'_>) -> bool {
        !updates.fields().any(|field| {
            self.exact
                .iter()
                .any(|index| index.unique && index.fields.iter().any(|indexed| indexed == field))
                || self.neural.iter().any(|index| index.field == field)
        })
    }

    fn neural_candidates(
        &self,
        index_name: Option<&str>,
        field: Option<&str>,
        query: &[f64],
        limit: Option<usize>,
    ) -> Result<Vec<(usize, f64)>> {
        let index = self
            .neural
            .iter()
            .find(|index| {
                index_name.is_none_or(|name| index.name == name)
                    && field.is_none_or(|field| index.field == field)
            })
            .ok_or_else(|| {
                Error::IndexMissing(index_name.or(field).unwrap_or("<neural>").to_string())
            })?;

        index.search(query, limit)
    }
}

fn auto_index_name(fields: &[String]) -> String {
    let fields = fields
        .iter()
        .map(|field| {
            field
                .chars()
                .map(|ch| match ch {
                    'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' => ch,
                    _ => '_',
                })
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("_");
    format!("__auto_{fields}")
}

#[derive(Debug, Clone)]
struct ExactIndex {
    name: String,
    field: String,
    fields: Vec<String>,
    unique: bool,
    entries: ExactEntries,
}

#[derive(Debug, Clone)]
enum ExactEntries {
    Unique(HashMap<ValueKey, usize>),
    Multi(HashMap<ValueKey, Vec<usize>>),
    Compound(BTreeMap<Vec<ValueKey>, Vec<usize>>),
}

impl ExactIndex {
    fn new(name: &str, field: &str, unique: bool) -> Self {
        Self {
            name: name.to_string(),
            field: field.to_string(),
            fields: vec![field.to_string()],
            unique,
            entries: if unique {
                ExactEntries::Unique(HashMap::new())
            } else {
                ExactEntries::Multi(HashMap::new())
            },
        }
    }

    fn new_compound(name: &str, fields: Vec<String>, unique: bool) -> Self {
        let field = fields.first().cloned().unwrap_or_default();
        Self {
            name: name.to_string(),
            field,
            fields,
            unique,
            entries: ExactEntries::Compound(BTreeMap::new()),
        }
    }

    fn rebuild(&mut self, documents: &[Document]) -> Result<()> {
        self.clear();
        for (position, document) in documents.iter().enumerate() {
            self.index_document(position, document)?;
        }
        Ok(())
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    fn index_document(&mut self, position: usize, document: &Document) -> Result<()> {
        let fields = &self.fields;
        match &mut self.entries {
            ExactEntries::Unique(entries) => {
                let Some(key) = document.get(&self.field).map(ValueKey::new) else {
                    return Ok(());
                };
                if entries.insert(key, position).is_some() {
                    return Err(Error::UniqueIndexViolation {
                        index: self.name.clone(),
                        field: self.field.clone(),
                    });
                }
            }
            ExactEntries::Multi(entries) => {
                let Some(key) = document.get(&self.field).map(ValueKey::new) else {
                    return Ok(());
                };
                insert_sorted_position(entries.entry(key).or_default(), position);
            }
            ExactEntries::Compound(entries) => {
                let Some(key) = document_key(fields, document) else {
                    return Ok(());
                };
                let positions = entries.entry(key).or_default();
                if self.unique && !positions.is_empty() {
                    return Err(Error::UniqueIndexViolation {
                        index: self.name.clone(),
                        field: self.fields.join(","),
                    });
                }
                insert_sorted_position(positions, position);
            }
        }
        Ok(())
    }

    fn lookup_filter(&self, filter: &Document) -> Option<Vec<usize>> {
        match &self.entries {
            ExactEntries::Unique(entries) => {
                let value = equality_filter_value(filter, &self.field)?;
                Some(
                    entries
                        .get(&ValueKey::new(value))
                        .copied()
                        .into_iter()
                        .collect(),
                )
            }
            ExactEntries::Multi(entries) => {
                let value = equality_filter_value(filter, &self.field)?;
                Some(
                    entries
                        .get(&ValueKey::new(value))
                        .cloned()
                        .unwrap_or_default(),
                )
            }
            ExactEntries::Compound(entries) => self.lookup_compound(filter, entries),
        }
    }

    fn lookup_filter_len(&self, filter: &Document) -> Option<usize> {
        match &self.entries {
            ExactEntries::Unique(entries) => {
                let value = equality_filter_value(filter, &self.field)?;
                Some(usize::from(entries.contains_key(&ValueKey::new(value))))
            }
            ExactEntries::Multi(entries) => {
                let value = equality_filter_value(filter, &self.field)?;
                Some(entries.get(&ValueKey::new(value)).map_or(0, Vec::len))
            }
            ExactEntries::Compound(entries) => self.lookup_compound_len(filter, entries),
        }
    }

    fn single_equality_field<'a>(&'a self, filter: &Document) -> Option<&'a str> {
        match &self.entries {
            ExactEntries::Unique(_) | ExactEntries::Multi(_) => {
                page_equality_filter_value(filter, &self.field)?;
                Some(&self.field)
            }
            ExactEntries::Compound(_) => None,
        }
    }

    fn lookup_covered_filter_len(&self, filter: &Document) -> Option<usize> {
        if !page_filter_fields_covered(filter, &self.fields) {
            return None;
        }
        self.lookup_filter_len(filter)
    }

    fn lookup_filter_page(&self, filter: &Document, end: usize) -> Option<(Vec<usize>, bool)> {
        if !page_filter_fields_covered(filter, &self.fields) {
            return None;
        }

        match &self.entries {
            ExactEntries::Unique(entries) => {
                let value = page_equality_filter_value(filter, &self.field)?;
                let positions = entries
                    .get(&ValueKey::new(value))
                    .copied()
                    .into_iter()
                    .collect();
                Some((positions, false))
            }
            ExactEntries::Multi(entries) => {
                let value = page_equality_filter_value(filter, &self.field)?;
                let Some(positions) = entries.get(&ValueKey::new(value)) else {
                    return Some((Vec::new(), false));
                };
                Some(take_positions_page(positions.iter().copied(), end))
            }
            ExactEntries::Compound(entries) => self.lookup_compound_page(filter, entries, end),
        }
    }

    fn ordered_page_candidates(
        &self,
        filter: &Document,
        sort_field: &str,
        ascending: bool,
        end: usize,
    ) -> Option<(Vec<usize>, bool, bool)> {
        if !ascending
            || self.fields.last()? != sort_field
            || !filter_fields_covered(filter, &self.fields)
        {
            return None;
        }
        let ExactEntries::Compound(entries) = &self.entries else {
            return None;
        };
        let (prefix, string_prefix) = compound_order_prefix(filter, &self.fields)?;
        let mut positions = Vec::new();
        let mut tie_free = true;
        for (key, group) in entries
            .range(prefix.clone()..)
            .take_while(|(key, _)| compound_key_has_prefix(key, &prefix))
            .filter(|(key, _)| {
                string_prefix.as_deref().is_none_or(|string_prefix| {
                    compound_key_matches_string_prefix(key, &prefix, string_prefix)
                })
            })
        {
            let _ = key;
            tie_free &= group.len() <= 1;
            positions.extend(group.iter().copied());
            if positions.len() > end {
                return Some((positions, true, tie_free));
            }
        }
        Some((positions, false, tie_free))
    }

    fn remove_document(&mut self, position: usize, document: &Document) {
        let fields = &self.fields;
        match &mut self.entries {
            ExactEntries::Unique(entries) => {
                let Some(key) = document.get(&self.field).map(ValueKey::new) else {
                    return;
                };
                if entries.get(&key).copied() == Some(position) {
                    entries.remove(&key);
                }
            }
            ExactEntries::Multi(entries) => {
                let Some(key) = document.get(&self.field).map(ValueKey::new) else {
                    return;
                };
                if let Some(positions) = entries.get_mut(&key) {
                    remove_sorted_position(positions, position);
                    if positions.is_empty() {
                        entries.remove(&key);
                    }
                }
            }
            ExactEntries::Compound(entries) => {
                let Some(key) = document_key(fields, document) else {
                    return;
                };
                if let Some(positions) = entries.get_mut(&key) {
                    remove_sorted_position(positions, position);
                    if positions.is_empty() {
                        entries.remove(&key);
                    }
                }
            }
        }
    }

    fn update_document_in_place(
        &mut self,
        position: usize,
        old_document: &Document,
        new_document: &Document,
    ) -> Result<()> {
        if self.document_keys_equal(old_document, new_document) {
            return Ok(());
        }
        self.remove_document(position, old_document);
        self.index_document(position, new_document)
    }

    fn document_keys_equal(&self, left: &Document, right: &Document) -> bool {
        match &self.entries {
            ExactEntries::Unique(_) | ExactEntries::Multi(_) => {
                scalar_document_key(&self.field, left) == scalar_document_key(&self.field, right)
            }
            ExactEntries::Compound(_) => {
                document_key(&self.fields, left) == document_key(&self.fields, right)
            }
        }
    }

    fn remap_after_delete(&mut self, remap: &PositionRemap<'_>) {
        match &mut self.entries {
            ExactEntries::Unique(entries) => {
                let mut remove_keys = Vec::new();
                for (key, position) in entries.iter_mut() {
                    if let Some(remapped) = remap.position(*position) {
                        *position = remapped;
                    } else {
                        remove_keys.push(key.clone());
                    }
                }
                for key in remove_keys {
                    entries.remove(&key);
                }
            }
            ExactEntries::Multi(entries) => {
                let mut remove_keys = Vec::new();
                for (key, positions) in entries.iter_mut() {
                    remap_positions_after_delete(positions, remap);
                    if positions.is_empty() {
                        remove_keys.push(key.clone());
                    }
                }
                for key in remove_keys {
                    entries.remove(&key);
                }
            }
            ExactEntries::Compound(entries) => {
                let mut remove_keys = Vec::new();
                for (key, positions) in entries.iter_mut() {
                    remap_positions_after_delete(positions, remap);
                    if positions.is_empty() {
                        remove_keys.push(key.clone());
                    }
                }
                for key in remove_keys {
                    entries.remove(&key);
                }
            }
        }
    }

    fn lookup_compound(
        &self,
        filter: &Document,
        entries: &BTreeMap<Vec<ValueKey>, Vec<usize>>,
    ) -> Option<Vec<usize>> {
        let (prefix, string_prefix) = compound_filter_prefix(filter, &self.fields)?;
        if let Some(string_prefix) = string_prefix {
            return Some(
                entries
                    .range(prefix.clone()..)
                    .take_while(|(key, _)| compound_key_has_prefix(key, &prefix))
                    .filter(|(key, _)| {
                        compound_key_matches_string_prefix(key, &prefix, &string_prefix)
                    })
                    .flat_map(|(_, positions)| positions.iter().copied())
                    .collect(),
            );
        }

        Some(entries.get(&prefix).cloned().unwrap_or_default())
    }

    fn lookup_compound_len(
        &self,
        filter: &Document,
        entries: &BTreeMap<Vec<ValueKey>, Vec<usize>>,
    ) -> Option<usize> {
        let (prefix, string_prefix) = compound_filter_prefix(filter, &self.fields)?;
        if let Some(string_prefix) = string_prefix {
            return Some(
                entries
                    .range(prefix.clone()..)
                    .take_while(|(key, _)| compound_key_has_prefix(key, &prefix))
                    .filter(|(key, _)| {
                        compound_key_matches_string_prefix(key, &prefix, &string_prefix)
                    })
                    .map(|(_, positions)| positions.len())
                    .sum(),
            );
        }

        Some(entries.get(&prefix).map_or(0, Vec::len))
    }

    fn lookup_compound_page(
        &self,
        filter: &Document,
        entries: &BTreeMap<Vec<ValueKey>, Vec<usize>>,
        end: usize,
    ) -> Option<(Vec<usize>, bool)> {
        let (prefix, string_prefix) = compound_page_filter_prefix(filter, &self.fields)?;
        if let Some(string_prefix) = string_prefix {
            let mut positions = Vec::new();
            for (_, group) in entries
                .range(prefix.clone()..)
                .take_while(|(key, _)| compound_key_has_prefix(key, &prefix))
                .filter(|(key, _)| compound_key_matches_string_prefix(key, &prefix, &string_prefix))
            {
                positions.extend(group.iter().copied());
                if positions.len() > end {
                    positions.truncate(end);
                    return Some((positions, true));
                }
            }
            return Some((positions, false));
        }

        let Some(positions) = entries.get(&prefix) else {
            return Some((Vec::new(), false));
        };
        Some(take_positions_page(positions.iter().copied(), end))
    }

    fn spec(&self) -> IndexSpec {
        IndexSpec {
            name: self.name.clone(),
            field: self.field.clone(),
            fields: (self.fields.len() > 1).then(|| self.fields.clone()),
            kind: IndexKind::Exact,
            unique: self.unique,
            dimensions: None,
        }
    }
}

impl ExactEntries {
    fn clear(&mut self) {
        match self {
            ExactEntries::Unique(entries) => entries.clear(),
            ExactEntries::Multi(entries) => entries.clear(),
            ExactEntries::Compound(entries) => entries.clear(),
        }
    }
}

#[derive(Debug, Clone)]
struct NeuralIndex {
    name: String,
    field: String,
    dimensions: usize,
    planes: Vec<Vec<f64>>,
    buckets: BTreeMap<u64, BTreeSet<usize>>,
    vectors: Vec<Option<Vec<f64>>>,
}

impl NeuralIndex {
    const PLANES: usize = 16;

    fn new(name: &str, field: &str, dimensions: usize) -> Self {
        Self {
            name: name.to_string(),
            field: field.to_string(),
            dimensions,
            planes: projection_planes(name.as_bytes(), dimensions),
            buckets: BTreeMap::new(),
            vectors: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.buckets.clear();
        self.vectors.clear();
    }

    fn index_document(&mut self, position: usize, document: &Document) -> Result<()> {
        let Some(value) = document.get(&self.field) else {
            return Ok(());
        };
        let vector = json_vector(value).ok_or(Error::ExpectedArray("vector field"))?;
        if vector.len() != self.dimensions {
            return Err(Error::VectorDimensionMismatch {
                field: self.field.clone(),
                expected: self.dimensions,
                actual: vector.len(),
            });
        }
        let signature = self.signature(&vector);
        self.buckets.entry(signature).or_default().insert(position);
        if position >= self.vectors.len() {
            self.vectors.resize_with(position + 1, || None);
        }
        self.vectors[position] = Some(normalized_vector(vector));
        Ok(())
    }

    fn remove_document(&mut self, position: usize, document: &Document) {
        let Some(value) = document.get(&self.field) else {
            return;
        };
        let Some(vector) = json_vector(value) else {
            if let Some(vector) = self.vectors.get_mut(position) {
                *vector = None;
            }
            return;
        };
        if vector.len() == self.dimensions {
            let signature = self.signature(&vector);
            if let Some(bucket) = self.buckets.get_mut(&signature) {
                bucket.remove(&position);
                if bucket.is_empty() {
                    self.buckets.remove(&signature);
                }
            }
        }
        if let Some(vector) = self.vectors.get_mut(position) {
            *vector = None;
        }
    }

    fn remap_after_delete(&mut self, remap: &PositionRemap<'_>) {
        self.vectors = std::mem::take(&mut self.vectors)
            .into_iter()
            .enumerate()
            .filter_map(|(position, vector)| remap.position(position).map(|_| vector))
            .collect();
        self.buckets = self
            .buckets
            .iter()
            .filter_map(|(signature, positions)| {
                let positions = positions
                    .iter()
                    .filter_map(|position| remap.position(*position))
                    .collect::<BTreeSet<_>>();
                (!positions.is_empty()).then_some((*signature, positions))
            })
            .collect();
    }

    fn search(&self, query: &[f64], limit: Option<usize>) -> Result<Vec<(usize, f64)>> {
        if query.len() != self.dimensions {
            return Err(Error::VectorDimensionMismatch {
                field: self.field.clone(),
                expected: self.dimensions,
                actual: query.len(),
            });
        }

        let query_norm = vector_norm(query);
        let signature = self.signature(query);
        let mut candidates = self
            .buckets
            .get(&signature)
            .map(|bucket| bucket.iter().copied().collect::<Vec<_>>())
            .unwrap_or_default();
        let mut needs_dedup = false;

        // Projection buckets are a route, not a truth oracle. If the exact bucket is sparse,
        // widen by Hamming distance so small datasets still behave predictably.
        if candidates.len() < 32 {
            for bit in 0..Self::PLANES {
                if candidates.len() >= 32 {
                    break;
                }
                if let Some(bucket) = self.buckets.get(&(signature ^ (1 << bit))) {
                    candidates.extend(bucket.iter().copied());
                    needs_dedup = true;
                }
            }
        }

        if candidates.len() < 32 {
            let mut scored = Vec::with_capacity(limit.unwrap_or(self.vectors.len()));
            for (position, vector) in self.vectors.iter().enumerate() {
                let Some(vector) = vector else {
                    continue;
                };
                let candidate = (position, normalized_query_dot(query, query_norm, vector));
                if let Some(limit) = limit {
                    push_top_scored_candidate(&mut scored, candidate, limit);
                } else {
                    scored.push(candidate);
                }
            }
            return Ok(rank_scored_candidates(scored, limit));
        }

        if needs_dedup {
            candidates.sort_unstable();
            candidates.dedup();
        }

        let scored = candidates
            .into_iter()
            .filter_map(|position| {
                self.vectors
                    .get(position)
                    .and_then(Option::as_ref)
                    .map(|vector| (position, normalized_query_dot(query, query_norm, vector)))
            })
            .collect::<Vec<_>>();
        Ok(rank_scored_candidates(scored, limit))
    }

    fn signature(&self, vector: &[f64]) -> u64 {
        let mut signature = 0_u64;
        for (plane, projection) in self.planes.iter().enumerate() {
            let dot = dot_product(vector, projection);
            if dot >= 0.0 {
                signature |= 1 << plane;
            }
        }
        signature
    }

    fn spec(&self) -> IndexSpec {
        IndexSpec {
            name: self.name.clone(),
            field: self.field.clone(),
            fields: None,
            kind: IndexKind::Neural,
            unique: false,
            dimensions: Some(self.dimensions),
        }
    }
}

fn equality_filter_value<'a>(filter: &'a Document, field: &str) -> Option<&'a Value> {
    match filter.get(field)? {
        Value::Object(operator) => operator.get("$eq"),
        value => Some(value),
    }
}

fn prefix_filter_value<'a>(filter: &'a Document, field: &str) -> Option<&'a str> {
    filter.get(field)?.as_object()?.get("$prefix")?.as_str()
}

fn page_equality_filter_value<'a>(filter: &'a Document, field: &str) -> Option<&'a Value> {
    match filter.get(field)? {
        Value::Object(operator) if operator.len() == 1 => operator.get("$eq"),
        Value::Object(_) => None,
        value => Some(value),
    }
}

fn page_prefix_filter_value<'a>(filter: &'a Document, field: &str) -> Option<&'a str> {
    let operator = filter.get(field)?.as_object()?;
    (operator.len() == 1)
        .then(|| operator.get("$prefix")?.as_str())
        .flatten()
}

fn scalar_document_key(field: &str, document: &Document) -> Option<ValueKey> {
    document.get(field).map(ValueKey::new)
}

fn document_key(fields: &[String], document: &Document) -> Option<Vec<ValueKey>> {
    fields
        .iter()
        .map(|field| document.get(field).map(ValueKey::new))
        .collect()
}

fn compound_filter_prefix(
    filter: &Document,
    fields: &[String],
) -> Option<(Vec<ValueKey>, Option<String>)> {
    if fields.is_empty() {
        return None;
    }

    let mut prefix = Vec::with_capacity(fields.len());
    for (index, field) in fields.iter().enumerate() {
        if let Some(value) = equality_filter_value(filter, field) {
            prefix.push(ValueKey::new(value));
            continue;
        }
        if index == fields.len() - 1 {
            return Some((
                prefix,
                Some(prefix_filter_value(filter, field)?.to_string()),
            ));
        }
        return None;
    }

    Some((prefix, None))
}

fn compound_page_filter_prefix(
    filter: &Document,
    fields: &[String],
) -> Option<(Vec<ValueKey>, Option<String>)> {
    if fields.is_empty() {
        return None;
    }

    let mut prefix = Vec::with_capacity(fields.len());
    for (index, field) in fields.iter().enumerate() {
        if let Some(value) = page_equality_filter_value(filter, field) {
            prefix.push(ValueKey::new(value));
            continue;
        }
        if index == fields.len() - 1 {
            return Some((
                prefix,
                Some(page_prefix_filter_value(filter, field)?.to_string()),
            ));
        }
        return None;
    }

    Some((prefix, None))
}

fn compound_order_prefix(
    filter: &Document,
    fields: &[String],
) -> Option<(Vec<ValueKey>, Option<String>)> {
    if fields.len() < 2 {
        return None;
    }

    let mut prefix = Vec::with_capacity(fields.len() - 1);
    for field in &fields[..fields.len() - 1] {
        prefix.push(ValueKey::new(equality_filter_value(filter, field)?));
    }

    let sort_field = fields.last()?;
    if let Some(value) = equality_filter_value(filter, sort_field) {
        prefix.push(ValueKey::new(value));
        return Some((prefix, None));
    }
    let string_prefix = prefix_filter_value(filter, sort_field).map(str::to_string);
    if filter.get(sort_field).is_some() && string_prefix.is_none() {
        return None;
    }
    Some((prefix, string_prefix))
}

fn filter_fields_covered(filter: &Document, fields: &[String]) -> bool {
    filter
        .keys()
        .all(|field| fields.iter().any(|indexed| indexed == field))
}

fn page_filter_fields_covered(filter: &Document, fields: &[String]) -> bool {
    filter.iter().all(|(field, _)| {
        fields.iter().any(|indexed| indexed == field)
            && (page_equality_filter_value(filter, field).is_some()
                || page_prefix_filter_value(filter, field).is_some())
    })
}

fn compound_key_has_prefix(key: &[ValueKey], prefix: &[ValueKey]) -> bool {
    key.starts_with(prefix)
}

fn compound_key_matches_string_prefix(
    key: &[ValueKey],
    prefix: &[ValueKey],
    string_prefix: &str,
) -> bool {
    if key.len() != prefix.len() + 1 || !compound_key_has_prefix(key, prefix) {
        return false;
    }
    matches!(
        key.last(),
        Some(ValueKey::String(value)) if value.starts_with(string_prefix)
    )
}

fn take_positions_page(positions: impl Iterator<Item = usize>, end: usize) -> (Vec<usize>, bool) {
    let mut page = Vec::with_capacity(end);
    for position in positions {
        if page.len() == end {
            return (page, true);
        }
        page.push(position);
    }
    (page, false)
}

enum PositionRemap<'a> {
    Single(usize),
    Search(&'a [usize]),
    Dense(Vec<usize>),
}

impl<'a> PositionRemap<'a> {
    const DELETED: usize = usize::MAX;

    fn new(old_len: usize, deleted_positions: &'a [usize]) -> Self {
        if deleted_positions.len() == 1 {
            return Self::Single(deleted_positions[0]);
        }
        if deleted_positions.len() <= 8 {
            return Self::Search(deleted_positions);
        }

        let mut deleted = deleted_positions.iter().copied().peekable();
        let mut shift = 0;
        let mut positions = Vec::with_capacity(old_len);
        for position in 0..old_len {
            while deleted
                .peek()
                .is_some_and(|deleted_position| *deleted_position < position)
            {
                deleted.next();
                shift += 1;
            }
            if deleted
                .peek()
                .is_some_and(|deleted_position| *deleted_position == position)
            {
                deleted.next();
                shift += 1;
                positions.push(Self::DELETED);
            } else {
                positions.push(position - shift);
            }
        }
        Self::Dense(positions)
    }

    fn position(&self, position: usize) -> Option<usize> {
        match self {
            Self::Single(deleted) => {
                if position == *deleted {
                    None
                } else if position > *deleted {
                    Some(position - 1)
                } else {
                    Some(position)
                }
            }
            Self::Search(deleted_positions) => match deleted_positions.binary_search(&position) {
                Ok(_) => None,
                Err(shift) => Some(position - shift),
            },
            Self::Dense(positions) => positions
                .get(position)
                .copied()
                .filter(|position| *position != Self::DELETED),
        }
    }
}

fn remap_positions_after_delete(positions: &mut Vec<usize>, remap: &PositionRemap<'_>) {
    positions.retain_mut(|position| {
        let Some(remapped) = remap.position(*position) else {
            return false;
        };
        *position = remapped;
        true
    });
}

fn remove_sorted_position(positions: &mut Vec<usize>, position: usize) {
    if let Ok(index) = positions.binary_search(&position) {
        positions.remove(index);
    }
}

fn insert_sorted_position(positions: &mut Vec<usize>, position: usize) {
    if positions.last().is_none_or(|last| *last < position) {
        positions.push(position);
        return;
    }

    match positions.binary_search(&position) {
        Ok(_) => {}
        Err(index) => positions.insert(index, position),
    }
}

fn compare_scored_candidates(left: &(usize, f64), right: &(usize, f64)) -> std::cmp::Ordering {
    right
        .1
        .partial_cmp(&left.1)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| left.0.cmp(&right.0))
}

fn rank_scored_candidates(
    mut scored: Vec<(usize, f64)>,
    limit: Option<usize>,
) -> Vec<(usize, f64)> {
    if let Some(limit) = limit {
        if limit < scored.len() {
            scored.select_nth_unstable_by(limit, compare_scored_candidates);
            scored.truncate(limit);
        }
    }
    scored.sort_by(compare_scored_candidates);
    scored
}

fn push_top_scored_candidate(
    scored: &mut Vec<(usize, f64)>,
    candidate: (usize, f64),
    limit: usize,
) {
    if limit == 0 {
        return;
    }
    if scored.len() < limit {
        scored.push(candidate);
        return;
    }
    let Some((worst_index, worst)) = scored
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| compare_scored_candidates(left, right))
    else {
        return;
    };
    if compare_scored_candidates(&candidate, worst).is_lt() {
        scored[worst_index] = candidate;
    }
}

pub fn json_vector(value: &Value) -> Option<Vec<f64>> {
    value
        .as_array()?
        .iter()
        .map(Value::as_f64)
        .collect::<Option<Vec<_>>>()
}

fn normalized_vector(mut vector: Vec<f64>) -> Vec<f64> {
    let norm = vector_norm(&vector);
    if norm == 0.0 {
        return vector;
    }
    for value in &mut vector {
        *value /= norm;
    }
    vector
}

fn vector_norm(vector: &[f64]) -> f64 {
    let mut sum = 0.0;
    for value in vector {
        sum += value * value;
    }
    sum.sqrt()
}

fn normalized_query_dot(query: &[f64], query_norm: f64, stored_normalized: &[f64]) -> f64 {
    if query_norm == 0.0 {
        return 0.0;
    }
    dot_product(query, stored_normalized) / query_norm
}

fn dot_product(left: &[f64], right: &[f64]) -> f64 {
    let len = left.len().min(right.len());
    let mut sum = 0.0;
    let mut index = 0;
    while index < len {
        sum += left[index] * right[index];
        index += 1;
    }
    sum
}

fn projection_planes(index_seed: &[u8], dimensions: usize) -> Vec<Vec<f64>> {
    (0..NeuralIndex::PLANES)
        .map(|plane| {
            (0..dimensions)
                .map(|dimension| projection(index_seed, plane, dimension))
                .collect()
        })
        .collect()
}

fn projection(index_seed: &[u8], plane: usize, dimension: usize) -> f64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in index_seed {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash ^= (plane as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    hash ^= (dimension as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    hash = splitmix64(hash);
    if hash & 1 == 0 {
        -1.0
    } else {
        1.0
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}
