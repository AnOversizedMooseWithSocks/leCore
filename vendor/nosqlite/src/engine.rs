use std::{cmp::Ordering, collections::BTreeMap};

use parking_lot::{Mutex, RwLock};
use serde::Serialize;
use serde_json::{json, Number, Value};

use crate::{
    encoder::{encode_text, encode_value, EncoderSpec},
    index::{json_vector, CollectionState, IndexKind, IndexSpec},
    kernel,
    mutation::{
        apply_cleanup_expired, apply_compiled_updates, apply_delete_many_compiled, CompiledUpdates,
    },
    neural::NeuralSpace,
    query::{compile_filter, compile_filter_excluding, CompiledFilter},
    storage::{Catalog, CollectionCatalog, Durability, Storage},
    Document, Error, Result,
};

const AUTO_INDEX_PROMOTION_THRESHOLD: usize = 3;
const MAX_AUTO_INDEXES_PER_COLLECTION: usize = 4;

pub use crate::storage::StorageMode;

#[derive(Debug, Clone)]
pub struct EngineOptions {
    pub storage: StorageMode,
    pub durability: Durability,
    pub shard_count: u64,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            storage: StorageMode::Memory,
            durability: Durability::Sync,
            shard_count: 64,
        }
    }
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(tag = "ok", rename_all = "camelCase")]
pub enum CommandResult {
    Created {
        collection: String,
    },
    Dropped {
        collection: String,
    },
    Snapshotted {
        collection: String,
        count: usize,
    },
    Compacted {
        collections: usize,
        #[serde(rename = "lastSeq")]
        last_seq: u64,
    },
    Inserted {
        count: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        ids: Option<Vec<Value>>,
    },
    Found {
        count: usize,
        documents: Vec<Document>,
        #[serde(rename = "lastEvaluatedKey")]
        #[serde(skip_serializing_if = "Option::is_none")]
        last_evaluated_key: Option<Value>,
    },
    Counted {
        count: usize,
    },
    Updated {
        matched: usize,
        modified: usize,
    },
    Deleted {
        count: usize,
    },
    Indexed {
        collection: String,
        indexes: Vec<String>,
    },
    EncoderCreated {
        encoder: String,
    },
    Encoded {
        encoder: String,
        dimensions: usize,
        vector: Vec<f64>,
    },
    NeuralSpaceCreated {
        space: String,
    },
    Learned {
        space: String,
        prototypes: usize,
    },
    Pong {
        protocol: u32,
    },
    Shutdown,
}

#[derive(Debug, Clone)]
struct ProjectionSpec {
    fields: Vec<String>,
}

#[derive(Debug)]
pub struct Engine {
    collections: RwLock<BTreeMap<String, CollectionState>>,
    encoders: RwLock<BTreeMap<String, EncoderSpec>>,
    neural_spaces: RwLock<BTreeMap<String, NeuralSpace>>,
    auto_index_hits: Mutex<BTreeMap<String, BTreeMap<Vec<String>, usize>>>,
    storage: Storage,
    shard_count: u64,
}

impl Engine {
    pub fn new(options: EngineOptions) -> Result<Self> {
        let storage = Storage::new(options.storage, options.durability)?;
        let catalog = storage.load_catalog()?;
        let collections = storage
            .load()?
            .into_iter()
            .map(|(name, documents)| {
                let mut state = CollectionState::new(documents)?;
                if let Some(collection_catalog) = catalog.collections.get(&name) {
                    state.create_indexes(collection_catalog.indexes.clone())?;
                }
                Ok((name, state))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;

        Ok(Self {
            collections: RwLock::new(collections),
            encoders: RwLock::new(catalog.encoders),
            neural_spaces: RwLock::new(catalog.neural_spaces),
            auto_index_hits: Mutex::new(BTreeMap::new()),
            storage,
            shard_count: options.shard_count.max(1),
        })
    }

    pub fn execute_json(&self, command: &str) -> Result<CommandResult> {
        let value = serde_json::from_str(command)?;
        self.execute(value)
    }

    pub fn execute(&self, command: Value) -> Result<CommandResult> {
        let mut command = match command {
            Value::Object(command) => command,
            _ => return Err(Error::ExpectedObject("command")),
        };

        if command.get("ping").is_some() {
            return Ok(CommandResult::Pong { protocol: 1 });
        }
        if let Some(encoder) = command.get("createEncoder") {
            return self.create_encoder(
                required_str(encoder, "createEncoder")?,
                parse_encoder_spec(&command)?,
            );
        }
        if let Some(encoder) = command.get("encodeText") {
            return self.encode_text_command(
                required_str(encoder, "encodeText")?,
                required_str(
                    command.get("text").ok_or(Error::MissingField("text"))?,
                    "text",
                )?,
            );
        }
        if let Some(space) = command.get("createNeuralSpace") {
            return self.create_neural_space(
                required_str(space, "createNeuralSpace")?,
                required_str(
                    command
                        .get("encoder")
                        .ok_or(Error::MissingField("encoder"))?,
                    "encoder",
                )?,
                command
                    .get("dimensions")
                    .and_then(Value::as_u64)
                    .map(|dimensions| dimensions as usize),
                command
                    .get("clusters")
                    .or_else(|| command.get("maxPrototypes"))
                    .and_then(Value::as_u64)
                    .map(|clusters| clusters as usize)
                    .unwrap_or(256),
            );
        }
        if let Some(space) = command.get("learnText") {
            return self.learn_text(
                required_str(space, "learnText")?,
                required_str(
                    command.get("text").ok_or(Error::MissingField("text"))?,
                    "text",
                )?,
                command
                    .get("label")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                command.get("weight").and_then(Value::as_f64).unwrap_or(1.0),
            );
        }
        if let Some(space) = command.get("learnVector") {
            return self.learn_vector(
                required_str(space, "learnVector")?,
                required_vector(command.get("vector"), "vector")?,
                command
                    .get("label")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                command.get("weight").and_then(Value::as_f64).unwrap_or(1.0),
            );
        }
        if let Some(collection) = command.get("create") {
            return self.create_collection(required_str(collection, "create")?);
        }
        if let Some(collection) = command.get("drop") {
            return self.drop_collection(required_str(collection, "drop")?);
        }
        if let Some(collection) = command.get("snapshot") {
            return self.snapshot_collection(required_str(collection, "snapshot")?);
        }
        if command.get("compact").is_some() {
            return self.compact();
        }
        if let Some(collection) = command.get("insert") {
            let collection = required_str(collection, "insert")?.to_string();
            let encode = optional_encode_spec(command.get("encode"))?;
            let return_ids = command
                .get("returnIds")
                .and_then(Value::as_bool)
                .unwrap_or(true);
            return self.insert_values_with_encode(
                &collection,
                required_array_value(
                    command
                        .remove("documents")
                        .ok_or(Error::MissingField("documents"))?,
                    "documents",
                )?,
                encode,
                return_ids,
            );
        }
        if let Some(collection) = command.get("createIndexes") {
            return self.create_indexes(
                required_str(collection, "createIndexes")?,
                required_array(command.get("indexes"), "indexes")?,
            );
        }
        if let Some(collection) = command.get("find") {
            let collection = required_str(collection, "find")?.to_string();
            let filter = optional_object_value(command.remove("filter"), "filter")?;
            let sort = optional_sort_value(command.remove("sort"))?;
            let projection = optional_projection_value(command.remove("projection"))?;
            return self.find_with_projection(
                &collection,
                filter,
                command
                    .get("limit")
                    .and_then(Value::as_u64)
                    .map(|limit| limit as usize),
                sort,
                command
                    .get("pageToken")
                    .or_else(|| command.get("exclusiveStartKey"))
                    .and_then(page_offset),
                projection,
            );
        }
        if let Some(collection) = command.get("count") {
            let collection = required_str(collection, "count")?.to_string();
            let filter = optional_object_value(command.remove("filter"), "filter")?;
            return self.count(&collection, filter);
        }
        if let Some(collection) = command.get("vectorSearch") {
            let collection = required_str(collection, "vectorSearch")?.to_string();
            let filter = optional_object_value(command.remove("filter"), "filter")?;
            return self.vector_search(
                &collection,
                command.get("index").and_then(Value::as_str),
                command.get("field").and_then(Value::as_str),
                required_vector(command.get("query"), "query")?,
                command.get("k").and_then(Value::as_u64).unwrap_or(10) as usize,
                filter,
            );
        }
        if let Some(collection) = command.get("semanticSearch") {
            return self.semantic_search(parse_semantic_search_spec(
                required_str(collection, "semanticSearch")?,
                &command,
            )?);
        }
        if let Some(collection) = command.get("update") {
            let collection = required_str(collection, "update")?.to_string();
            let filter = optional_object_value(command.remove("filter"), "filter")?;
            let updates = required_object_value(
                command
                    .remove("updates")
                    .ok_or(Error::MissingField("updates"))?,
                "updates",
            )?;
            return self.update(&collection, filter, updates);
        }
        if let Some(collection) = command.get("delete") {
            let collection = required_str(collection, "delete")?.to_string();
            let filter = optional_object_value(command.remove("filter"), "filter")?;
            return self.delete(&collection, filter);
        }
        if let Some(collection) = command.get("cleanupExpired") {
            return self.cleanup_expired(
                required_str(collection, "cleanupExpired")?,
                command
                    .get("ttlField")
                    .and_then(Value::as_str)
                    .unwrap_or("ttl"),
                command
                    .get("now")
                    .and_then(Value::as_i64)
                    .unwrap_or_else(now_epoch_seconds),
            );
        }

        Err(Error::UnsupportedCommand)
    }

    pub fn create_collection(&self, name: &str) -> Result<CommandResult> {
        let mut collections = self.collections.write();
        if collections.contains_key(name) {
            return Err(Error::CollectionExists(name.to_string()));
        }

        collections.insert(name.to_string(), CollectionState::empty());
        self.storage.save_collection(name, &[])?;
        self.storage.append_create_collection(name)?;

        Ok(CommandResult::Created {
            collection: name.to_string(),
        })
    }

    pub fn drop_collection(&self, name: &str) -> Result<CommandResult> {
        let mut collections = self.collections.write();
        if collections.remove(name).is_none() {
            return Err(Error::CollectionMissing(name.to_string()));
        }

        self.storage.append_drop_collection(name)?;
        self.storage.delete_collection(name)?;
        self.save_catalog(&collections)?;
        Ok(CommandResult::Dropped {
            collection: name.to_string(),
        })
    }

    pub fn snapshot_collection(&self, name: &str) -> Result<CommandResult> {
        let collections = self.collections.read();
        let collection = collections
            .get(name)
            .ok_or_else(|| Error::CollectionMissing(name.to_string()))?;
        self.storage.append_snapshot(name, &collection.documents)?;
        Ok(CommandResult::Snapshotted {
            collection: name.to_string(),
            count: collection.documents.len(),
        })
    }

    pub fn compact(&self) -> Result<CommandResult> {
        let collections = self.collections.read();
        let views = collections
            .iter()
            .map(|(name, collection)| (name.clone(), collection.documents.clone()))
            .collect::<BTreeMap<_, _>>();
        let last_seq = self.storage.compact_checkpoints(&views)?;
        Ok(CommandResult::Compacted {
            collections: views.len(),
            last_seq,
        })
    }

    pub fn create_encoder(&self, name: &str, spec: EncoderSpec) -> Result<CommandResult> {
        spec.validate()?;
        let collections = self.collections.read();
        let mut encoders = self.encoders.write();
        if encoders.contains_key(name) {
            return Err(Error::EncoderExists(name.to_string()));
        }
        encoders.insert(name.to_string(), spec);
        let neural_spaces = self.neural_spaces.read();
        self.save_catalog_parts(&collections, &encoders, &neural_spaces)?;
        Ok(CommandResult::EncoderCreated {
            encoder: name.to_string(),
        })
    }

    pub fn encode_text_command(&self, name: &str, text: &str) -> Result<CommandResult> {
        let encoders = self.encoders.read();
        let spec = encoders
            .get(name)
            .ok_or_else(|| Error::EncoderMissing(name.to_string()))?;
        let vector = encode_text(spec, text);
        Ok(CommandResult::Encoded {
            encoder: name.to_string(),
            dimensions: vector.len(),
            vector,
        })
    }

    pub fn create_neural_space(
        &self,
        name: &str,
        encoder: &str,
        dimensions: Option<usize>,
        max_prototypes: usize,
    ) -> Result<CommandResult> {
        let encoder_spec = self.encoder_spec(encoder)?;
        let dimensions = dimensions.unwrap_or(encoder_spec.dimensions);
        if dimensions != encoder_spec.dimensions {
            return Err(Error::VectorDimensionMismatch {
                field: "neuralSpace.dimensions".to_string(),
                expected: encoder_spec.dimensions,
                actual: dimensions,
            });
        }

        let collections = self.collections.read();
        let encoders = self.encoders.read();
        let mut neural_spaces = self.neural_spaces.write();
        if neural_spaces.contains_key(name) {
            return Err(Error::NeuralSpaceExists(name.to_string()));
        }
        neural_spaces.insert(
            name.to_string(),
            NeuralSpace::new(encoder, dimensions, max_prototypes),
        );
        self.save_catalog_parts(&collections, &encoders, &neural_spaces)?;
        Ok(CommandResult::NeuralSpaceCreated {
            space: name.to_string(),
        })
    }

    pub fn learn_text(
        &self,
        space: &str,
        text: &str,
        label: Option<String>,
        weight: f64,
    ) -> Result<CommandResult> {
        let encoder = {
            let spaces = self.neural_spaces.read();
            spaces
                .get(space)
                .ok_or_else(|| Error::NeuralSpaceMissing(space.to_string()))?
                .encoder
                .clone()
        };
        let spec = self.encoder_spec(&encoder)?;
        self.learn_vector(space, encode_text(&spec, text), label, weight)
    }

    pub fn learn_vector(
        &self,
        space: &str,
        vector: Vec<f64>,
        label: Option<String>,
        weight: f64,
    ) -> Result<CommandResult> {
        let collections = self.collections.read();
        let encoders = self.encoders.read();
        let mut neural_spaces = self.neural_spaces.write();
        let neural_space = neural_spaces
            .get_mut(space)
            .ok_or_else(|| Error::NeuralSpaceMissing(space.to_string()))?;
        neural_space.learn(&vector, label, weight)?;
        let prototypes = neural_space.prototypes.len();
        self.save_catalog_parts(&collections, &encoders, &neural_spaces)?;
        Ok(CommandResult::Learned {
            space: space.to_string(),
            prototypes,
        })
    }

    pub fn insert(&self, name: &str, documents: &[Value]) -> Result<CommandResult> {
        self.insert_with_encode(name, documents, None, true)
    }

    fn insert_with_encode(
        &self,
        name: &str,
        documents: &[Value],
        encode: Option<EncodeSpec>,
        return_ids: bool,
    ) -> Result<CommandResult> {
        self.insert_values_with_encode(name, documents.to_vec(), encode, return_ids)
    }

    fn insert_values_with_encode(
        &self,
        name: &str,
        documents: Vec<Value>,
        encode: Option<EncodeSpec>,
        return_ids: bool,
    ) -> Result<CommandResult> {
        let mut collections = self.collections.write();
        let collection = collections
            .entry(name.to_string())
            .or_insert_with(CollectionState::empty);
        let original_len = collection.documents.len();
        let document_count = documents.len();
        collection.documents.reserve(document_count);
        let mut ids = return_ids.then(|| Vec::with_capacity(document_count));
        let encoder = encode
            .as_ref()
            .map(|encode| self.encoder_spec(&encode.encoder))
            .transpose()?;

        for document in documents {
            let mut document = match document {
                Value::Object(document) => document,
                _ => return Err(Error::ExpectedObject("documents[]")),
            };
            if let (Some(encode), Some(encoder)) = (&encode, &encoder) {
                let source = document
                    .get(&encode.field)
                    .ok_or(Error::MissingField("encode.field source"))?;
                let vector = encode_value(encoder, source)?;
                document.insert(
                    encode.into.clone(),
                    Value::Array(vector.into_iter().map(Value::from).collect()),
                );
            }

            let id = if let Some(id) = document.get("_id") {
                let returned_id = ids.is_some().then(|| id.clone());
                let shard = self.shard_for(id);
                document.insert("_shard".to_string(), Value::Number(Number::from(shard)));
                returned_id
            } else {
                let generated = format!("{:016x}", kernel::next_id());
                let shard = self.shard_for_generated_string_id(&generated);
                let returned_id = if ids.is_some() {
                    let id = Value::String(generated);
                    document.insert("_id".to_string(), id.clone());
                    Some(id)
                } else {
                    document.insert("_id".to_string(), Value::String(generated));
                    None
                };
                document.insert("_shard".to_string(), Value::Number(Number::from(shard)));
                returned_id
            };
            if let (Some(ids), Some(id)) = (&mut ids, id) {
                ids.push(id);
            }
            let position = collection.documents.len();
            collection.documents.push(document);
            if let Err(error) = collection.index_document(position) {
                collection.documents.truncate(original_len);
                collection.rebuild_indexes()?;
                return Err(error);
            }
        }

        let inserted = &collection.documents[original_len..];
        self.storage.append_documents(name, inserted)?;
        Ok(CommandResult::Inserted {
            count: document_count,
            ids,
        })
    }

    pub fn create_indexes(&self, name: &str, specs: &[Value]) -> Result<CommandResult> {
        let mut collections = self.collections.write();
        let collection = collections
            .get_mut(name)
            .ok_or_else(|| Error::CollectionMissing(name.to_string()))?;
        let mut names = Vec::with_capacity(specs.len());
        let mut parsed_specs = Vec::with_capacity(specs.len());

        for spec in specs {
            let spec = parse_index_spec(spec)?;
            names.push(spec.name.clone());
            parsed_specs.push(spec);
        }
        collection.create_indexes(parsed_specs)?;

        self.save_catalog(&collections)?;
        Ok(CommandResult::Indexed {
            collection: name.to_string(),
            indexes: names,
        })
    }

    pub fn find(
        &self,
        name: &str,
        filter: Option<Document>,
        limit: Option<usize>,
        sort: Option<SortSpec>,
        page_offset: Option<usize>,
    ) -> Result<CommandResult> {
        self.find_with_projection(name, filter, limit, sort, page_offset, None)
    }

    fn find_with_projection(
        &self,
        name: &str,
        filter: Option<Document>,
        limit: Option<usize>,
        sort: Option<SortSpec>,
        page_offset: Option<usize>,
        projection: Option<ProjectionSpec>,
    ) -> Result<CommandResult> {
        let empty_filter = Document::new();
        let filter = filter.as_ref().unwrap_or(&empty_filter);
        let limit = limit.unwrap_or(usize::MAX);
        let projection = projection.as_ref();

        let (result, used_index) = {
            let collections = self.collections.read();
            let collection = collections
                .get(name)
                .ok_or_else(|| Error::CollectionMissing(name.to_string()))?;

            if let Some(sort) = &sort {
                let requested_offset = page_offset.unwrap_or(0);
                let requested_end = requested_offset.saturating_add(limit);
                if let Some((positions, has_more, already_sorted)) = collection
                    .ordered_page_candidates(filter, &sort.field, sort.ascending, requested_end)
                {
                    let mut sorted_positions = positions;
                    if !already_sorted {
                        sorted_positions.sort_by(|left, right| {
                            compare_positioned_documents(collection, *left, *right, sort)
                        });
                    }
                    let offset = requested_offset.min(sorted_positions.len());
                    let end = requested_end.min(sorted_positions.len());
                    let last_evaluated_key = (has_more || end < sorted_positions.len())
                        .then(|| json!({ "offset": end }));
                    let documents: Vec<Document> = sorted_positions[offset..end]
                        .iter()
                        .filter_map(|position| collection.documents.get(*position))
                        .map(|document| project_document(document, projection))
                        .collect();

                    (
                        CommandResult::Found {
                            count: documents.len(),
                            documents,
                            last_evaluated_key,
                        },
                        true,
                    )
                } else {
                    let candidates = collection.indexed_candidates_with_coverage(filter);
                    let filter_covered_by_index = candidates
                        .as_ref()
                        .is_some_and(|candidates| candidates.covered);
                    let compiled_filter =
                        (!filter_covered_by_index).then(|| compile_filter(filter));
                    let positions: Box<dyn Iterator<Item = usize>> =
                        if let Some(candidates) = candidates {
                            Box::new(candidates.positions.into_iter())
                        } else {
                            Box::new(0..collection.documents.len())
                        };
                    let mut sorted_positions = positions
                        .filter(|position| {
                            collection.documents.get(*position).is_some_and(|document| {
                                compiled_filter
                                    .as_ref()
                                    .is_none_or(|filter| filter.matches(document))
                            })
                        })
                        .collect::<Vec<_>>();
                    let offset = page_offset.unwrap_or(0).min(sorted_positions.len());
                    let end = (offset + limit).min(sorted_positions.len());
                    if end < sorted_positions.len() {
                        sorted_positions.select_nth_unstable_by(end, |left, right| {
                            compare_positioned_documents(collection, *left, *right, sort)
                        });
                    }
                    sorted_positions[..end].sort_by(|left, right| {
                        compare_positioned_documents(collection, *left, *right, sort)
                    });
                    let last_evaluated_key =
                        (end < sorted_positions.len()).then(|| json!({ "offset": end }));
                    let documents: Vec<Document> = sorted_positions[offset..end]
                        .iter()
                        .filter_map(|position| collection.documents.get(*position))
                        .map(|document| project_document(document, projection))
                        .collect();

                    (
                        CommandResult::Found {
                            count: documents.len(),
                            documents,
                            last_evaluated_key,
                        },
                        filter_covered_by_index,
                    )
                }
            } else {
                let offset = page_offset.unwrap_or(0);
                let end = offset.saturating_add(limit);
                if limit != usize::MAX {
                    if let Some((positions, has_more)) =
                        collection.indexed_page_candidates(filter, end)
                    {
                        let documents: Vec<Document> = positions[offset.min(positions.len())..]
                            .iter()
                            .filter_map(|position| collection.documents.get(*position))
                            .map(|document| project_document(document, projection))
                            .collect();
                        let last_evaluated_key = has_more.then(|| json!({ "offset": end }));
                        return Ok(CommandResult::Found {
                            count: documents.len(),
                            documents,
                            last_evaluated_key,
                        });
                    }
                }

                let candidates = collection.indexed_candidates_with_coverage(filter);
                let filter_covered_by_index = candidates
                    .as_ref()
                    .is_some_and(|candidates| candidates.covered);
                let compiled_filter = (!filter_covered_by_index).then(|| compile_filter(filter));
                let positions: Box<dyn Iterator<Item = usize>> =
                    if let Some(candidates) = candidates {
                        Box::new(candidates.positions.into_iter())
                    } else {
                        Box::new(0..collection.documents.len())
                    };
                let mut matched = 0;
                let mut documents = Vec::new();
                let mut last_evaluated_key = None;

                for position in positions {
                    let Some(document) = collection.documents.get(position) else {
                        continue;
                    };
                    if compiled_filter
                        .as_ref()
                        .is_some_and(|filter| !filter.matches(document))
                    {
                        continue;
                    }

                    let current = matched;
                    matched += 1;
                    if current >= offset && current < end {
                        documents.push(project_document(document, projection));
                    } else if current >= end {
                        last_evaluated_key = Some(json!({ "offset": end }));
                        break;
                    }
                }

                (
                    CommandResult::Found {
                        count: documents.len(),
                        documents,
                        last_evaluated_key,
                    },
                    filter_covered_by_index,
                )
            }
        };

        if !used_index {
            if let Some(fields) = auto_index_shape(filter) {
                self.maybe_promote_auto_index(name, fields)?;
            }
        }

        Ok(result)
    }

    pub fn count(&self, name: &str, filter: Option<Document>) -> Result<CommandResult> {
        let empty_filter = Document::new();
        let filter = filter.as_ref().unwrap_or(&empty_filter);

        let (count, used_index) = {
            let collections = self.collections.read();
            let collection = collections
                .get(name)
                .ok_or_else(|| Error::CollectionMissing(name.to_string()))?;

            if filter.is_empty() {
                (collection.documents.len(), true)
            } else if let Some(count) = collection.indexed_candidate_count(filter) {
                (count, true)
            } else if let Some(candidates) = collection.indexed_candidates_with_coverage(filter) {
                let compiled_filter = compile_filter(filter);
                (
                    candidates
                        .positions
                        .into_iter()
                        .filter(|position| {
                            collection
                                .documents
                                .get(*position)
                                .is_some_and(|document| compiled_filter.matches(document))
                        })
                        .count(),
                    false,
                )
            } else {
                let compiled_filter = compile_filter(filter);
                (
                    collection
                        .documents
                        .iter()
                        .filter(|document| compiled_filter.matches(document))
                        .count(),
                    false,
                )
            }
        };

        if !used_index {
            if let Some(fields) = auto_index_shape(filter) {
                self.maybe_promote_auto_index(name, fields)?;
            }
        }

        Ok(CommandResult::Counted { count })
    }

    pub fn vector_search(
        &self,
        name: &str,
        index_name: Option<&str>,
        field: Option<&str>,
        query: Vec<f64>,
        k: usize,
        filter: Option<Document>,
    ) -> Result<CommandResult> {
        let collections = self.collections.read();
        let collection = collections
            .get(name)
            .ok_or_else(|| Error::CollectionMissing(name.to_string()))?;
        let empty_filter = Document::new();
        let filter = filter.as_ref().unwrap_or(&empty_filter);
        let filter_is_empty = filter.is_empty();
        let compiled_filter = (!filter_is_empty).then(|| compile_filter(filter));
        let candidate_limit = filter_is_empty.then_some(k);

        let mut documents = Vec::with_capacity(k);
        for (position, score) in
            collection.neural_candidates(index_name, field, &query, candidate_limit)?
        {
            let Some(document) = collection.documents.get(position) else {
                continue;
            };
            if compiled_filter
                .as_ref()
                .is_some_and(|filter| !filter.matches(document))
            {
                continue;
            }
            let mut document = document.clone();
            document.insert("_score".to_string(), json!(score));
            documents.push(document);
            if documents.len() >= k {
                break;
            }
        }

        Ok(CommandResult::Found {
            count: documents.len(),
            documents,
            last_evaluated_key: None,
        })
    }

    fn semantic_search(&self, request: SemanticSearchSpec) -> Result<CommandResult> {
        let query = self.adapted_query_vector(&request)?;
        self.vector_search(
            &request.collection,
            request.index.as_deref(),
            request.field.as_deref(),
            query,
            request.k,
            request.filter,
        )
    }

    pub fn update(
        &self,
        name: &str,
        filter: Option<Document>,
        updates: Document,
    ) -> Result<CommandResult> {
        let mut collections = self.collections.write();
        let collection = collections
            .get_mut(name)
            .ok_or_else(|| Error::CollectionMissing(name.to_string()))?;
        let empty_filter = Document::new();
        let filter = filter.as_ref().unwrap_or(&empty_filter);
        let compiled_updates = CompiledUpdates::new(&updates);
        let affects_indexes = collection.updates_affect_indexes(&compiled_updates);
        if !affects_indexes {
            let mut matched = 0;
            let mut modified = 0;
            if let Some(candidates) = collection.indexed_candidates_with_residual_hint(filter) {
                let covered = candidates.covered;
                let compiled_filter = (!covered).then(|| {
                    compile_filter_excluding(filter, candidates.residual_filter_excluded_field)
                });
                for position in candidates.positions {
                    let Some(document) = collection.documents.get_mut(position) else {
                        continue;
                    };
                    if compiled_filter
                        .as_ref()
                        .is_some_and(|filter| !filter.matches(document))
                    {
                        continue;
                    }
                    matched += 1;
                    if apply_compiled_updates(document, &compiled_updates) {
                        modified += 1;
                    }
                }
            } else {
                let compiled_filter = compile_filter(filter);
                for document in &mut collection.documents {
                    if !compiled_filter.matches(document) {
                        continue;
                    }
                    matched += 1;
                    if apply_compiled_updates(document, &compiled_updates) {
                        modified += 1;
                    }
                }
            }
            self.storage.append_update(name, filter, &updates)?;
            return Ok(CommandResult::Updated { matched, modified });
        }
        let can_update_indexes_in_place =
            collection.updates_can_update_indexes_in_place(&compiled_updates);
        let mut update_state = UpdateApplyState {
            matched: 0,
            modified: 0,
            affects_indexes,
            can_update_indexes_in_place,
            in_place_changes: Vec::new(),
            rollback_changes: Vec::new(),
        };
        if let Some(candidates) = collection.indexed_candidates_with_residual_hint(filter) {
            let covered = candidates.covered;
            let candidate_count = candidates.positions.len();
            let compiled_filter = (!covered).then(|| {
                compile_filter_excluding(filter, candidates.residual_filter_excluded_field)
            });
            if update_state.affects_indexes {
                if update_state.can_update_indexes_in_place {
                    update_state.in_place_changes.reserve(candidate_count);
                } else {
                    update_state.rollback_changes.reserve(candidate_count);
                }
            }
            for position in candidates.positions {
                apply_update_at_position(
                    collection,
                    compiled_filter.as_ref(),
                    &compiled_updates,
                    position,
                    &mut update_state,
                );
            }
        } else {
            let compiled_filter = compile_filter(filter);
            for position in 0..collection.documents.len() {
                apply_update_at_position(
                    collection,
                    Some(&compiled_filter),
                    &compiled_updates,
                    position,
                    &mut update_state,
                );
            }
        }

        if update_state.affects_indexes {
            if update_state.can_update_indexes_in_place {
                if let Err(error) = collection.update_documents_in_place(
                    update_state
                        .in_place_changes
                        .iter()
                        .map(|(position, old_document)| (*position, old_document)),
                ) {
                    for (position, old_document) in update_state.in_place_changes {
                        if let Some(document) = collection.documents.get_mut(position) {
                            *document = old_document;
                        }
                    }
                    collection.rebuild_indexes()?;
                    return Err(error);
                }
            } else if let Err(error) =
                collection.update_documents(update_state.rollback_changes.iter().map(
                    |(position, old_document, new_document)| {
                        (*position, old_document, new_document)
                    },
                ))
            {
                for (position, old_document, _) in update_state.rollback_changes {
                    if let Some(document) = collection.documents.get_mut(position) {
                        *document = old_document;
                    }
                }
                return Err(error);
            }
        }
        self.storage.append_update(name, filter, &updates)?;
        Ok(CommandResult::Updated {
            matched: update_state.matched,
            modified: update_state.modified,
        })
    }

    pub fn delete(&self, name: &str, filter: Option<Document>) -> Result<CommandResult> {
        let mut collections = self.collections.write();
        let collection = collections
            .get_mut(name)
            .ok_or_else(|| Error::CollectionMissing(name.to_string()))?;
        let empty_filter = Document::new();
        let filter = filter.as_ref().unwrap_or(&empty_filter);
        let before = collection.documents.len();

        let used_index = if filter.is_empty() {
            collection.documents.clear();
            false
        } else {
            let compiled_filter = compile_filter(filter);
            if let Some(candidates) = collection.indexed_candidates_with_coverage(filter) {
                let covered = candidates.covered;
                let mut deleted_positions = Vec::with_capacity(candidates.positions.len());
                for position in candidates.positions {
                    let Some(document) = collection.documents.get(position) else {
                        continue;
                    };
                    if covered || compiled_filter.matches(document) {
                        deleted_positions.push(position);
                    }
                }
                deleted_positions.sort_unstable();
                deleted_positions.dedup();
                collection.delete_positions(&deleted_positions);
                true
            } else {
                apply_delete_many_compiled(&mut collection.documents, &compiled_filter);
                false
            }
        };
        let deleted = before - collection.documents.len();

        if !used_index {
            collection.rebuild_indexes()?;
        }
        self.storage.append_delete(name, filter)?;
        Ok(CommandResult::Deleted { count: deleted })
    }

    pub fn cleanup_expired(&self, name: &str, ttl_field: &str, now: i64) -> Result<CommandResult> {
        let mut collections = self.collections.write();
        let collection = collections
            .get_mut(name)
            .ok_or_else(|| Error::CollectionMissing(name.to_string()))?;
        let before = collection.documents.len();

        apply_cleanup_expired(&mut collection.documents, ttl_field, now);
        let deleted = before - collection.documents.len();

        collection.rebuild_indexes()?;
        self.storage.append_cleanup_expired(name, ttl_field, now)?;
        Ok(CommandResult::Deleted { count: deleted })
    }

    #[doc(hidden)]
    pub fn internal_auto_index_count(&self, name: &str) -> usize {
        self.collections
            .read()
            .get(name)
            .map(CollectionState::internal_exact_index_count)
            .unwrap_or(0)
    }

    fn maybe_promote_auto_index(&self, name: &str, fields: Vec<String>) -> Result<()> {
        let should_promote = {
            let mut hits = self.auto_index_hits.lock();
            let count = hits
                .entry(name.to_string())
                .or_default()
                .entry(fields.clone())
                .or_default();
            *count += 1;
            *count >= AUTO_INDEX_PROMOTION_THRESHOLD
        };
        if !should_promote {
            return Ok(());
        }

        let mut collections = self.collections.write();
        let collection = collections
            .get_mut(name)
            .ok_or_else(|| Error::CollectionMissing(name.to_string()))?;
        if collection.has_exact_index_fields(&fields)
            || collection.internal_exact_index_count() >= MAX_AUTO_INDEXES_PER_COLLECTION
        {
            return Ok(());
        }
        collection.create_internal_exact_index(fields)
    }

    fn shard_for(&self, id: &Value) -> u64 {
        if let Some(id) = id.as_str() {
            return self.shard_for_generated_string_id(id);
        }
        let raw = id.to_string();
        kernel::hash_bytes(raw.as_bytes()) % self.shard_count
    }

    fn shard_for_generated_string_id(&self, id: &str) -> u64 {
        hash_bytes_parts([b"\"", id.as_bytes(), b"\""]) % self.shard_count
    }

    fn save_catalog(&self, collections: &BTreeMap<String, CollectionState>) -> Result<()> {
        let encoders = self.encoders.read();
        let neural_spaces = self.neural_spaces.read();
        self.save_catalog_parts(collections, &encoders, &neural_spaces)
    }

    fn save_catalog_parts(
        &self,
        collections: &BTreeMap<String, CollectionState>,
        encoders: &BTreeMap<String, EncoderSpec>,
        neural_spaces: &BTreeMap<String, NeuralSpace>,
    ) -> Result<()> {
        let catalog = Catalog {
            collections: collections
                .iter()
                .map(|(name, collection)| {
                    (
                        name.clone(),
                        CollectionCatalog {
                            indexes: collection.index_specs(),
                        },
                    )
                })
                .collect(),
            encoders: encoders.clone(),
            neural_spaces: neural_spaces.clone(),
        };
        self.storage.save_catalog(&catalog)
    }

    fn encoder_spec(&self, name: &str) -> Result<EncoderSpec> {
        self.encoders
            .read()
            .get(name)
            .cloned()
            .ok_or_else(|| Error::EncoderMissing(name.to_string()))
    }

    fn adapted_query_vector(&self, request: &SemanticSearchSpec) -> Result<Vec<f64>> {
        if let Some(space_name) = &request.space {
            let (encoder, vector) = {
                let spaces = self.neural_spaces.read();
                let space = spaces
                    .get(space_name)
                    .ok_or_else(|| Error::NeuralSpaceMissing(space_name.clone()))?;
                let encoder = space.encoder.clone();
                let spec = self.encoder_spec(&encoder)?;
                let vector = space
                    .adapt_query(&encode_text(&spec, &request.text), request.label.as_deref())?;
                (encoder, vector)
            };
            if request
                .encoder
                .as_ref()
                .is_some_and(|requested| requested != &encoder)
            {
                return Err(Error::UnsupportedEncoder(format!(
                    "semanticSearch encoder does not match neural space `{space_name}`"
                )));
            }
            return Ok(vector);
        }

        let encoder = request
            .encoder
            .as_ref()
            .ok_or(Error::MissingField("encoder"))?;
        let spec = self.encoder_spec(encoder)?;
        Ok(encode_text(&spec, &request.text))
    }
}

#[derive(Debug, Clone)]
struct EncodeSpec {
    encoder: String,
    field: String,
    into: String,
}

#[derive(Debug, Clone)]
struct SemanticSearchSpec {
    collection: String,
    encoder: Option<String>,
    space: Option<String>,
    label: Option<String>,
    index: Option<String>,
    field: Option<String>,
    text: String,
    k: usize,
    filter: Option<Document>,
}

struct UpdateApplyState {
    matched: usize,
    modified: usize,
    affects_indexes: bool,
    can_update_indexes_in_place: bool,
    in_place_changes: Vec<(usize, Document)>,
    rollback_changes: Vec<(usize, Document, Document)>,
}

fn apply_update_at_position(
    collection: &mut CollectionState,
    compiled_filter: Option<&CompiledFilter<'_>>,
    updates: &CompiledUpdates<'_>,
    position: usize,
    state: &mut UpdateApplyState,
) {
    if position >= collection.documents.len() {
        return;
    }
    if compiled_filter.is_some_and(|filter| !filter.matches(&collection.documents[position])) {
        return;
    }

    state.matched += 1;
    if state.affects_indexes && !updates.would_change(&collection.documents[position]) {
        return;
    }

    let old_document = state
        .affects_indexes
        .then(|| collection.documents[position].clone());
    if !apply_compiled_updates(&mut collection.documents[position], updates) {
        return;
    }

    if state.affects_indexes {
        let old_document = old_document.expect("old document captured for indexed update");
        if state.can_update_indexes_in_place {
            state.in_place_changes.push((position, old_document));
        } else {
            let new_document = collection.documents[position].clone();
            state
                .rollback_changes
                .push((position, old_document, new_document));
        }
    }
    state.modified += 1;
}

#[derive(Debug, Clone)]
pub struct SortSpec {
    pub field: String,
    pub ascending: bool,
}

fn required_str<'a>(value: &'a Value, field: &'static str) -> Result<&'a str> {
    value.as_str().ok_or(Error::ExpectedObject(field))
}

fn required_array<'a>(value: Option<&'a Value>, field: &'static str) -> Result<&'a [Value]> {
    value
        .ok_or(Error::MissingField(field))?
        .as_array()
        .map(Vec::as_slice)
        .ok_or(Error::ExpectedArray(field))
}

fn required_array_value(value: Value, field: &'static str) -> Result<Vec<Value>> {
    match value {
        Value::Array(values) => Ok(values),
        _ => Err(Error::ExpectedArray(field)),
    }
}

fn required_object_value(value: Value, field: &'static str) -> Result<Document> {
    match value {
        Value::Object(object) => Ok(object),
        _ => Err(Error::ExpectedObject(field)),
    }
}

fn required_vector(value: Option<&Value>, field: &'static str) -> Result<Vec<f64>> {
    json_vector(value.ok_or(Error::MissingField(field))?).ok_or(Error::ExpectedArray(field))
}

fn optional_object(value: Option<&Value>, field: &'static str) -> Result<Option<Document>> {
    value
        .map(|value| {
            value
                .as_object()
                .cloned()
                .ok_or(Error::ExpectedObject(field))
        })
        .transpose()
}

fn optional_object_value(value: Option<Value>, field: &'static str) -> Result<Option<Document>> {
    value
        .map(|value| required_object_value(value, field))
        .transpose()
}

fn optional_sort(value: Option<&Value>) -> Result<Option<SortSpec>> {
    let Some(value) = value else {
        return Ok(None);
    };

    if let Some(object) = value.as_object() {
        let (field, direction) = object.iter().next().ok_or(Error::MissingField("sort"))?;
        return Ok(Some(SortSpec {
            field: field.clone(),
            ascending: direction.as_i64().unwrap_or(1) >= 0,
        }));
    }

    Err(Error::ExpectedObject("sort"))
}

fn optional_sort_value(value: Option<Value>) -> Result<Option<SortSpec>> {
    value
        .as_ref()
        .map_or(Ok(None), |value| optional_sort(Some(value)))
}

fn optional_projection(value: Option<&Value>) -> Result<Option<ProjectionSpec>> {
    let Some(value) = value else {
        return Ok(None);
    };

    let object = value
        .as_object()
        .ok_or(Error::ExpectedObject("projection"))?;
    let fields = object
        .iter()
        .filter(|(_, include)| projection_includes_field(include))
        .map(|(field, _)| field.clone())
        .collect();
    Ok(Some(ProjectionSpec { fields }))
}

fn optional_projection_value(value: Option<Value>) -> Result<Option<ProjectionSpec>> {
    value
        .as_ref()
        .map_or(Ok(None), |value| optional_projection(Some(value)))
}

fn projection_includes_field(value: &Value) -> bool {
    match value {
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_i64().unwrap_or(0) != 0,
        _ => false,
    }
}

fn project_document(document: &Document, projection: Option<&ProjectionSpec>) -> Document {
    let Some(projection) = projection else {
        return document.clone();
    };

    let mut projected = Document::with_capacity(projection.fields.len());
    for field in &projection.fields {
        if let Some(value) = document.get(field) {
            projected.insert(field.clone(), value.clone());
        }
    }
    projected
}

fn optional_encode_spec(value: Option<&Value>) -> Result<Option<EncodeSpec>> {
    value.map(parse_encode_spec).transpose()
}

fn parse_encode_spec(value: &Value) -> Result<EncodeSpec> {
    let spec = value.as_object().ok_or(Error::ExpectedObject("encode"))?;
    Ok(EncodeSpec {
        encoder: spec
            .get("encoder")
            .and_then(Value::as_str)
            .ok_or(Error::MissingField("encode.encoder"))?
            .to_string(),
        field: spec
            .get("field")
            .and_then(Value::as_str)
            .ok_or(Error::MissingField("encode.field"))?
            .to_string(),
        into: spec
            .get("into")
            .and_then(Value::as_str)
            .ok_or(Error::MissingField("encode.into"))?
            .to_string(),
    })
}

fn auto_index_shape(filter: &Document) -> Option<Vec<String>> {
    if filter.is_empty() {
        return None;
    }

    if indexable_equality_value(filter, "pk").is_some()
        && (indexable_equality_value(filter, "sk").is_some()
            || indexable_prefix_value(filter, "sk").is_some())
    {
        return Some(vec!["pk".to_string(), "sk".to_string()]);
    }

    let mut fields = filter
        .iter()
        .filter_map(|(field, _)| indexable_equality_value(filter, field).map(|_| field.to_string()))
        .collect::<Vec<_>>();
    if fields.is_empty() {
        return None;
    }
    fields.sort();
    Some(fields)
}

fn indexable_equality_value<'a>(filter: &'a Document, field: &str) -> Option<&'a Value> {
    let value = match filter.get(field)? {
        Value::Object(operator) if operator.len() == 1 => operator.get("$eq")?,
        Value::Object(_) => return None,
        value => value,
    };
    is_indexable_scalar(value).then_some(value)
}

fn indexable_prefix_value<'a>(filter: &'a Document, field: &str) -> Option<&'a str> {
    let prefix = filter.get(field)?.as_object()?.get("$prefix")?.as_str()?;
    (prefix.len() <= 128).then_some(prefix)
}

fn is_indexable_scalar(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => true,
        Value::String(value) => value.len() <= 128,
        Value::Array(_) | Value::Object(_) => false,
    }
}

fn parse_semantic_search_spec(
    collection: &str,
    command: &serde_json::Map<String, Value>,
) -> Result<SemanticSearchSpec> {
    let encoder = command
        .get("encoder")
        .and_then(Value::as_str)
        .map(str::to_string);
    let space = command
        .get("space")
        .and_then(Value::as_str)
        .map(str::to_string);
    if encoder.is_none() && space.is_none() {
        return Err(Error::MissingField("encoder"));
    }

    Ok(SemanticSearchSpec {
        collection: collection.to_string(),
        encoder,
        space,
        label: command
            .get("label")
            .and_then(Value::as_str)
            .map(str::to_string),
        index: command
            .get("index")
            .and_then(Value::as_str)
            .map(str::to_string),
        field: command
            .get("field")
            .and_then(Value::as_str)
            .map(str::to_string),
        text: required_str(
            command.get("text").ok_or(Error::MissingField("text"))?,
            "text",
        )?
        .to_string(),
        k: command.get("k").and_then(Value::as_u64).unwrap_or(10) as usize,
        filter: optional_object(command.get("filter"), "filter")?,
    })
}

fn parse_encoder_spec(command: &serde_json::Map<String, Value>) -> Result<EncoderSpec> {
    let spec = EncoderSpec {
        kind: command
            .get("kind")
            .and_then(Value::as_str)
            .unwrap_or("text")
            .to_string(),
        provider: command
            .get("provider")
            .and_then(Value::as_str)
            .unwrap_or(crate::encoder::DEFAULT_PROVIDER)
            .to_string(),
        dimensions: command
            .get("dimensions")
            .and_then(Value::as_u64)
            .map(|dimensions| dimensions as usize)
            .unwrap_or(384),
        seed: command.get("seed").and_then(Value::as_u64).unwrap_or(0),
    };
    spec.validate()?;
    Ok(spec)
}

fn page_offset(value: &Value) -> Option<usize> {
    value.as_u64().map(|offset| offset as usize).or_else(|| {
        value
            .as_object()?
            .get("offset")?
            .as_u64()
            .map(|offset| offset as usize)
    })
}

fn compare_documents(left: &Document, right: &Document, sort: &SortSpec) -> Ordering {
    let ordering = compare_values(left.get(&sort.field), right.get(&sort.field))
        .then_with(|| compare_values(left.get("_id"), right.get("_id")));
    if sort.ascending {
        ordering
    } else {
        ordering.reverse()
    }
}

fn compare_positioned_documents(
    collection: &CollectionState,
    left: usize,
    right: usize,
    sort: &SortSpec,
) -> Ordering {
    let left = collection
        .documents
        .get(left)
        .expect("sorted position came from documents");
    let right = collection
        .documents
        .get(right)
        .expect("sorted position came from documents");
    compare_documents(left, right, sort)
}

fn compare_values(left: Option<&Value>, right: Option<&Value>) -> Ordering {
    match (left, right) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(Value::Number(left)), Some(Value::Number(right))) => left
            .as_f64()
            .partial_cmp(&right.as_f64())
            .unwrap_or(Ordering::Equal),
        (Some(Value::String(left)), Some(Value::String(right))) => left.cmp(right),
        (Some(Value::Bool(left)), Some(Value::Bool(right))) => left.cmp(right),
        (Some(left), Some(right)) => left.to_string().cmp(&right.to_string()),
    }
}

fn now_epoch_seconds() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0)
}

fn hash_bytes_parts<const N: usize>(parts: [&[u8]; N]) -> u64 {
    let mut hash = 1469598103934665603_u64;
    for part in parts {
        for byte in part {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(1099511628211_u64);
        }
    }
    hash
}

fn parse_index_spec(value: &Value) -> Result<IndexSpec> {
    let spec = value
        .as_object()
        .ok_or(Error::ExpectedObject("indexes[]"))?;

    if let Some(key) = spec.get("key") {
        let key = key.as_object().ok_or(Error::ExpectedObject("key"))?;
        let fields = key.keys().cloned().collect::<Vec<_>>();
        let field = fields.first().ok_or(Error::MissingField("key"))?;
        let name = spec
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| {
                fields
                    .iter()
                    .map(|field| format!("{field}_1"))
                    .collect::<Vec<_>>()
                    .join("_")
            });
        return Ok(IndexSpec {
            name,
            field: field.clone(),
            fields: (fields.len() > 1).then_some(fields),
            kind: IndexKind::Exact,
            unique: spec.get("unique").and_then(Value::as_bool).unwrap_or(false),
            dimensions: None,
        });
    }

    if let Some(field) = spec.get("neural").and_then(Value::as_str) {
        let dimensions = spec
            .get("dimensions")
            .and_then(Value::as_u64)
            .map(|dimensions| dimensions as usize)
            .ok_or(Error::MissingField("dimensions"))?;
        let name = spec
            .get("name")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| format!("{field}_neural"));
        return Ok(IndexSpec {
            name,
            field: field.to_string(),
            fields: None,
            kind: IndexKind::Neural,
            unique: false,
            dimensions: Some(dimensions),
        });
    }

    Err(Error::MissingField("key"))
}
