use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, ErrorKind, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use fs2::FileExt;
use parking_lot::Mutex;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::{
    encoder::EncoderSpec,
    index::IndexSpec,
    kernel,
    mutation::{apply_cleanup_expired, apply_delete_many, apply_update_many},
    neural::NeuralSpace,
    Document, Error, Result,
};

const CATALOG_FILE: &str = "__nosqlite_catalog.json";
const EVENTS_FILE: &str = "__nosqlite_events.jsonl";
const MANIFEST_FILE: &str = "__nosqlite_manifest.json";
const LOCK_FILE: &str = ".nosqlite.lock";

type CollectionViews = BTreeMap<String, Vec<Document>>;

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Catalog {
    #[serde(default)]
    pub collections: BTreeMap<String, CollectionCatalog>,
    #[serde(default)]
    pub encoders: BTreeMap<String, EncoderSpec>,
    #[serde(default)]
    pub neural_spaces: BTreeMap<String, NeuralSpace>,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CollectionCatalog {
    #[serde(default)]
    pub indexes: Vec<IndexSpec>,
}

#[derive(Debug, Clone)]
pub enum StorageMode {
    Memory,
    FileSystem(PathBuf),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Durability {
    Sync,
    Buffered,
}

#[derive(Debug)]
pub struct Storage {
    mode: StorageMode,
    _lock: Option<File>,
    next_seq: Mutex<Option<u64>>,
    event_file: Mutex<Option<File>>,
    manifest: Mutex<Option<StorageManifest>>,
    durability: Durability,
}

impl Storage {
    pub fn new(mode: StorageMode, durability: Durability) -> Result<Self> {
        let mut lock = None;
        if let StorageMode::FileSystem(path) = &mode {
            fs::create_dir_all(path)?;
            let lock_file = OpenOptions::new()
                .create(true)
                .truncate(false)
                .read(true)
                .write(true)
                .open(path.join(LOCK_FILE))?;
            lock_file.try_lock_exclusive()?;
            lock = Some(lock_file);
        }

        Ok(Self {
            mode,
            _lock: lock,
            next_seq: Mutex::new(None),
            event_file: Mutex::new(None),
            manifest: Mutex::new(None),
            durability,
        })
    }

    pub fn load(&self) -> Result<BTreeMap<String, Vec<Document>>> {
        let StorageMode::FileSystem(path) = &self.mode else {
            return Ok(BTreeMap::new());
        };

        let (mut collections, checkpoint_seq) = load_collection_views(path)?;

        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let file_path = entry.path();
            if file_path.extension().and_then(|ext| ext.to_str()) != Some("wal") {
                continue;
            }
            let Some(name) = file_path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            let documents = collections.entry(name.to_string()).or_default();
            apply_wal(&file_path, documents)?;
        }
        let manifest = load_manifest(path)?;
        let last_event_seq =
            apply_sync_events(path, &mut collections, checkpoint_seq, manifest.as_ref())?;
        *self.manifest.lock() = manifest;
        *self.next_seq.lock() = Some(last_event_seq + 1);

        Ok(collections)
    }

    pub fn load_catalog(&self) -> Result<Catalog> {
        let StorageMode::FileSystem(path) = &self.mode else {
            return Ok(Catalog::default());
        };

        let catalog_path = path.join(CATALOG_FILE);
        match fs::read_to_string(catalog_path) {
            Ok(raw) if raw.trim().is_empty() => Ok(Catalog::default()),
            Ok(raw) => Ok(serde_json::from_str(&raw)?),
            Err(error) if error.kind() == ErrorKind::NotFound => Ok(Catalog::default()),
            Err(error) => Err(Error::Storage(error)),
        }
    }

    pub fn save_catalog(&self, catalog: &Catalog) -> Result<()> {
        let StorageMode::FileSystem(path) = &self.mode else {
            return Ok(());
        };

        fs::create_dir_all(path)?;
        let file_path = path.join(CATALOG_FILE);
        let tmp_path = file_path.with_extension("json.tmp");
        let body = serde_json::to_string_pretty(catalog)?;

        fs::write(&tmp_path, body)?;
        fs::rename(tmp_path, file_path)?;
        Ok(())
    }

    pub fn save_collection(&self, name: &str, documents: &[Document]) -> Result<()> {
        let StorageMode::FileSystem(path) = &self.mode else {
            return Ok(());
        };

        fs::create_dir_all(path)?;
        let file_path = collection_path(path, name);
        let tmp_path = file_path.with_extension("json.tmp");
        let body = serde_json::to_string_pretty(documents)?;

        fs::write(&tmp_path, body)?;
        fs::rename(tmp_path, file_path)?;
        truncate_wal(path, name)?;
        Ok(())
    }

    pub fn save_checkpoint(&self, name: &str, documents: &[Document], last_seq: u64) -> Result<()> {
        let StorageMode::FileSystem(path) = &self.mode else {
            return Ok(());
        };

        fs::create_dir_all(path)?;
        write_json_zstd_atomic(&collection_compressed_path(path, name), documents)?;
        remove_if_exists(&collection_path(path, name))?;
        write_json_atomic(
            &collection_meta_path(path, name),
            &ProjectionMeta {
                last_seq,
                document_count: documents.len(),
                compression: Some("zstd".to_string()),
            },
        )?;
        truncate_wal(path, name)?;
        Ok(())
    }

    pub fn compact_checkpoints(
        &self,
        collections: &BTreeMap<String, Vec<Document>>,
    ) -> Result<u64> {
        let StorageMode::FileSystem(path) = &self.mode else {
            return Ok(0);
        };
        let last_seq = self.current_event_seq(path)?;
        for (name, documents) in collections {
            self.save_checkpoint(name, documents, last_seq)?;
        }
        self.rotate_event_segment_after_compact(path, collections.keys(), last_seq)?;
        Ok(last_seq)
    }

    pub fn append_create_collection(&self, name: &str) -> Result<()> {
        self.append_sync_event_ref(name, WalRecordRef::Create)
    }

    pub fn append_snapshot(&self, name: &str, documents: &[Document]) -> Result<()> {
        self.append_sync_event_ref(name, WalRecordRef::Snapshot { documents })
    }

    pub fn append_documents(&self, name: &str, documents: &[Document]) -> Result<()> {
        if documents.is_empty() {
            return Ok(());
        }

        self.append_sync_event_ref(name, WalRecordRef::Insert { documents })
    }

    pub fn append_update(&self, name: &str, filter: &Document, updates: &Document) -> Result<()> {
        self.append_sync_event_ref(name, WalRecordRef::Update { filter, updates })
    }

    pub fn append_delete(&self, name: &str, filter: &Document) -> Result<()> {
        self.append_sync_event_ref(name, WalRecordRef::Delete { filter })
    }

    pub fn append_cleanup_expired(&self, name: &str, ttl_field: &str, now: i64) -> Result<()> {
        self.append_sync_event_ref(name, WalRecordRef::CleanupExpired { ttl_field, now })
    }

    pub fn append_drop_collection(&self, name: &str) -> Result<()> {
        self.append_sync_event_ref(name, WalRecordRef::Drop)
    }

    fn append_sync_event_ref(&self, collection: &str, op: WalRecordRef<'_>) -> Result<()> {
        let StorageMode::FileSystem(path) = &self.mode else {
            return Ok(());
        };

        let mut next_seq = self.next_seq.lock();
        let seq = match *next_seq {
            Some(seq) => seq,
            None => last_sync_event_seq(path)? + 1,
        };
        let mut manifest = self.manifest.lock();
        if manifest.is_none() {
            *manifest = Some(load_or_default_manifest(path)?);
        }
        let event_path = event_segment_path(
            path,
            manifest
                .as_ref()
                .expect("manifest initialized")
                .active_segment,
        );
        let ts = now_epoch_seconds();
        let event = SyncEventRef {
            seq: Some(seq),
            ts: Some(ts),
            collection,
            op,
            checksum: None,
        };
        let event_body = serde_json::to_string(&event)?;
        let mut event_file = self.event_file.lock();
        if event_file.is_none() {
            *event_file = Some(
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(event_path)?,
            );
        }
        let file = event_file.as_mut().expect("event file initialized");
        write_sync_event_line(file, &event_body)?;
        if self.durability == Durability::Sync {
            file.sync_data()?;
        }
        *next_seq = Some(seq + 1);
        Ok(())
    }

    pub fn delete_collection(&self, name: &str) -> Result<()> {
        let StorageMode::FileSystem(path) = &self.mode else {
            return Ok(());
        };

        remove_if_exists(&collection_path(path, name))?;
        remove_if_exists(&collection_compressed_path(path, name))?;
        remove_if_exists(&collection_meta_path(path, name))?;
        remove_if_exists(&wal_path(path, name))
    }

    fn current_event_seq(&self, path: &Path) -> Result<u64> {
        if let Some(next_seq) = *self.next_seq.lock() {
            return Ok(next_seq.saturating_sub(1));
        }
        last_sync_event_seq(path)
    }

    fn rotate_event_segment_after_compact<'a>(
        &self,
        root: &Path,
        collections: impl Iterator<Item = &'a String>,
        last_seq: u64,
    ) -> Result<()> {
        let mut guard = self.manifest.lock();
        let mut manifest = match guard.take() {
            Some(manifest) => manifest,
            None => load_manifest(root)?.unwrap_or_else(default_manifest),
        };
        for name in collections {
            manifest.checkpoints.insert(name.clone(), last_seq);
        }

        if let Some(max_seq) = event_file_max_seq(&events_path(root))? {
            manifest.legacy_max_seq = Some(max_seq);
        }

        if last_seq > 0 {
            let active_segment = manifest.active_segment.max(1);
            if let Some((min_seq, max_seq)) =
                event_file_bounds(&event_segment_path(root, active_segment))?
            {
                upsert_segment_meta(
                    &mut manifest,
                    EventSegmentMeta {
                        id: active_segment,
                        min_seq,
                        max_seq,
                        prunable: false,
                    },
                );
            }
            manifest.active_segment = active_segment + 1;
            *self.event_file.lock() = None;
        }

        mark_prunable_segments(&mut manifest);
        save_manifest(root, &manifest)?;
        *guard = Some(manifest);
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "camelCase")]
enum WalRecord {
    Create,
    Drop,
    Snapshot { documents: Vec<Document> },
    Insert { documents: Vec<Document> },
    Update { filter: Document, updates: Document },
    Delete { filter: Document },
    CleanupExpired { ttl_field: String, now: i64 },
}

#[derive(Debug, Serialize)]
#[serde(tag = "op", rename_all = "camelCase")]
enum WalRecordRef<'a> {
    Create,
    Drop,
    Snapshot {
        documents: &'a [Document],
    },
    Insert {
        documents: &'a [Document],
    },
    Update {
        filter: &'a Document,
        updates: &'a Document,
    },
    Delete {
        filter: &'a Document,
    },
    CleanupExpired {
        ttl_field: &'a str,
        now: i64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SyncEvent {
    #[serde(skip_serializing_if = "Option::is_none")]
    seq: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ts: Option<i64>,
    collection: String,
    #[serde(flatten)]
    op: WalRecord,
    #[serde(skip_serializing_if = "Option::is_none")]
    checksum: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SyncEventEnvelope {
    seq: Option<u64>,
    ts: Option<i64>,
    checksum: Option<u64>,
}

#[derive(Debug, Serialize)]
struct SyncEventRef<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    seq: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ts: Option<i64>,
    collection: &'a str,
    #[serde(flatten)]
    op: WalRecordRef<'a>,
    #[serde(skip_serializing_if = "Option::is_none")]
    checksum: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ProjectionMeta {
    last_seq: u64,
    document_count: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    compression: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StorageManifest {
    #[serde(default = "default_active_segment")]
    active_segment: u64,
    #[serde(default)]
    segments: Vec<EventSegmentMeta>,
    #[serde(default)]
    checkpoints: BTreeMap<String, u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    legacy_max_seq: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct EventSegmentMeta {
    id: u64,
    min_seq: u64,
    max_seq: u64,
    #[serde(default)]
    prunable: bool,
}

impl StorageManifest {
    fn segment(&self, id: u64) -> Option<&EventSegmentMeta> {
        self.segments.iter().find(|segment| segment.id == id)
    }
}

fn default_active_segment() -> u64 {
    1
}

fn default_manifest() -> StorageManifest {
    StorageManifest {
        active_segment: default_active_segment(),
        segments: Vec::new(),
        checkpoints: BTreeMap::new(),
        legacy_max_seq: None,
    }
}

fn load_manifest(root: &Path) -> Result<Option<StorageManifest>> {
    match fs::read_to_string(manifest_path(root)) {
        Ok(raw) if raw.trim().is_empty() => Ok(None),
        Ok(raw) => Ok(Some(serde_json::from_str(&raw)?)),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(Error::Storage(error)),
    }
}

fn load_or_default_manifest(root: &Path) -> Result<StorageManifest> {
    if let Some(manifest) = load_manifest(root)? {
        return Ok(manifest);
    }

    let mut manifest = default_manifest();
    let discovered = discover_event_segment_ids(root)?;
    if let Some(max_segment) = discovered.iter().max() {
        manifest.active_segment = max_segment + 1;
    }
    Ok(manifest)
}

fn save_manifest(root: &Path, manifest: &StorageManifest) -> Result<()> {
    write_json_atomic(&manifest_path(root), manifest)
}

fn event_segment_ids(root: &Path, manifest: Option<&StorageManifest>) -> Result<Vec<u64>> {
    let mut ids = discover_event_segment_ids(root)?;
    if let Some(manifest) = manifest {
        ids.push(manifest.active_segment);
        ids.extend(manifest.segments.iter().map(|segment| segment.id));
    }
    ids.sort_unstable();
    ids.dedup();
    Ok(ids)
}

fn discover_event_segment_ids(root: &Path) -> Result<Vec<u64>> {
    let mut ids = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if let Some(id) = event_segment_id(file_name) {
            ids.push(id);
        }
    }
    ids.sort_unstable();
    ids.dedup();
    Ok(ids)
}

fn event_file_max_seq(path: &Path) -> Result<Option<u64>> {
    Ok(event_file_bounds(path)?.map(|(_, max_seq)| max_seq))
}

fn event_file_bounds(path: &Path) -> Result<Option<(u64, u64)>> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(Error::Storage(error)),
    };

    let mut reader = BufReader::new(file);
    let mut first = None;
    let mut last = 0;
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        let Ok(envelope) = serde_json::from_str::<SyncEventEnvelope>(&line) else {
            break;
        };
        let seq = envelope.seq.unwrap_or(last + 1);
        first.get_or_insert(seq);
        last = seq;
    }

    Ok(first.map(|first| (first, last)))
}

fn last_sync_event_seq(root: &Path) -> Result<u64> {
    let mut last = event_file_max_seq(&events_path(root))?.unwrap_or(0);
    for id in discover_event_segment_ids(root)? {
        last = last.max(event_file_max_seq(&event_segment_path(root, id))?.unwrap_or(0));
    }
    Ok(last)
}

fn upsert_segment_meta(manifest: &mut StorageManifest, meta: EventSegmentMeta) {
    if let Some(existing) = manifest
        .segments
        .iter_mut()
        .find(|segment| segment.id == meta.id)
    {
        *existing = meta;
    } else {
        manifest.segments.push(meta);
        manifest.segments.sort_by_key(|segment| segment.id);
    }
}

fn mark_prunable_segments(manifest: &mut StorageManifest) {
    let Some(min_checkpoint) = manifest.checkpoints.values().copied().min() else {
        return;
    };
    for segment in &mut manifest.segments {
        segment.prunable = segment.max_seq <= min_checkpoint;
    }
}

fn load_collection_views(root: &Path) -> Result<(CollectionViews, Option<u64>)> {
    let mut collections = BTreeMap::new();
    let mut checkpoint_seq: Option<u64> = None;

    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let file_path = entry.path();
        let Some(file_name) = file_path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(name) = file_name.strip_suffix(".json.zst") else {
            continue;
        };

        let documents = read_json_zstd(&file_path)?;
        collections.insert(name.to_string(), documents);
        if let Some(meta) = load_projection_meta(root, name)? {
            checkpoint_seq =
                Some(checkpoint_seq.map_or(meta.last_seq, |seq| seq.min(meta.last_seq)));
        }
    }

    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let file_path = entry.path();
        let file_name = file_path.file_name().and_then(|name| name.to_str());
        if matches!(
            file_name,
            Some(CATALOG_FILE) | Some(EVENTS_FILE) | Some(MANIFEST_FILE) | Some(LOCK_FILE)
        ) {
            continue;
        }
        if file_name.is_some_and(|name| name.ends_with(".meta.json")) {
            continue;
        }
        if file_path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }

        let Some(name) = file_path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        if collections.contains_key(name) {
            continue;
        }

        let raw = fs::read_to_string(&file_path)?;
        let documents = if raw.trim().is_empty() {
            Vec::new()
        } else {
            serde_json::from_str(&raw)?
        };
        collections.insert(name.to_string(), documents);
        if let Some(meta) = load_projection_meta(root, name)? {
            checkpoint_seq =
                Some(checkpoint_seq.map_or(meta.last_seq, |seq| seq.min(meta.last_seq)));
        }
    }

    Ok((collections, checkpoint_seq))
}

fn apply_sync_events(
    root: &Path,
    collections: &mut BTreeMap<String, Vec<Document>>,
    after_seq: Option<u64>,
    manifest: Option<&StorageManifest>,
) -> Result<u64> {
    let mut last_seq = 0;
    let legacy_path = events_path(root);

    if manifest
        .and_then(|manifest| manifest.legacy_max_seq)
        .zip(after_seq)
        .is_some_and(|(max_seq, after_seq)| max_seq <= after_seq)
    {
        last_seq = last_seq.max(
            manifest
                .and_then(|manifest| manifest.legacy_max_seq)
                .unwrap_or(0),
        );
    } else if legacy_path.exists() {
        last_seq = last_seq.max(apply_sync_event_file(&legacy_path, collections, after_seq)?);
    }

    for segment_id in event_segment_ids(root, manifest)? {
        let meta = manifest.and_then(|manifest| manifest.segment(segment_id));
        if meta
            .as_ref()
            .zip(after_seq)
            .is_some_and(|(meta, after_seq)| meta.max_seq <= after_seq)
        {
            last_seq = last_seq.max(meta.expect("segment meta").max_seq);
            continue;
        }
        last_seq = last_seq.max(apply_sync_event_file(
            &event_segment_path(root, segment_id),
            collections,
            after_seq,
        )?);
    }

    Ok(last_seq)
}

fn apply_sync_event_file(
    path: &Path,
    collections: &mut BTreeMap<String, Vec<Document>>,
    after_seq: Option<u64>,
) -> Result<u64> {
    let mut file = match OpenOptions::new().read(true).write(true).open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(0),
        Err(error) => return Err(Error::Storage(error)),
    };
    let mut reader = BufReader::new(file.try_clone()?);
    let mut valid_len = 0_u64;
    let mut expected_seq = None;
    let mut last_seq = 0;
    let mut line = String::new();

    loop {
        line.clear();
        let bytes = reader.read_line(&mut line)?;
        if bytes == 0 {
            break;
        }
        if line.trim().is_empty() {
            valid_len += bytes as u64;
            continue;
        }
        if let Some(after_seq) = after_seq {
            let Ok(envelope) = serde_json::from_str::<SyncEventEnvelope>(&line) else {
                truncate_to_valid_tail(&mut file, valid_len)?;
                break;
            };
            if let Some(seq) = envelope.seq {
                if !event_envelope_is_valid(&envelope, &line, &mut expected_seq) {
                    truncate_to_valid_tail(&mut file, valid_len)?;
                    break;
                }
                last_seq = seq;
                if seq <= after_seq {
                    valid_len += bytes as u64;
                    continue;
                }
                let Ok(event) = serde_json::from_str::<SyncEvent>(&line) else {
                    truncate_to_valid_tail(&mut file, valid_len)?;
                    break;
                };
                apply_record(collections, &event.collection, event.op);
                valid_len += bytes as u64;
                continue;
            }
        }
        let Ok(event) = serde_json::from_str::<SyncEvent>(&line) else {
            truncate_to_valid_tail(&mut file, valid_len)?;
            break;
        };
        if !event_is_valid(&event, &line, &mut expected_seq) {
            truncate_to_valid_tail(&mut file, valid_len)?;
            break;
        }
        last_seq = event.seq.unwrap_or(last_seq + 1);
        if event
            .seq
            .is_none_or(|seq| after_seq.is_none_or(|after| seq > after))
        {
            apply_record(collections, &event.collection, event.op);
        }
        valid_len += bytes as u64;
    }

    Ok(last_seq)
}

fn event_envelope_is_valid(
    event: &SyncEventEnvelope,
    raw_line: &str,
    expected_seq: &mut Option<u64>,
) -> bool {
    let Some(seq) = event.seq else {
        return true;
    };
    if event.ts.is_none() {
        return false;
    };
    let Some(checksum) = event.checksum else {
        return false;
    };
    if expected_seq.is_some_and(|expected| seq != expected) {
        return false;
    }
    let Some(actual) = checksum_from_raw_line(raw_line) else {
        return false;
    };
    if actual != checksum {
        return false;
    }
    *expected_seq = Some(seq + 1);
    true
}

fn event_is_valid(event: &SyncEvent, raw_line: &str, expected_seq: &mut Option<u64>) -> bool {
    let Some(seq) = event.seq else {
        return true;
    };
    if event.ts.is_none() {
        return false;
    };
    let Some(checksum) = event.checksum else {
        return false;
    };
    if expected_seq.is_some_and(|expected| seq != expected) {
        return false;
    }
    let Some(actual) = checksum_from_raw_line(raw_line) else {
        return false;
    };
    if actual != checksum {
        return false;
    }
    *expected_seq = Some(seq + 1);
    true
}

fn truncate_to_valid_tail(file: &mut File, valid_len: u64) -> Result<()> {
    file.set_len(valid_len)?;
    file.seek(SeekFrom::Start(valid_len))?;
    Ok(())
}

fn write_sync_event_line(file: &mut File, event_body: &str) -> Result<()> {
    let checksum = kernel::hash_bytes(event_body.as_bytes());
    let body = event_body.strip_suffix('}').unwrap_or(event_body);
    let mut line = Vec::with_capacity(body.len() + 34);
    line.extend_from_slice(body.as_bytes());
    line.extend_from_slice(b",\"checksum\":");
    write!(&mut line, "{checksum}")?;
    line.extend_from_slice(b"}\n");
    file.write_all(&line)?;
    Ok(())
}

fn checksum_from_raw_line(line: &str) -> Option<u64> {
    let line = line.trim_end_matches(['\n', '\r']);
    let checksum_start = line.rfind(",\"checksum\":")?;
    Some(fnv1a64_parts([&line.as_bytes()[..checksum_start], b"}"]))
}

fn fnv1a64_parts<const N: usize>(parts: [&[u8]; N]) -> u64 {
    let mut hash = 1469598103934665603_u64;
    for part in parts {
        for byte in part {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(1099511628211_u64);
        }
    }
    hash
}

fn apply_wal(path: &Path, documents: &mut Vec<Document>) -> Result<()> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(Error::Storage(error)),
    };

    let mut reader = BufReader::new(file);
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<WalRecord>(&line)? {
            WalRecord::Create => {}
            WalRecord::Drop => documents.clear(),
            WalRecord::Snapshot {
                documents: snapshot,
            } => *documents = snapshot,
            WalRecord::Insert {
                documents: inserted,
            } => documents.extend(inserted),
            WalRecord::Update { filter, updates } => {
                apply_update_many(documents, &filter, &updates)
            }
            WalRecord::Delete { filter } => apply_delete_many(documents, &filter),
            WalRecord::CleanupExpired { ttl_field, now } => {
                apply_cleanup_expired(documents, &ttl_field, now)
            }
        }
    }

    Ok(())
}

fn apply_record(
    collections: &mut BTreeMap<String, Vec<Document>>,
    collection: &str,
    record: WalRecord,
) {
    match record {
        WalRecord::Create => {
            collections.entry(collection.to_string()).or_default();
        }
        WalRecord::Drop => {
            collections.remove(collection);
        }
        WalRecord::Snapshot { documents } => {
            collections.insert(collection.to_string(), documents);
        }
        WalRecord::Insert { documents } => {
            collections
                .entry(collection.to_string())
                .or_default()
                .extend(documents);
        }
        WalRecord::Update { filter, updates } => {
            let documents = collections.entry(collection.to_string()).or_default();
            apply_update_many(documents, &filter, &updates);
        }
        WalRecord::Delete { filter } => {
            let documents = collections.entry(collection.to_string()).or_default();
            apply_delete_many(documents, &filter);
        }
        WalRecord::CleanupExpired { ttl_field, now } => {
            let documents = collections.entry(collection.to_string()).or_default();
            apply_cleanup_expired(documents, &ttl_field, now);
        }
    }
}

fn truncate_wal(root: &Path, name: &str) -> Result<()> {
    match fs::remove_file(wal_path(root, name)) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(Error::Storage(error)),
    }
}

fn load_projection_meta(root: &Path, name: &str) -> Result<Option<ProjectionMeta>> {
    let path = collection_meta_path(root, name);
    match fs::read_to_string(path) {
        Ok(raw) if raw.trim().is_empty() => Ok(None),
        Ok(raw) => Ok(Some(serde_json::from_str(&raw)?)),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(Error::Storage(error)),
    }
}

fn write_json_atomic<T: Serialize + ?Sized>(path: &Path, value: &T) -> Result<()> {
    let tmp_path = path.with_extension("json.tmp");
    let body = serde_json::to_string_pretty(value)?;
    fs::write(&tmp_path, body)?;
    fs::rename(tmp_path, path)?;
    Ok(())
}

fn write_json_zstd_atomic<T: Serialize + ?Sized>(path: &Path, value: &T) -> Result<()> {
    let tmp_path = path.with_extension("zst.tmp");
    let body = serde_json::to_vec(value)?;
    let compressed = zstd::stream::encode_all(body.as_slice(), 3)?;
    fs::write(&tmp_path, compressed)?;
    fs::rename(tmp_path, path)?;
    Ok(())
}

fn read_json_zstd<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let compressed = fs::read(path)?;
    let body = zstd::stream::decode_all(compressed.as_slice())?;
    Ok(serde_json::from_slice(&body)?)
}

fn remove_if_exists(path: &Path) -> Result<()> {
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(Error::Storage(error)),
    }
}

fn collection_path(root: &Path, name: &str) -> PathBuf {
    root.join(format!("{}.json", safe_collection_name(name)))
}

fn collection_compressed_path(root: &Path, name: &str) -> PathBuf {
    root.join(format!("{}.json.zst", safe_collection_name(name)))
}

fn collection_meta_path(root: &Path, name: &str) -> PathBuf {
    root.join(format!("{}.meta.json", safe_collection_name(name)))
}

fn safe_collection_name(name: &str) -> String {
    let safe_name: String = name
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' => ch,
            _ => '_',
        })
        .collect();
    safe_name
}

fn wal_path(root: &Path, name: &str) -> PathBuf {
    collection_path(root, name).with_extension("wal")
}

fn events_path(root: &Path) -> PathBuf {
    root.join(EVENTS_FILE)
}

fn manifest_path(root: &Path) -> PathBuf {
    root.join(MANIFEST_FILE)
}

fn event_segment_path(root: &Path, id: u64) -> PathBuf {
    root.join(format!("__nosqlite_events.{id:06}.jsonl"))
}

fn event_segment_id(file_name: &str) -> Option<u64> {
    file_name
        .strip_prefix("__nosqlite_events.")?
        .strip_suffix(".jsonl")?
        .parse()
        .ok()
}

fn now_epoch_seconds() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs() as i64)
        .unwrap_or(0)
}
