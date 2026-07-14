use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nosqlite::{CommandResult, Engine, EngineOptions, StorageMode};
use serde_json::{json, Map, Value};
use tempfile::tempdir;

const BATCH_SIZE: usize = 500;

fn bench_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_memory");

    for docs in [1_000usize, 10_000, 50_000] {
        group.throughput(Throughput::Elements(docs as u64));
        group.bench_with_input(BenchmarkId::new("batch_insert", docs), &docs, |b, &docs| {
            b.iter(|| {
                let engine = Engine::new(EngineOptions::default()).unwrap();
                insert_docs(&engine, "events", docs);
                black_box(engine);
            });
        });

        group.bench_with_input(BenchmarkId::new("point_find", docs), &docs, |b, &docs| {
            let engine = seeded_memory_engine(docs);
            b.iter(|| {
                for index in (0..250).map(|n| (n * 37) % docs) {
                    let result = engine
                        .find(
                            "events",
                            Some(filter("external_id", event_id(index))),
                            Some(1),
                            None,
                            None,
                        )
                        .unwrap();
                    black_box(result);
                }
            });
        });

        group.bench_with_input(
            BenchmarkId::new("indexed_point_find", docs),
            &docs,
            |b, &docs| {
                let engine = seeded_indexed_memory_engine(docs);
                b.iter(|| {
                    for index in (0..250).map(|n| (n * 37) % docs) {
                        let result = engine
                            .find(
                                "events",
                                Some(filter("external_id", event_id(index))),
                                Some(1),
                                None,
                                None,
                            )
                            .unwrap();
                        black_box(result);
                    }
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("filtered_update", docs),
            &docs,
            |b, &docs| {
                b.iter_batched(
                    || seeded_memory_engine(docs),
                    |engine| {
                        let result = engine
                            .update(
                                "events",
                                Some(filter("bucket", json!(7))),
                                set_update("state", json!("hot")),
                            )
                            .unwrap();
                        black_box(result);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(
            BenchmarkId::new("neural_vector_search", docs),
            &docs,
            |b, &docs| {
                let engine = seeded_neural_memory_engine(docs);
                b.iter(|| {
                    for index in (0..100).map(|n| (n * 37) % docs) {
                        let result = engine
                            .vector_search(
                                "vectors",
                                Some("embedding_neural"),
                                None,
                                embedding(index),
                                10,
                                None,
                            )
                            .unwrap();
                        black_box(result);
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_filesystem(c: &mut Criterion) {
    let mut group = c.benchmark_group("engine_filesystem");

    for docs in [1_000usize, 5_000] {
        group.throughput(Throughput::Elements(docs as u64));
        group.bench_with_input(
            BenchmarkId::new("persist_insert", docs),
            &docs,
            |b, &docs| {
                b.iter_batched(
                    || tempdir().unwrap(),
                    |dir| {
                        let engine = Engine::new(EngineOptions {
                            storage: StorageMode::FileSystem(dir.path().to_path_buf()),
                            ..EngineOptions::default()
                        })
                        .unwrap();
                        insert_docs(&engine, "events", docs);
                        black_box(engine);
                    },
                    criterion::BatchSize::SmallInput,
                );
            },
        );

        group.bench_with_input(BenchmarkId::new("cold_load", docs), &docs, |b, &docs| {
            b.iter_batched(
                || {
                    let dir = tempdir().unwrap();
                    let engine = Engine::new(EngineOptions {
                        storage: StorageMode::FileSystem(dir.path().to_path_buf()),
                        ..EngineOptions::default()
                    })
                    .unwrap();
                    insert_docs(&engine, "events", docs);
                    drop(engine);
                    dir
                },
                |dir| {
                    let engine = Engine::new(EngineOptions {
                        storage: StorageMode::FileSystem(dir.path().to_path_buf()),
                        ..EngineOptions::default()
                    })
                    .unwrap();
                    let result = engine
                        .find(
                            "events",
                            Some(filter("bucket", json!(3))),
                            Some(10),
                            None,
                            None,
                        )
                        .unwrap();
                    black_box(result);
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn seeded_memory_engine(docs: usize) -> Engine {
    let engine = Engine::new(EngineOptions::default()).unwrap();
    insert_docs(&engine, "events", docs);
    engine
}

fn seeded_indexed_memory_engine(docs: usize) -> Engine {
    let engine = seeded_memory_engine(docs);
    engine
        .create_indexes(
            "events",
            &[json!({
                "key": { "external_id": 1 },
                "name": "external_id_1",
                "unique": true
            })],
        )
        .unwrap();
    engine
}

fn seeded_neural_memory_engine(docs: usize) -> Engine {
    let engine = Engine::new(EngineOptions::default()).unwrap();
    for start in (0..docs).step_by(BATCH_SIZE) {
        let end = (start + BATCH_SIZE).min(docs);
        let batch = (start..end)
            .map(|index| {
                json!({
                    "_id": format!("vector-{index:08}"),
                    "embedding": embedding(index),
                    "bucket": index % 20
                })
            })
            .collect::<Vec<_>>();
        let CommandResult::Inserted { count, .. } = engine.insert("vectors", &batch).unwrap()
        else {
            unreachable!("insert returns inserted result");
        };
        assert_eq!(count, end - start);
    }
    engine
        .create_indexes(
            "vectors",
            &[json!({
                "neural": "embedding",
                "dimensions": 16,
                "name": "embedding_neural"
            })],
        )
        .unwrap();
    engine
}

fn insert_docs(engine: &Engine, collection: &str, docs: usize) {
    for start in (0..docs).step_by(BATCH_SIZE) {
        let end = (start + BATCH_SIZE).min(docs);
        let batch = (start..end).map(document).collect::<Vec<_>>();
        let CommandResult::Inserted { count, .. } = engine.insert(collection, &batch).unwrap()
        else {
            unreachable!("insert returns inserted result");
        };
        assert_eq!(count, end - start);
    }
}

fn document(index: usize) -> Value {
    json!({
        "external_id": event_id(index),
        "bucket": index % 20,
        "score": (index % 10_000) as f64 / 100.0,
        "state": "warm",
        "payload": {
            "model": "embedding-v1",
            "dimensions": 384,
            "tokens": index % 8192
        }
    })
}

fn event_id(index: usize) -> Value {
    json!(format!("event-{index:08}"))
}

fn filter(key: &str, value: Value) -> Map<String, Value> {
    Map::from_iter([(key.to_string(), value)])
}

fn set_update(key: &str, value: Value) -> Map<String, Value> {
    Map::from_iter([("$set".to_string(), json!({ key: value }))])
}

fn embedding(index: usize) -> Vec<f64> {
    let phase = index as f64 * 0.017;
    (0..16)
        .map(|dim| {
            let dim = dim as f64 + 1.0;
            (phase * dim).sin() + (phase / dim).cos() * 0.25
        })
        .collect()
}

criterion_group!(benches, bench_memory, bench_filesystem);
criterion_main!(benches);
