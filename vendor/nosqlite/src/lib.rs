mod encoder;
mod engine;
mod index;
mod kernel;
mod mutation;
mod neural;
mod query;
mod storage;

pub use encoder::{encode_text, EncoderSpec};
pub use engine::{CommandResult, Engine, EngineOptions, SortSpec, StorageMode};
pub use mutation::apply_updates;
pub use neural::{NeuralSpace, Prototype};
pub use query::matches_filter;
pub use storage::Durability;

pub type Document = serde_json::Map<String, serde_json::Value>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("collection `{0}` already exists")]
    CollectionExists(String),
    #[error("collection `{0}` does not exist")]
    CollectionMissing(String),
    #[error("command is missing required field `{0}`")]
    MissingField(&'static str),
    #[error("unsupported command")]
    UnsupportedCommand,
    #[error("expected JSON object for `{0}`")]
    ExpectedObject(&'static str),
    #[error("expected JSON array for `{0}`")]
    ExpectedArray(&'static str),
    #[error("index `{0}` already exists")]
    IndexExists(String),
    #[error("index `{0}` does not exist")]
    IndexMissing(String),
    #[error("encoder `{0}` already exists")]
    EncoderExists(String),
    #[error("encoder `{0}` does not exist")]
    EncoderMissing(String),
    #[error("unsupported encoder `{0}`")]
    UnsupportedEncoder(String),
    #[error("neural space `{0}` already exists")]
    NeuralSpaceExists(String),
    #[error("neural space `{0}` does not exist")]
    NeuralSpaceMissing(String),
    #[error("unique index `{index}` rejected duplicate value for `{field}`")]
    UniqueIndexViolation { index: String, field: String },
    #[error("vector field `{field}` expected {expected} dimensions, got {actual}")]
    VectorDimensionMismatch {
        field: String,
        expected: usize,
        actual: usize,
    },
    #[error("storage error: {0}")]
    Storage(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;
