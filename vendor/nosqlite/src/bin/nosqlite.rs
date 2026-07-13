use std::{
    io::{self, BufRead, Write},
    path::PathBuf,
};

use clap::Parser;
use nosqlite::{CommandResult, Durability, Engine, EngineOptions, StorageMode};

#[derive(Debug, Parser)]
#[command(about = "MongoDB-shaped embedded document database shell")]
struct Args {
    #[arg(long)]
    data_dir: Option<PathBuf>,
    #[arg(long, default_value = "sync", value_parser = ["sync", "buffered"])]
    durability: String,
}

fn main() -> nosqlite::Result<()> {
    let args = Args::parse();
    let storage = args
        .data_dir
        .map(StorageMode::FileSystem)
        .unwrap_or(StorageMode::Memory);
    let engine = Engine::new(EngineOptions {
        storage,
        durability: parse_durability(&args.durability),
        ..EngineOptions::default()
    })?;

    let stdin = io::stdin();
    let mut stdin = stdin.lock();
    let mut stdout = io::stdout();
    let mut line = String::new();

    writeln!(stdout, "nosqlite ready; send one JSON command per line")?;
    loop {
        line.clear();
        if stdin.read_line(&mut line)? == 0 {
            break;
        }
        if line.trim().is_empty() {
            continue;
        }

        let command = match serde_json::from_str(&line) {
            Ok(command) => command,
            Err(error) => {
                writeln!(
                    stdout,
                    "{{\"ok\":\"error\",\"message\":{}}}",
                    serde_json::to_string(&error.to_string())?
                )?;
                stdout.flush()?;
                continue;
            }
        };

        if is_shutdown(&command) {
            serde_json::to_writer(&mut stdout, &CommandResult::Shutdown)?;
            stdout.write_all(b"\n")?;
            stdout.flush()?;
            break;
        }

        match engine.execute(command) {
            Ok(result) => {
                serde_json::to_writer(&mut stdout, &result)?;
                stdout.write_all(b"\n")?;
            }
            Err(error) => writeln!(
                stdout,
                "{{\"ok\":\"error\",\"message\":{}}}",
                serde_json::to_string(&error.to_string())?
            )?,
        }
        stdout.flush()?;
    }

    Ok(())
}

fn parse_durability(value: &str) -> Durability {
    match value {
        "buffered" => Durability::Buffered,
        _ => Durability::Sync,
    }
}

fn is_shutdown(command: &serde_json::Value) -> bool {
    command
        .as_object()
        .is_some_and(|command| command.get("shutdown").is_some())
}
