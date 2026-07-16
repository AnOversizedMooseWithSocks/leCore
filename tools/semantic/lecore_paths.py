#!/usr/bin/env python3
"""lecore_paths.py -- one place that knows where everything lives. NumPy-free, stdlib only.

THE LAYOUT (Moose's machine):

    <parent>/
        leCore/                       the repository (importable: `import lecore`)
        scripts/                      <- everything below, and this file
            lecore_paths.py
            nomic_census.py, llm_census.py, distill_router.py, model_analysis_program.py, ...
            .knowledge_cache.json     content-addressed embedding cache
            nomic_text/               config.json, model.safetensors, vocab.txt
            qwen3.5_0.8b/             config.json, tokenizer.json, model.safetensors-00001-of-00001.safetensors, ...
            smol/                     SmolLM2-135M

WHY A MODULE AND NOT A CONSTANT IN EACH SCRIPT:
Eight scripts hard-coding the same three paths is eight places to break when a folder is renamed --
the same "gap" failure mode the session rules warn about, just in the filesystem. One module, one
truth, and every script's argparse defaults come from here. Pass an explicit path to override.

Every lookup is a GLOB, not a literal: Qwen ships its weights as
`model.safetensors-00001-of-00001.safetensors`, nomic as plain `model.safetensors`, and sharded
models as `model-00001-of-000NN.safetensors`. Guessing the filename is how a script dies at 3 AM.
"""
import sys, pathlib


# This file lives at <repo>/tools/semantic/. The repo root is therefore two levels up -- computed from
# __file__, NOT a hardcoded folder name, so it works on any clone regardless of what the repo dir is
# called (a hardcoded 'leCore' sibling broke every clone that renamed it -- e.g. 'holostuff').
SCRIPTS = pathlib.Path(__file__).resolve().parent          # tools/semantic (weights/cache live here)
REPO = SCRIPTS.parent.parent                               # <repo> root (has holographic/, lecore_data/)
PARENT = REPO.parent                                       # kept for any old references

NOMIC_DIR = SCRIPTS / 'nomic_text'
QWEN_DIR = SCRIPTS / 'qwen3.5_0.8b'
SMOL_DIR = SCRIPTS / 'smol'

CACHE = SCRIPTS / '.knowledge_cache.json'


def _one(directory, *patterns, what='file'):
    """First match across patterns, in order. Fails LOUDLY with what it looked for -- a silent
    fallback to the wrong file is worse than a crash."""
    for pat in patterns:
        hits = sorted(directory.glob(pat))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"no {what} in {directory}\n  looked for: {', '.join(patterns)}\n"
        f"  present: {sorted(p.name for p in directory.glob('*'))[:8] if directory.is_dir() else '<dir missing>'}")


def weights(directory):
    """The safetensors file, whatever the publisher decided to call it."""
    return _one(directory,
                'model.safetensors',                          # nomic, smol
                'model.safetensors-*-of-*.safetensors',       # qwen3.5 (yes, really)
                'model-*-of-*.safetensors',                   # ordinary shards
                '*.safetensors',
                what='safetensors')


def nomic_weights(): return weights(NOMIC_DIR)
def nomic_vocab():   return _one(NOMIC_DIR, 'vocab.txt', what='vocab.txt')
def qwen_weights():  return weights(QWEN_DIR)
def qwen_tokenizer():return _one(QWEN_DIR, 'tokenizer.json', what='tokenizer.json')
def qwen_config():   return _one(QWEN_DIR, 'config.json', what='config.json')
def smol_weights():  return weights(SMOL_DIR)
def smol_tokenizer():return _one(SMOL_DIR, 'tokenizer.json', what='tokenizer.json')


def add_repo_to_path():
    """So `import lecore` works from scripts/ without installing anything."""
    if REPO.is_dir() and str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    return REPO


def report():
    print(f"  scripts : {SCRIPTS}")
    print(f"  repo    : {REPO}  {'OK' if REPO.is_dir() else 'MISSING'}")
    print(f"  cache   : {CACHE}  {'OK' if CACHE.is_file() else 'missing (built on first run)'}")
    for name, fn in (('nomic weights', nomic_weights), ('nomic vocab', nomic_vocab),
                     ('qwen weights', qwen_weights), ('qwen tokenizer', qwen_tokenizer),
                     ('smol weights', smol_weights)):
        try:
            p = fn()
            print(f"  {name:16s}: {p.name}  ({p.stat().st_size/1e6:.0f} MB)")
        except FileNotFoundError as e:
            print(f"  {name:16s}: -- {str(e).splitlines()[0]}")


if __name__ == '__main__':
    report()
