#!/bin/sh
# Fetch the two real BEIR test collections from their GitHub mirrors (data not vendored: licenses + size).
# NFCorpus: 3,633 docs / 323 test queries / graded qrels.  SciFact: 5,183 docs / 300 test claims.
set -e
mkdir -p data && cd data
curl -sL https://codeload.github.com/keeeevinShen/RAG_nfcorpus/zip/refs/heads/main -o nf.zip && unzip -qo nf.zip
curl -sL https://codeload.github.com/mukuuund/scifact-retrieval-system/zip/refs/heads/main -o sf.zip && unzip -qo sf.zip "scifact-retrieval-system-main/data/corpus.jsonl"
curl -sL https://codeload.github.com/saikrishnab-sahajai/retrieval-evolution-study/zip/refs/heads/main -o res.zip && unzip -qo res.zip "retrieval-evolution-study-main/data/datasets/scifact/*"
echo "done; point the bench scripts' root at $(pwd)"
