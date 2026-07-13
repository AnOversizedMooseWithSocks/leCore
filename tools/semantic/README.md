# tools/semantic -- the embedding coverage gate

CI home of the semantic router's build+exam (workflow: .github/workflows/semantic-coverage.yml).
knowledge_index.py builds the content-addressed embedding cache over the engine's own text
(docstrings, catalog, NOTES incl. kept negatives) and runs the 12-ask routing suite;
`--exam --require-top5 8 --require-median 2` turns the suite into a merge gate.
Weights are NOT committed: CI fetches them from the pinned NOMIC_WEIGHTS_URL / NOMIC_VOCAB_URL
repo variables and caches them. Local use: see the sibling scripts/ folder's README.
