"""swebench_rerank.py -- STAGE 2: re-rank the top-30 path candidates by their CONTENTS (still no model).

For each instance: fetch the top-30 candidates' text at base_commit in ONE `git archive` call (the blobless
clone lazily fetches missing blobs in a batch), tokenise identifiers, and score BM25 over those 30 documents
with the issue's identifiers as the query, blended with the path score. Checkpointed to JSONL per instance so
a reaped run resumes. Run detached: ~2-5 s per instance.

    PYTHONHASHSEED=0 nohup python3 swebench_rerank.py > rerank.log 2>&1 &
"""
import io
import json
import math
import os
import re
import subprocess
import tarfile
import collections
import sys

sys.path.insert(0, "/home/claude/swebench")
from swebench_localize import toks, localize   # the same tokeniser and stage-1 ranker

rows = {r["instance_id"]: r for r in json.load(open("/home/claude/swebench/verified.json"))}
trees = json.load(open("/home/claude/swebench/trees_at_base.json"))
OUT = "/home/claude/swebench/rerank_results.jsonl"
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["id"] for l in open(OUT) if l.strip()}
TOPN = 30


def fetch(repo, sha, paths):
    """Contents of `paths` at `sha` via one git archive call (one batched lazy fetch)."""
    p = subprocess.run(["git", "archive", "--format=tar", sha] + paths, cwd="/home/claude/swebench/repos/%s.git" % repo,
                       capture_output=True, timeout=600)
    out = {}
    if p.returncode != 0:
        return out
    with tarfile.open(fileobj=io.BytesIO(p.stdout)) as tf:
        for m in tf.getmembers():
            if m.isfile():
                try:
                    out[m.name] = tf.extractfile(m).read().decode("utf-8", "ignore")
                except Exception:
                    pass
    return out


def bm25_rank(query_tokens, docs, k1=1.2, b=0.75):
    """Plain BM25 over the candidate documents; the query is the issue's identifier tokens."""
    dl = {f: len(t) for f, t in docs.items()}
    avg = sum(dl.values()) / max(len(dl), 1)
    df = collections.Counter()
    tfs = {}
    for f, t in docs.items():
        c = collections.Counter(t)
        tfs[f] = c
        df.update(c.keys())
    N = len(docs)
    q = collections.Counter(query_tokens)
    scores = {}
    for f in docs:
        s = 0.0
        for term, qn in q.items():
            if term not in tfs[f]:
                continue
            idf = math.log(1 + (N - df[term] + 0.5) / (df[term] + 0.5))
            tf = tfs[f][term]
            s += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl[f] / avg)) * min(qn, 3)
        scores[f] = s
    return scores


with open(OUT, "a") as fh:
    for iid, inst in rows.items():
        if iid in done:
            continue
        files = trees.get(iid, [])
        top, margin = localize(inst, files, k=TOPN)
        repo = inst["repo"].split("/")[1]
        docs_raw = fetch(repo, inst["base_commit"], top)
        text = inst["problem_statement"] + " " + inst.get("hints_text", "")
        # identifiers in the issue: code-ish tokens carry the evidence (not the prose)
        ids = [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text) if ("_" in w or any(c.isupper() for c in w[1:]) or w.islower() and len(w) > 5)]
        qtok = toks(" ".join(ids))
        docs = {f: toks(docs_raw.get(f, "")) for f in top}
        content = bm25_rank(qtok, docs) if docs else {}
        # blend: rank-based, so the two scales do not fight (the doc-01 lesson)
        path_rank = {f: i for i, f in enumerate(top)}
        cont_sorted = sorted(top, key=lambda f: -content.get(f, 0.0))
        cont_rank = {f: i for i, f in enumerate(cont_sorted)}
        fused = sorted(top, key=lambda f: 1.0 / (60 + path_rank[f]) + 1.0 / (60 + cont_rank[f]), reverse=True)  # reciprocal rank fusion
        gold = set(inst["gold_files"])
        rec = {"id": iid, "repo": inst["repo"], "gold": sorted(gold), "path_top": top[:10], "content_top": cont_sorted[:10], "fused_top": fused[:10],
               "fetched": len(docs_raw), "content_margin": (sorted(content.values(), reverse=True)[:2] + [0, 0])[0] - (sorted(content.values(), reverse=True)[:2] + [0, 0])[1]}
        for name, lst in (("path", top), ("content", cont_sorted), ("fused", fused)):
            rec[name] = {str(k): bool(gold & set(lst[:k])) for k in (1, 3, 5, 10)}
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print(iid, "fetched", len(docs_raw), "| path top1", rec["path"]["1"], "content top1", rec["content"]["1"], "fused top1", rec["fused"]["1"], flush=True)
