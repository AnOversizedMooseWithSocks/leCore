# Applying this archive over a live repo

This archive is a SNAPSHOT OF THE SOURCE TREE. Four kinds of file in the repo are owned by CI, not by this
snapshot, and applying the archive over them REVERTS whatever CI last committed. They are deliberately
EXCLUDED from this archive (`VERSION`, the routing index + seed) or safe to let CI rebuild (generated docs).

| path | owner | if you overwrite it |
|---|---|---|
| `VERSION` | `package.yml` (auto patch-bump per release) | the version goes BACKWARDS; the next bump collides with a number already on PyPI and the upload is rejected |
| `lecore_data/routing/index_128d.npz` | `semantic-coverage.yml` | the shipped index goes stale against the corpus; the seed/index lockstep test fails on main |
| `tools/semantic/routing_seed.npz.xz` | `semantic-coverage.yml` | same, the other half of the pair |
| `REFERENCE.md`, `CAPABILITIES.md`, `capabilities.json`, `API_QUICKREF.md`, `docs/FACULTY_MAP.md`, `docs/DOC_MAP.md`, `docs/PIPELINE_MAP.md`, `pipelines.json` | `docs.yml` | harmless: the next push regenerates them. They are included so a standalone extract is complete. |

## Diagnosing "N files changed but the diff looks empty"

Git renders three different things as an empty-looking diff. This names which one you have:

```sh
git diff --numstat | awk '$1 == 0 && $2 == 0 { print }'   # 0 added / 0 deleted -> mode or binary change
git diff --summary                                        # mode changes, printed explicitly
git ls-files --eol | grep -v 'i/lf'                       # files whose INDEX copy is not LF
```

* rows from the first command with a `-`/`-` count are BINARY files (git never shows their contents);
* `mode change 100644 => 100755` in the second is a permission-only change (this archive stores everything
  0644, no executable bits);
* anything listed by the third is an index/worktree line-ending mismatch against `.gitattributes`
  (`* text=auto`), fixed once and for all with `git add --renormalize . && git commit`.

The source files in this archive are verified LF-only for every text type -- `tests/test_repo_layout.py`
pins it, and the pin is mutation-tested.
