#!/bin/bash
set -Eeuo pipefail

exec > >(tee -a /var/log/qwen35-user-data.log) 2>&1

RUN_ROOT=/opt/qwen35-acceptance-v7
RESULT_DIR=${RUN_ROOT}/result
PROJECT_DIR=${RUN_ROOT}/project
WORKSPACE_DIR=${RUN_ROOT}/workspace
MODEL_DIR=${RUN_ROOT}/model
INSTALLED_DIR=${RUN_ROOT}/installed-checkpoint
LECORE_DIR=${RUN_ROOT}/lecore
ILXYR_DIR=${RUN_ROOT}/ilxyr
VENV_DIR=${RUN_ROOT}/venv
EVALUATION_CORPUS=${RUN_ROOT}/inputs/federalist-papers.txt
LAUNCH_MANIFEST=${RUN_ROOT}/inputs/launch-manifest.json
BENCHMARK_REPORT=${RESULT_DIR}/benchmark-report.json
BUCKET=zero-training-022118847419
PREFIX=qwen35-acceptance/cb3b1d2-2fc06364-a6c9d113/v7-formal
LECORE_COMMIT=cb3b1d2ac71c183bf9307ca7145a2a619ff30c30
ILXYR_COMMIT=e92382ff2a5e8714466533a160f6609b4ef9cee8
MODEL_REVISION=2fc06364715b967f1860aea9cf38778875588b17
INSTALL_SHA=e4323ead9159c09ceb1e656644eb24b6ebf996ee9a00ac0c584a521c3acf2c37
EVALUATION_SHA=a6c9d1135a04d10955fe11d210b7f642e1c2341d4f2c8369b9a832cc97839d94
WEIGHTS_SHA=04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696
TOKENIZER_SHA=5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42
REQUIREMENTS_SHA=91193feb4e831e25ec3d4988c22fd78f9d3e770409a8d7cafda2cf50b6189b78
LAUNCH_SHA=9863b88a92448769de0b68162e7e9485aad836f522dd9dc56e6558ed82835a6c
STAGE=boot
FORMAL_RUN_STARTED=false
EXPERIMENT_ID=
FINALIZED=false

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export LECORE_CC_CACHE=${RUN_ROOT}/cc-cache
export TOKENIZERS_PARALLELISM=false

mkdir -p "${RESULT_DIR}" "${RUN_ROOT}/inputs"

# Instance-local cost guard. EC2 is configured to terminate, not stop, when
# this fires. Five hours of r7i.2xlarge compute is $2.646; combined
# with the estimated $5.54 already spent, this stays below $10.
shutdown -h +300

write_status() {
  local exit_code=${1:-0}
  python3 - "${RESULT_DIR}/runner-status.json" "${STAGE}" "${exit_code}" \
    "${FORMAL_RUN_STARTED}" "${EXPERIMENT_ID}" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, stage, exit_code, formal, experiment_id = sys.argv[1:]
with open(path, "w", encoding="utf-8") as handle:
    json.dump({
        "schema": "lecore.qwen35-runner-status.v7",
        "stage": stage,
        "exit_code": int(exit_code),
        "formal_run_started": formal == "true",
        "experiment_id": experiment_id or None,
        "source_commit": "cb3b1d2ac71c183bf9307ca7145a2a619ff30c30",
        "experiment_version": 7,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
  aws s3 cp "${RESULT_DIR}/runner-status.json" \
    "s3://${BUCKET}/${PREFIX}/status.json" --only-show-errors || true
  aws s3 cp /var/log/qwen35-user-data.log \
    "s3://${BUCKET}/${PREFIX}/live/user-data.log" --only-show-errors || true
}

capture_instance_identity() {
  local token
  token=$(curl -fsS -X PUT \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600' \
    http://169.254.169.254/latest/api/token) || return 0
  curl -fsS -H "X-aws-ec2-metadata-token: ${token}" \
    http://169.254.169.254/latest/dynamic/instance-identity/document \
    > "${RESULT_DIR}/aws-instance-identity.json" || true
}

collect_bundle() {
  set +e
  capture_instance_identity
  cp /var/log/qwen35-user-data.log "${RESULT_DIR}/user-data.log" 2>/dev/null
  cp /var/log/cloud-init-output.log "${RESULT_DIR}/cloud-init-output.log" 2>/dev/null
  cp /var/lib/cloud/instance/user-data.txt "${RESULT_DIR}/bootstrap-user-data.sh" 2>/dev/null
  cp "${RUN_ROOT}/inputs/runner.sh" "${RESULT_DIR}/runner.sh" 2>/dev/null
  cp "${LAUNCH_MANIFEST}" "${RESULT_DIR}/launch-manifest.json" 2>/dev/null
  cp "${EVALUATION_CORPUS}" "${RESULT_DIR}/evaluation-corpus.txt" 2>/dev/null
  cp "${LECORE_DIR}/REFERENCE.md" "${RESULT_DIR}/installation-corpus.txt" 2>/dev/null
  cp "${LECORE_DIR}/experiments/qwen35_acceptance/requirements-cpu.txt" \
    "${RESULT_DIR}/requirements-cpu.txt" 2>/dev/null
  if [ -d "${INSTALLED_DIR}.acceptance" ]; then
    cp -R "${INSTALLED_DIR}.acceptance" "${RESULT_DIR}/acceptance-artifacts"
  fi
  if [ -d "${LECORE_DIR}/.git" ]; then
    git -C "${LECORE_DIR}" status --porcelain --untracked-files=no \
      > "${RESULT_DIR}/lecore-status.txt" 2>&1
    git -C "${LECORE_DIR}" show -s --format=fuller HEAD \
      > "${RESULT_DIR}/lecore-commit.txt" 2>&1
  fi
  if [ -d "${ILXYR_DIR}/.git" ]; then
    git -C "${ILXYR_DIR}" show -s --format=fuller HEAD \
      > "${RESULT_DIR}/ilxyr-commit.txt" 2>&1
  fi
  local bundle_roots=(result)
  [ -d "${PROJECT_DIR}" ] && bundle_roots+=(project)
  [ -d "${WORKSPACE_DIR}" ] && bundle_roots+=(workspace)
  tar -C "${RUN_ROOT}" -czf "${RUN_ROOT}/qwen35-evidence.tgz" \
    "${bundle_roots[@]}" 2>/dev/null
  sha256sum "${RUN_ROOT}/qwen35-evidence.tgz" \
    > "${RUN_ROOT}/qwen35-evidence.tgz.sha256" 2>/dev/null
  aws s3 cp "${RUN_ROOT}/qwen35-evidence.tgz" \
    "s3://${BUCKET}/${PREFIX}/qwen35-evidence.tgz" --only-show-errors
  aws s3 cp "${RUN_ROOT}/qwen35-evidence.tgz.sha256" \
    "s3://${BUCKET}/${PREFIX}/qwen35-evidence.tgz.sha256" --only-show-errors
  aws s3 cp "${RESULT_DIR}" "s3://${BUCKET}/${PREFIX}/result/" \
    --recursive --only-show-errors
  FINALIZED=true
  set -e
}

finish() {
  local exit_code=$?
  trap - EXIT
  set +e
  if [ "${exit_code}" -eq 0 ]; then
    STAGE=complete
  elif [ "${FORMAL_RUN_STARTED}" = true ]; then
    STAGE=execution_failure
  else
    STAGE=preflight_failure
  fi
  write_status "${exit_code}"
  if [ "${FINALIZED}" != true ]; then
    collect_bundle
  fi
  write_status "${exit_code}"
  shutdown -h now
  exit "${exit_code}"
}
trap finish EXIT

write_status 0

STAGE=system_setup
write_status 0
dnf install -y git gcc python3.12 python3.12-pip rust cargo tar gzip
if ! command -v aws >/dev/null 2>&1; then
  dnf install -y awscli2 || dnf install -y awscli
fi

aws s3 cp "s3://${BUCKET}/${PREFIX}/input/federalist-papers.txt" \
  "${EVALUATION_CORPUS}" --only-show-errors
aws s3 cp "s3://${BUCKET}/${PREFIX}/input/launch-manifest.json" \
  "${LAUNCH_MANIFEST}" --only-show-errors
echo "${LAUNCH_SHA}  ${LAUNCH_MANIFEST}" | sha256sum -c -

STAGE=source_materialization
write_status 0
git clone --filter=blob:none --branch codex/qwen-ilxyr-experiments \
  https://github.com/atimics/holostuff.git "${LECORE_DIR}"
git -C "${LECORE_DIR}" checkout --detach "${LECORE_COMMIT}"
test "$(git -C "${LECORE_DIR}" rev-parse HEAD)" = "${LECORE_COMMIT}"
test -z "$(git -C "${LECORE_DIR}" status --porcelain --untracked-files=no)"
echo "${INSTALL_SHA}  ${LECORE_DIR}/REFERENCE.md" | sha256sum -c -
echo "${EVALUATION_SHA}  ${EVALUATION_CORPUS}" | sha256sum -c -
echo "${REQUIREMENTS_SHA}  ${LECORE_DIR}/experiments/qwen35_acceptance/requirements-cpu.txt" \
  | sha256sum -c -

git clone https://github.com/cenetex/ilXyr.git "${ILXYR_DIR}"
git -C "${ILXYR_DIR}" checkout --detach "${ILXYR_COMMIT}"
test "$(git -C "${ILXYR_DIR}" rev-parse HEAD)" = "${ILXYR_COMMIT}"
cargo build --locked --release -p ilxyr-cli --manifest-path "${ILXYR_DIR}/Cargo.toml"
ILXYR=${ILXYR_DIR}/target/release/ilxyr

STAGE=dependency_materialization
write_status 0
python3.12 -m venv "${VENV_DIR}"
PYTHON=${VENV_DIR}/bin/python
PIP=${VENV_DIR}/bin/pip
"${PIP}" install -r "${LECORE_DIR}/experiments/qwen35_acceptance/requirements-cpu.txt"
"${PIP}" install pytest==9.0.2
"${PIP}" check
"${PIP}" freeze --all | LC_ALL=C sort > "${RESULT_DIR}/requirements-resolved.txt"
"${PYTHON}" - <<'PY' > "${RESULT_DIR}/official-dependency-imports.txt"
import torch
import torchvision
import transformers
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("transformers", transformers.__version__)
print(AutoProcessor.__name__, Qwen3_5ForConditionalGeneration.__name__)
PY

STAGE=frozen_test_preflight
write_status 0
"${PYTHON}" -m pytest -q --run-slow \
  --junitxml="${RESULT_DIR}/frozen-preflight-junit.xml" \
  "${LECORE_DIR}/tests/test_qwen_acceptance.py" \
  "${LECORE_DIR}/tests/test_qwen_gdn_acceleration.py" \
  "${LECORE_DIR}/tests/test_qwen_runtime_benchmark.py" \
  "${LECORE_DIR}/tests/test_qwen_v4_policy_identity.py" \
  "${LECORE_DIR}/tests/test_qwen_cpu_environment.py" \
  > "${RESULT_DIR}/frozen-preflight-tests.txt" 2>&1
"${PYTHON}" - "${RESULT_DIR}/frozen-preflight-junit.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET

skipped = ET.parse(sys.argv[1]).getroot().findall(".//skipped")
if skipped:
    raise SystemExit("critical Qwen tests skipped: %d" % len(skipped))
PY
"${PYTHON}" "${LECORE_DIR}/tools/ci_qwen_ilxyr_preflight.py" \
  --ilxyr-cli "${ILXYR}" --ilxyr-source "${ILXYR_DIR}" \
  > "${RESULT_DIR}/miniature-ilxyr-preflight.json"
"${PYTHON}" - "${RESULT_DIR}/miniature-ilxyr-preflight.json" <<'PY'
import json
import sys

preflight = json.load(open(sys.argv[1]))
required = {
    "execution_started": True,
    "resolved_outcome": "accepted",
    "evidence_events": 1,
    "valid": True,
}
if any(preflight.get(key) != value for key, value in required.items()):
    raise SystemExit("frozen-ilxyr executor preflight incomplete: %r" % preflight)
PY
test -z "$(git -C "${LECORE_DIR}" status --porcelain --untracked-files=no)"

STAGE=model_materialization
write_status 0
"${PYTHON}" - "${MODEL_DIR}" "${MODEL_REVISION}" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen3.5-0.8B",
    revision=sys.argv[2],
    local_dir=sys.argv[1],
)
PY
echo "${WEIGHTS_SHA}  ${MODEL_DIR}/model.safetensors-00001-of-00001.safetensors" \
  | sha256sum -c -
echo "${TOKENIZER_SHA}  ${MODEL_DIR}/tokenizer.json" | sha256sum -c -

STAGE=real_model_benchmark
write_status 0
"${PYTHON}" "${LECORE_DIR}/tools/benchmark_qwen_runtime.py" \
  "${MODEL_DIR}" "${EVALUATION_CORPUS}" \
  --chunk-sizes 128,256,512 --tokens 512 --warmup-tokens 16 \
  --threads 8 --gdn-backend c --parity-atol 0.000001 \
  --ram-classes 32,64,128 --concurrent-runtimes 2 \
  --headroom-factor 1.25 --system-reserve-gib 2 \
  --full-run-peak-mib 19560.75 \
  --eval-tokens-per-runtime 4096 --eval-passes 2 \
  --fixed-overhead-minutes 45 \
  --ram-hourly-usd 32=0.2646,64=0.5292,128=1.0584 \
  --output "${BENCHMARK_REPORT}" \
  > "${RESULT_DIR}/benchmark.stdout" 2> "${RESULT_DIR}/benchmark.stderr"
BENCHMARK_SHA=$(sha256sum "${BENCHMARK_REPORT}" | cut -d' ' -f1)
echo "${BENCHMARK_SHA}  ${BENCHMARK_REPORT}" > "${RESULT_DIR}/benchmark-report.json.sha256"

STAGE=project_generation
write_status 0
"${PYTHON}" "${LECORE_DIR}/experiments/qwen35_acceptance/generate.py" \
  "${MODEL_DIR}" "${LECORE_DIR}/REFERENCE.md" "${EVALUATION_CORPUS}" \
  "${PROJECT_DIR}" --installed-dir "${INSTALLED_DIR}" --python "${PYTHON}" \
  --min-tokens 4096 --timeout-seconds 21600 --compute-credits 100 \
  --experiment-version 7 --initial-chunk-size 128 \
  --benchmark-report "${BENCHMARK_REPORT}" \
  --memory-budget-fraction 0.20 --evaluation-mode auto --gdn-backend c \
  --progress-upload-uri "s3://${BUCKET}/${PREFIX}/progress.jsonl" \
  --ilxyr-cli "${ILXYR}" > "${RESULT_DIR}/generation.json"

"${PYTHON}" - "${PROJECT_DIR}" "${RESULT_DIR}" "${LECORE_COMMIT}" \
  "${INSTALL_SHA}" "${EVALUATION_SHA}" "${WEIGHTS_SHA}" \
  "${TOKENIZER_SHA}" "${REQUIREMENTS_SHA}" "${LAUNCH_MANIFEST}" \
  "${BENCHMARK_REPORT}" "${BENCHMARK_SHA}" "${PYTHON}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

(project_dir, result_dir, source_commit, install_sha, evaluation_sha,
 weights_sha, tokenizer_sha, requirements_sha, launch_path, benchmark_path,
 benchmark_sha, selected_python) = sys.argv[1:]
project_dir, result_dir = Path(project_dir), Path(result_dir)
launch = json.loads(Path(launch_path).read_text())
benchmark = json.loads(Path(benchmark_path).read_text())
project = json.loads((project_dir / "project.json").read_text())
experiment = json.loads((project_dir / "experiment.json").read_text())
policy = project["runner_policy"]
policy_digest = hashlib.sha256(json.dumps(
    policy, sort_keys=True, separators=(",", ":"), ensure_ascii=True
).encode("ascii")).hexdigest()
files = {item["path"]: item for item in project["model_files"]}
benchmark_files = {item["name"]: item for item in benchmark["inputs"]["model_files"]}
recommendation = int(benchmark["speed_and_cost_projection"]["recommended_chunk_size"])
successful = [row for row in benchmark["results"] if row.get("status") == "ok"]
execution_args = experiment["execution"]["args"]

def execution_arg(flag):
    try:
        return execution_args[execution_args.index(flag) + 1]
    except (ValueError, IndexError):
        return None

def native_complete(row):
    report = (row.get("acceleration") or {}).get("full_sequence_gdn_recurrence", {})
    checks = report.get("validated_regimes") or []
    return (report.get("requested") == "c" and report.get("active") == "c" and
            report.get("refused") is None and report.get("native_calls", 0) > 0 and
            report.get("native_tokens", 0) > 0 and
            {bool(item.get("resumed")) for item in checks if item.get("passed")} == {False, True} and
            all(item.get("passed") for item in checks))

checks = {
    "source_commit": project["source_commit"] == source_commit,
    "installation_corpus": project["corpora"]["installation"]["sha256"] == install_sha,
    "evaluation_corpus": project["corpora"]["evaluation"]["sha256"] == evaluation_sha,
    "corpora_distinct": install_sha != evaluation_sha,
    "weights": files["model.safetensors-00001-of-00001.safetensors"]["sha256"] == weights_sha,
    "tokenizer": files["tokenizer.json"]["sha256"] == tokenizer_sha,
    "complete_model_manifest": len(files) >= 10,
    "experiment_version": project["experiment_version"] == 7 and project["experiment_id"].endswith(".v7.acceptance"),
    "policy_digest": project["runner_policy_digest"] == policy_digest,
    "closed_experiment_schema": "runner_policy" not in experiment and "runner_policy_digest" not in experiment,
    "policy_identity": policy_digest[:12] in project["experiment_id"],
    "checker_version": experiment["evidence_authority"]["provenance"]["checker"] == "checker://lecore/qwen35-acceptance/%s/v7" % policy_digest,
    "minimum_tokens": policy["accepted_requires_full_tokens"] == 4096 and policy["evaluation"]["accepted_requires_full_tokens"] == 4096,
    "full_go_only": policy["early_acceptance_allowed"] is False and policy["statistical_decision"]["early_acceptance_allowed"] is False,
    "sequential_policy": policy["sequential_looks"] == [1024, 2048, 3072, 4096] and policy["statistical_decision"]["sequential_resamples"] == 10000,
    "final_gate": policy["statistical_decision"]["final_interval"] == "paired_moving_block_bootstrap_two_sided_95pct" and policy["statistical_decision"]["maximum_perplexity_regression_ratio"] == 0.01,
    "reference_policy": policy["reference_parity"]["backend"] == "numpy" and policy["reference_parity"]["maximum_relative_logit_error"] == 0.001 and policy["reference_parity"]["must_pass_before_installation"] is True,
    "treatment_policy": policy["treatment"]["gdn_backend"] == "numpy" and policy["treatment"]["spectral_filtering_enabled"] is False and policy["treatment"]["weight_resident_metadata"] is False,
    "evaluation_policy": policy["evaluation"]["backend"] == "c" and policy["evaluation"]["common_chunk_schedule_required"] is True and policy["evaluation"]["mode"] == "auto",
    "native_acceptance_policy": policy["native_gdn_required_for_acceptance"] is True and policy["evaluation"]["native_gdn_required_for_acceptance"] is True and "--require-native-gdn" in execution_args,
    "official_policy": policy["official_compatibility"]["dependency_versions"] == {"Pillow":"12.3.0","torch":"2.11.0","torchvision":"0.26.0","transformers":"5.14.0"} and policy["official_compatibility"]["environment_lock_sha256"] == requirements_sha,
    "official_outputs": policy["official_compatibility"]["reload_requires_no_state_dict_incompatibilities"] is True and policy["official_compatibility"]["text_generation_required"] is True and policy["official_compatibility"]["vision_input_generation_required"] is True,
    "timeout_and_funding": policy["execution_limits"] == {"timeout_seconds":21600,"maximum_compute_credits":100},
    "venv_interpreter": experiment["execution"]["program"] == selected_python and selected_python.endswith("/venv/bin/python"),
    "execution_source_identity": execution_arg("--expected-source-commit") == source_commit,
    "execution_model_identity": execution_arg("--expected-model-digest") == project["model_digest"],
    "execution_installation_identity": execution_arg("--expected-installation-sha256") == install_sha,
    "execution_evaluation_identity": execution_arg("--expected-evaluation-sha256") == evaluation_sha,
    "benchmark_schema": benchmark["schema"] == "lecore.qwen_runtime_benchmark.v1",
    "benchmark_clean_source": benchmark["source"]["commit"] == source_commit and benchmark["source"]["tracked_checkout_dirty"] is False,
    "benchmark_corpus": benchmark["inputs"]["corpus"]["sha256"] == evaluation_sha,
    "benchmark_model": benchmark["inputs"]["weight_hashes_included"] is True and benchmark_files["model.safetensors-00001-of-00001.safetensors"]["sha256"] == weights_sha and benchmark_files["tokenizer.json"]["sha256"] == tokenizer_sha,
    "benchmark_trials": {int(row["chunk_size"]) for row in successful} == {128, 256, 512},
    "benchmark_parity": all(row.get("numerical_parity") for row in successful),
    "benchmark_native": all(native_complete(row) for row in successful),
    "benchmark_recommendation": recommendation in {128, 256, 512} and policy["max_chunk_size"] == recommendation,
    "benchmark_bound": policy["benchmark_report_sha256"] == benchmark_sha and policy["evaluation"]["benchmark_report_sha256"] == benchmark_sha,
    "benchmark_threads": benchmark["settings"]["threads_per_runtime"] == 8 and all(os.environ.get(name) == "8" for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS")),
    "progress_bound": policy["telemetry"]["progress_upload_uri"] == "s3://zero-training-022118847419/qwen35-acceptance/cb3b1d2-2fc06364-a6c9d113/v7-formal/progress.jsonl",
    "launch_source": launch["source"]["commit"] == source_commit,
    "launch_attempt": launch["attempt"]["experiment_version"] == 7 and launch["attempt"]["maximum_formal_runs"] == 1 and launch["attempt"]["retry_after_admission"] is False,
    "launch_instance": launch["aws"]["instance_type"] == "r7i.2xlarge" and launch["aws"]["memory_gib"] == 64 and launch["aws"]["hourly_compute_usd"] == 0.5292,
    "launch_ceiling": launch["aws"]["total_ceiling_usd"] == 10.0 and launch["aws"]["maximum_compute_usd"] <= 2.646 and launch["aws"]["prior_attempt_estimate_usd"] + launch["aws"]["maximum_compute_usd"] < 10.0,
}
if not all(checks.values()):
    raise SystemExit("admission preflight failed: %r" % {k:v for k,v in checks.items() if not v})
(result_dir / "model-manifest.json").write_text(json.dumps({
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "digest": project["model_digest"], "files": project["model_files"]
}, indent=2, sort_keys=True) + "\n")
(result_dir / "corpus-manifest.json").write_text(json.dumps(project["corpora"], indent=2, sort_keys=True) + "\n")
(result_dir / "admission-preflight.json").write_text(json.dumps({
    "checks": checks, "experiment_id": project["experiment_id"],
    "runner_policy_digest": policy_digest, "benchmark_report_sha256": benchmark_sha,
    "selected_chunk_size": recommendation
}, indent=2, sort_keys=True) + "\n")
print(project["experiment_id"])
PY
EXPERIMENT_ID=$("${PYTHON}" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["experiment_id"])' \
  "${PROJECT_DIR}/project.json")
write_status 0

capture_instance_identity
{
  uname -a
  lscpu
  free -m
  "${PYTHON}" --version
  "${PIP}" --version
  cc --version
  rustc --version
  cargo --version
  "${ILXYR}" --help
  env | LC_ALL=C sort | grep -E '^(OMP|OPENBLAS|MKL|NUMEXPR|LECORE|TOKENIZERS)'
} > "${RESULT_DIR}/environment.txt" 2>&1

STAGE=ilxyr_admission
write_status 0
"${ILXYR}" init "${WORKSPACE_DIR}" > "${RESULT_DIR}/ilxyr-init.json"
for name in hypothesis foundation engineering-review experiment-design; do
  "${ILXYR}" contribute "${WORKSPACE_DIR}" "${PROJECT_DIR}/${name}.json" \
    > "${RESULT_DIR}/ilxyr-contribute-${name}.json"
done
"${ILXYR}" compile "${WORKSPACE_DIR}" "${PROJECT_DIR}/experiment.json" \
  > "${RESULT_DIR}/ilxyr-compile.json"
"${ILXYR}" forecast "${WORKSPACE_DIR}" "${PROJECT_DIR}/forecast-empirical.json" \
  > "${RESULT_DIR}/ilxyr-forecast-empirical.json"
"${ILXYR}" forecast "${WORKSPACE_DIR}" "${PROJECT_DIR}/forecast-mechanistic.json" \
  > "${RESULT_DIR}/ilxyr-forecast-mechanistic.json"
"${ILXYR}" fund "${WORKSPACE_DIR}" "${PROJECT_DIR}/funding.json" \
  > "${RESULT_DIR}/ilxyr-funding.json"
"${ILXYR}" admit "${WORKSPACE_DIR}" "${EXPERIMENT_ID}" \
  > "${RESULT_DIR}/ilxyr-admission.json"
"${PYTHON}" - "${RESULT_DIR}/ilxyr-admission.json" <<'PY'
import json, sys
admission = json.load(open(sys.argv[1]))
if not admission.get("accepted") or not all(row.get("passed") for row in admission.get("checks", [])):
    raise SystemExit("ilxyr admission did not pass every gate")
PY

STAGE=formal_run
FORMAL_RUN_STARTED=true
write_status 0
set +e
timeout --signal=TERM --kill-after=60 21900 \
  "${ILXYR}" run "${WORKSPACE_DIR}" "${EXPERIMENT_ID}" --execute \
  > "${RESULT_DIR}/ilxyr-run.json" 2> "${RESULT_DIR}/ilxyr-run.stderr"
RUN_EXIT=$?
set -e

STAGE=verification
write_status "${RUN_EXIT}"
set +e
"${ILXYR}" status "${WORKSPACE_DIR}" "${EXPERIMENT_ID}" \
  > "${RESULT_DIR}/ilxyr-status.json" 2> "${RESULT_DIR}/ilxyr-status.stderr"
STATUS_EXIT=$?
"${ILXYR}" verify "${WORKSPACE_DIR}" \
  > "${RESULT_DIR}/ilxyr-verify.json" 2> "${RESULT_DIR}/ilxyr-verify.stderr"
VERIFY_EXIT=$?
set -e

"${PYTHON}" - "${RESULT_DIR}" "${EXPERIMENT_ID}" "${RUN_EXIT}" \
  "${STATUS_EXIT}" "${VERIFY_EXIT}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

result_dir = Path(sys.argv[1])
payload = {
    "schema": "lecore.qwen35-execution-summary.v7",
    "experiment_id": sys.argv[2],
    "run_exit_code": int(sys.argv[3]),
    "status_exit_code": int(sys.argv[4]),
    "verify_exit_code": int(sys.argv[5]),
    "completed_at": datetime.now(timezone.utc).isoformat(),
}
(result_dir / "execution-summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

if [ "${RUN_EXIT}" -ne 0 ] || [ "${STATUS_EXIT}" -ne 0 ] || \
   [ "${VERIFY_EXIT}" -ne 0 ]; then
  STAGE=execution_failure
  write_status 1
  collect_bundle
  exit 1
fi

STAGE=complete
write_status 0
collect_bundle
exit 0
