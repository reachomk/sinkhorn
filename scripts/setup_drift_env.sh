#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${ENV_NAME:-drift}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
REQ_FILE="${ROOT_DIR}/envs/requirements-drift.txt"

find_conda() {
  for c in "${HOME}/miniconda3/bin/conda" "${HOME}/anaconda3/bin/conda" "/opt/miniconda3/bin/conda"; do
    if [[ -x "${c}" ]]; then
      echo "${c}"
      return 0
    fi
  done
  if command -v conda >/dev/null 2>&1; then
    command -v conda
    return 0
  fi
  return 1
}

CONDA_BIN="${CONDA_BIN:-$(find_conda || true)}"
if [[ -z "${CONDA_BIN}" ]]; then
  echo "[ERROR] conda not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "[ERROR] requirements file not found: ${REQ_FILE}" >&2
  exit 1
fi

echo "[INFO] Using conda: ${CONDA_BIN}"
echo "[INFO] Target env: ${ENV_NAME} (python ${PYTHON_VERSION})"

if "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Env exists. Ensuring base packages..."
  "${CONDA_BIN}" install -n "${ENV_NAME}" -y "python=${PYTHON_VERSION}" pip ipykernel
else
  echo "[INFO] Creating env..."
  "${CONDA_BIN}" create -n "${ENV_NAME}" -y "python=${PYTHON_VERSION}" pip ipykernel
fi

echo "[INFO] Installing pip dependencies..."
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install --upgrade pip
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m pip install -r "${REQ_FILE}"

echo "[INFO] Verifying key imports..."
"${CONDA_BIN}" run -n "${ENV_NAME}" python - <<'PY'
import importlib
import sys

mods = [
    "torch",
    "torchvision",
    "numpy",
    "PIL",
    "tqdm",
    "yaml",
    "einops",
    "ot",
    "diffusers",
    "cleanfid",
]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as exc:
        missing.append((m, str(exc)))

if missing:
    print("[ERROR] Some imports failed:")
    for m, e in missing:
        print(f"  - {m}: {e}")
    sys.exit(1)

print("[OK] Environment is ready.")
PY

echo "[DONE] Setup complete."
echo "[NEXT] Activate: conda activate ${ENV_NAME}"
