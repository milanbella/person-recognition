#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-./.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
    echo "Python executable not found: $PYTHON" >&2
    echo "Create the virtual environment first or set PYTHON=/path/to/python." >&2
    exit 1
fi

"$PYTHON" -m pip install -r requirements.txt

# InsightFace installs the CPU distribution named onnxruntime. CPU and GPU
# distributions overwrite the same Python module, so install GPU last and alone.
"$PYTHON" -m pip uninstall -y onnxruntime onnxruntime-gpu
"$PYTHON" -m pip install --no-cache-dir 'onnxruntime-gpu[cuda,cudnn]==1.26.0'

"$PYTHON" - <<'PY'
import onnxruntime as ort

ort.preload_dlls(directory="")
providers = ort.get_available_providers()
print("ONNX Runtime providers:", providers)
if "CUDAExecutionProvider" not in providers:
    raise SystemExit("CUDAExecutionProvider is unavailable; GPU installation failed.")
PY

echo "GPU runtime installation verified."
