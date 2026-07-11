from __future__ import annotations

import threading
from importlib.metadata import PackageNotFoundError, version

import onnxruntime as ort


_prepare_lock = threading.Lock()
_prepared = False


def _installed_version(distribution_name: str) -> str | None:
    try:
        return version(distribution_name)
    except PackageNotFoundError:
        return None


def prepare_onnx_runtime() -> list[str]:
    """Preload pip-installed CUDA libraries and report provider conflicts."""
    global _prepared
    with _prepare_lock:
        if not _prepared:
            preload_dlls = getattr(ort, "preload_dlls", None)
            if preload_dlls is not None:
                try:
                    # Empty directory means NVIDIA CUDA/cuDNN packages in site-packages.
                    preload_dlls(directory="")
                except Exception as exc:
                    print(f"ONNX Runtime CUDA dependency preload failed: {exc}")
            _prepared = True

    providers = ort.get_available_providers()
    cpu_version = _installed_version("onnxruntime")
    gpu_version = _installed_version("onnxruntime-gpu")
    if cpu_version is not None and gpu_version is not None:
        print(
            "ONNX Runtime package conflict: both onnxruntime "
            f"{cpu_version} and onnxruntime-gpu {gpu_version} are installed. "
            "Run ./install_gpu_requirements.sh to restore the GPU runtime."
        )
    if "CUDAExecutionProvider" not in providers:
        print(f"ONNX Runtime available providers: {providers}")
    return providers
