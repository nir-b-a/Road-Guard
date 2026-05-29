"""
Phase 0 environment smoke test for the Road Guard DL lane pipeline.

Run INSIDE the conda env, from the repo root:
    python tools/smoke_test_env.py

It validates the stack in dependency order and prints PASS/FAIL per stage. The
decisive stage is [6] - it compiles a trivial inline CUDA extension, which
exercises the exact toolchain UnLanedet's custom ops need (nvcc + MSVC + ninja).
If [6] passes, UnLanedet's ops will compile; if it fails, fix the toolchain
BEFORE cloning UnLanedet so you debug a 10-line kernel, not a whole framework.

Exit code is non-zero if any CRITICAL stage fails.
"""

from __future__ import annotations

import platform
import sys
from typing import Callable

# (name, ok, critical, detail)
_results: list[tuple[str, bool, bool, str]] = []


def _run(name: str, critical: bool, fn: Callable[[], str | None]) -> None:
    try:
        detail = fn() or "ok"
        _results.append((name, True, critical, detail))
        print(f"[PASS] {name}: {detail}")
    except Exception as exc:  # noqa: BLE001 - capture every failure, don't abort
        _results.append((name, False, critical, f"{type(exc).__name__}: {exc}"))
        print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")


# --- stages ------------------------------------------------------------------

def py_version() -> str:
    v = sys.version_info
    msg = f"{platform.python_version()} ({platform.system()})"
    if not (v.major == 3 and v.minor in (10, 11)):
        msg += "  <-- WARNING: 3.10/3.11 recommended for compiled extensions"
    return msg


def numpy_below_2() -> str:
    import numpy as np
    if int(np.__version__.split(".")[0]) >= 2:
        raise RuntimeError(f"numpy {np.__version__} >= 2.0 - pin numpy==1.26.4")
    return np.__version__


def torch_cuda_build() -> str:
    import torch
    if torch.version.cuda is None:
        raise RuntimeError("torch is CPU-only - reinstall from the cu118 index")
    return f"torch {torch.__version__}, cuda {torch.version.cuda}"


def cuda_device() -> str:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False - check driver/install")
    cap = torch.cuda.get_device_capability(0)
    free, total = torch.cuda.mem_get_info(0)
    gb = 1024 ** 3
    note = "  (Pascal: prefer fp32; fp16 gives no speedup)" if cap == (6, 1) else ""
    return (f"{torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}  "
            f"VRAM {free/gb:.1f}/{total/gb:.1f} GB free{note}")


def torchvision_nms() -> str:
    import torch
    import torchvision
    from torchvision.ops import nms
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=torch.float32, device="cuda")
    keep = nms(boxes, torch.tensor([0.9, 0.8], device="cuda"), 0.5)
    return f"torchvision {torchvision.__version__}, nms kept {keep.numel()} box(es)"


def yolo_forward() -> str:
    import numpy as np
    import ultralytics
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")  # downloads ~6MB on first run
    dummy = (np.random.rand(640, 640, 3) * 255).astype("uint8")
    model.predict(dummy, verbose=False, device=0)
    return f"ultralytics {ultralytics.__version__}, GPU inference OK"


def inline_cuda_compile() -> str:
    import torch
    from torch.utils.cpp_extension import load_inline

    cpp_src = "torch::Tensor add_one(torch::Tensor x);"
    cuda_src = r"""
    #include <torch/extension.h>
    __global__ void add_one_kernel(float* x, int n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) x[i] += 1.0f;
    }
    torch::Tensor add_one(torch::Tensor x) {
        auto y = x.clone();
        int n = y.numel();
        int threads = 256, blocks = (n + threads - 1) / threads;
        add_one_kernel<<<blocks, threads>>>(y.data_ptr<float>(), n);
        return y;
    }
    """
    ext = load_inline(
        name="rg_smoke_op",
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=["add_one"],
        verbose=False,
    )
    x = torch.ones(1024, device="cuda")
    if not torch.allclose(ext.add_one(x), x + 1):
        raise RuntimeError("kernel compiled but produced wrong result")
    return "nvcc + MSVC + ninja toolchain OK - UnLanedet ops should build"


def unlanedet_op() -> str:
    # Fill in once UnLanedet is cloned/installed and you know the op module path,
    # e.g.:  from unlanedet.ops import nms_impl  (then call it on a tiny input)
    raise NotImplementedError("not wired yet - do this after `pip install -e ./UnLanedet`")


def main() -> int:
    print("=" * 70)
    print("Road Guard - Phase 0 environment smoke test")
    print("=" * 70)
    _run("0. Python version", False, py_version)
    _run("1. numpy < 2.0", True, numpy_below_2)
    _run("2. torch + CUDA build", True, torch_cuda_build)
    _run("3. CUDA device visible", True, cuda_device)
    _run("4. torchvision CUDA op (nms)", True, torchvision_nms)
    _run("5. ultralytics YOLOv8 forward", True, yolo_forward)
    _run("6. Inline CUDA op compile (THE toolchain test)", True, inline_cuda_compile)
    _run("7. UnLanedet custom op import (optional)", False, unlanedet_op)

    print("-" * 70)
    critical_fail = [n for n, ok, crit, _ in _results if crit and not ok]
    if critical_fail:
        print(f"RESULT: {len(critical_fail)} CRITICAL stage(s) failed: {critical_fail}")
        print("Fix these before cloning UnLanedet.")
        return 1
    print("RESULT: all critical stages passed - clear to install UnLanedet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
