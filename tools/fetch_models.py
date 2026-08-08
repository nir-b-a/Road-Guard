"""
Road Guard - model fetcher / verifier.

One command that makes a fresh clone runnable: checks every model the pipeline
loads, downloads the ones that are auto-downloadable, and verifies the ones that
ship in the repo against a known SHA-256.

    python tools/fetch_models.py            # verify, and download whatever is missing
    python tools/fetch_models.py --check    # verify only, never touch the network
    python tools/fetch_models.py --list     # print the model inventory and exit

Exit code is 0 when every REQUIRED model is present and valid, 1 otherwise - so
this doubles as a pre-demo smoke check and as a CI gate.

Why the models fall into three groups
-------------------------------------
* ``repo``        - ours, fine-tuned in-house, small enough to commit. They arrive
                    with the clone; we only verify the bytes weren't corrupted (git
                    LFS mishaps, half-finished copies onto a demo laptop).
* ``ultralytics`` - third-party COCO-pretrained weights that Ultralytics downloads
                    on first use. Too large to commit (~109 MB) and freely fetchable.
* ``fast_alpr``   - third-party plate detector + OCR (ONNX) that fast_alpr pulls into
                    its own cache on first construction.

For the two third-party groups a checksum mismatch is a WARNING, not an error: the
upstream release can legitimately be re-cut, and we would rather run than block. For
our own committed weights a mismatch is a hard FAIL - those bytes should never change.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Model:
    key: str
    path: str                 # relative to repo root
    source: str               # "repo" | "ultralytics" | "fast_alpr"
    role: str
    origin: str               # who trained it - feeds the architecture disclosure
    sha256: str = ""
    size: int = 0
    required: bool = True     # False => optional, absence is reported but not fatal
    notes: str = ""


MODELS: list[Model] = [
    # ---- ours, committed -----------------------------------------------------
    Model(
        key="lane_seg",
        path="weights/phase3_v3_yellowprotect.pt",
        source="repo",
        role="YOLOv8-seg lane segmentation (solid / yellow / dashed / road)",
        origin="Road Guard - fine-tuned in-house on a Roboflow-curated dataset",
        sha256="2dd566be78129e9d6b757d5dd9aa7c1c5bc1b995deba835dc6790554334db84e",
        size=6829037,
    ),
    Model(
        key="tire",
        path="models/tire_yolo11n.pt",
        source="repo",
        role="YOLO11n tire detector - Stage-2 solid-line cascade",
        origin="Road Guard - fine-tuned in-house (YOLO11n base)",
        sha256="8e358bcf317f14c56cb33199926ca07f66484f421df2aa4d8cff84db371271a3",
        size=5454419,
        notes="Without it the Stage-2 cascade degrades to Stage-1 only (more false positives).",
    ),
    Model(
        key="plate_detector_israeli",
        path="israeli_plates.pt",
        source="repo",
        role="Israeli licence-plate locator (PaddleOCRDetectorReader path)",
        origin="Road Guard - fine-tuned in-house",
        sha256="67baef8b225f2477d52d3e78ac1e608a752985d1ea45456dad644d449cc7750b",
        size=6250090,
        required=False,
        notes="Not on the live path: FastALPR is the active reader (lpr/reader.py).",
    ),

    # ---- third-party, auto-downloaded ---------------------------------------
    Model(
        key="yolo11x",
        path="yolo11x.pt",
        source="ultralytics",
        role="YOLO11-X detector - vehicles and traffic lights (main tracker)",
        origin="EXTERNAL - Ultralytics, COCO-pretrained (AGPL-3.0)",
        sha256="7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24",
        size=114636239,
    ),
]

# fast_alpr resolves these itself into its own cache; there is no stable path we can
# stat, so they are verified by constructing the reader rather than by checksum.
FAST_ALPR_MODELS = [
    ("yolo-v9-t-384-license-plate-end2end", "YOLOv9-t plate detector (ONNX)"),
    ("global-plates-mobile-vit-v2-model", "MobileViT-v2 plate OCR (ONNX)"),
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"
if os.name == "nt" and not os.environ.get("WT_SESSION"):
    # Old conhost sessions render the escapes literally; drop colour there.
    GREEN = RED = YELLOW = DIM = RESET = ""

OK, FAIL, WARN, SKIP = f"{GREEN}OK{RESET}", f"{RED}FAIL{RESET}", f"{YELLOW}WARN{RESET}", f"{DIM}--{RESET}"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 so a 109 MB weight doesn't land in memory all at once."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify(model: Model) -> tuple[str, str]:
    """Return (status, message) for an on-disk model."""
    full = REPO / model.path
    if not full.exists():
        return "missing", "not on disk"

    actual_size = full.stat().st_size
    if model.size and actual_size != model.size:
        return "bad", f"size {actual_size:,} != expected {model.size:,}"

    if model.sha256:
        actual = sha256_of(full)
        if actual != model.sha256:
            return "bad", f"sha256 {actual[:12]}... != expected {model.sha256[:12]}..."

    return "ok", f"{actual_size / 1e6:.1f} MB verified"


def download_ultralytics(model: Model) -> tuple[bool, str]:
    """Ultralytics fetches a known weight name into the CWD on first construction."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return False, "ultralytics not installed - run: pip install -r requirements.txt"

    cwd = os.getcwd()
    try:
        # Ultralytics downloads relative to the CWD, so pin it to the repo root and
        # the weight lands exactly where Constants.YOLO_VERSION expects it.
        os.chdir(REPO)
        YOLO(Path(model.path).name)
        return True, "downloaded"
    except Exception as exc:
        return False, f"download failed: {exc}"
    finally:
        os.chdir(cwd)


def download_fast_alpr() -> tuple[bool, str]:
    """Constructing ALPR pulls both ONNX models into fast_alpr's cache."""
    try:
        from fast_alpr import ALPR
    except ImportError:
        return False, "fast_alpr not installed (optional - LPR path only)"
    try:
        # Same ocr_model as lpr/reader.py, so we warm exactly the cache it will use.
        ALPR(ocr_model="global-plates-mobile-vit-v2-model")
        return True, "detector + OCR cached"
    except Exception as exc:
        return False, f"could not initialise: {exc}"


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_list() -> int:
    print(f"\n{'MODEL':<26} {'SOURCE':<12} {'REQ':<5} ROLE")
    print("-" * 100)
    for m in MODELS:
        print(f"{m.key:<26} {m.source:<12} {'yes' if m.required else 'no':<5} {m.role}")
        print(f"{'':<26} {DIM}{m.origin}{RESET}")
    for name, role in FAST_ALPR_MODELS:
        print(f"{name:<26} {'fast_alpr':<12} {'no':<5} {role}")
        print(f"{'':<26} {DIM}EXTERNAL - auto-downloaded by fast_alpr on first use{RESET}")
    print()
    return 0


def run(check_only: bool) -> int:
    print(f"\nRoad Guard model check  {DIM}({REPO}){RESET}\n")
    failures: list[str] = []
    warnings: list[str] = []

    for m in MODELS:
        status, msg = verify(m)

        if status == "missing" and not check_only and m.source == "ultralytics":
            ok, dl_msg = download_ultralytics(m)
            if ok:
                status, msg = verify(m)
                if status == "missing":
                    status, msg = "bad", "still absent after download"
            else:
                status, msg = "missing", dl_msg

        # Decide severity. Our own weights are strict; third-party ones are advisory.
        if status == "ok":
            mark = OK
        elif status == "bad" and m.source != "repo":
            mark, msg = WARN, f"{msg} (upstream weight may have been re-cut)"
            warnings.append(m.key)
        elif not m.required:
            mark = SKIP
            warnings.append(m.key)
        else:
            mark = FAIL
            failures.append(f"{m.key}: {msg}")

        print(f"  [{mark}] {m.key:<24} {m.path:<38} {msg}")
        if status != "ok" and m.notes:
            print(f"        {DIM}{m.notes}{RESET}")

    # fast_alpr is optional: the pipeline runs without plates, just with less evidence.
    if check_only:
        try:
            import fast_alpr  # noqa: F401
            print(f"  [{OK}] {'fast_alpr':<24} {'(cache)':<38} package importable")
        except ImportError:
            print(f"  [{SKIP}] {'fast_alpr':<24} {'(cache)':<38} not installed (optional)")
    else:
        ok, msg = download_fast_alpr()
        print(f"  [{OK if ok else SKIP}] {'fast_alpr':<24} {'(cache)':<38} {msg}")

    print()
    if failures:
        print(f"{RED}FAILED{RESET} - {len(failures)} required model(s) unusable:")
        for f in failures:
            print(f"  - {f}")
        print("\nOur own weights ship with the repo; if one is missing or corrupt, "
              "re-clone or `git checkout -- <path>`.")
        return 1

    if warnings:
        print(f"{GREEN}All required models present.{RESET} "
              f"{len(warnings)} optional/advisory item(s) noted above.")
    else:
        print(f"{GREEN}All models present and verified.{RESET}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch and verify Road Guard model weights.")
    ap.add_argument("--check", action="store_true",
                    help="verify only; never download (offline / CI / pre-demo check)")
    ap.add_argument("--list", action="store_true",
                    help="print the model inventory (source + origin) and exit")
    args = ap.parse_args()

    if args.list:
        return cmd_list()
    return run(check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
