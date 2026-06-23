"""
mock_backend -- a stand-in for the real Road Guard backend, so the full client<->server loop can be
proven end-to-end on one machine (no cloud, no DB).

It plays BOTH sides of the contract the backend owns:
  GET  /health                 -> liveness + what it is serving / where it saves
  GET  /video/{name}           -> serve the raw source video (what the brain pulls to analyse)
  GET  /video/{name}/meta      -> the AUTHORITATIVE reference VideoMeta fingerprint (ffprobe), so the
                                  app can fail-fast verify its download (violations.ingest_client)
  POST /ingest                 -> receive the brain's evidence bundle (.tar.gz, multipart field
                                  "bundle"), save it to the output folder, and write a JSON receipt
                                  summarising the manifest it found inside.

Run it (Terminal 1)::

    python backend_loop_demo/mock_backend.py \
        --video ~/Desktop/DeNnDugXxP0.mp4 \
        --out-dir ~/Desktop/backend_received_evidence \
        --port 8000

Then drive the brain against it from Terminal 2 (see run_brain.py / README.md).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import io
import json
import os
import sys
import tarfile
import uuid
from pathlib import Path
from typing import Optional

import requests

# Allow `python backend_loop_demo/mock_backend.py` from anywhere: put repo root on sys.path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from violations.video_integrity import probe_video


class DispatchRequest(BaseModel):
    """Operator -> backend: 'analyse this video on that brain'. The backend then dispatches the job."""
    brain_url: str                         # the brain service, e.g. http://127.0.0.1:9000
    video: Optional[str] = None            # defaults to the served video
    top_k_crossings: int = 3
    with_synthetic: bool = True
    annotate: bool = True

DEFAULT_VIDEO = str(Path.home() / "Desktop" / "DeNnDugXxP0.mp4")
DEFAULT_OUT_DIR = str(Path.home() / "Desktop" / "backend_received_evidence")


def _summarize_bundle(targz_bytes: bytes) -> dict:
    """Peek inside the received .tar.gz, read manifest.json, return a friendly receipt summary
    (violation count + the ranked queue order the brain asked for). Best-effort: never raises."""
    try:
        with tarfile.open(fileobj=io.BytesIO(targz_bytes), mode="r:gz") as tar:
            names = tar.getnames()
            mf = tar.extractfile("manifest.json")
            manifest = json.loads(mf.read().decode("utf-8")) if mf else {}
    except Exception as exc:  # noqa: BLE001 -- a receipt is best-effort, the bytes are already saved
        return {"manifest_ok": False, "error": f"{type(exc).__name__}: {exc}"}

    queue = [
        {
            "rank": i,
            "violation": v.get("violation"),
            "tier": v.get("tier"),
            "vehicle_id": v.get("vehicle_id"),
            "detector_confidence": v.get("detector_confidence"),
            "crossing_solid_line_confidence": v.get("crossing_solid_line_confidence"),
            "plate": v.get("plate"),
            "plate_score": v.get("plate_score"),
            "has_clip": bool((v.get("evidence") or {}).get("clip")),
            "n_crops": len((v.get("evidence") or {}).get("crops") or []),
            "has_report_docx": bool((v.get("evidence") or {}).get("report")),
        }
        for i, v in enumerate(manifest.get("violations", []))
    ]
    return {
        "manifest_ok": True,
        "manifest_version": manifest.get("manifest_version"),
        "priority_model": (manifest.get("priority_model") or {}).get("model"),
        "violation_count": manifest.get("violation_count"),
        "n_files_in_bundle": len(names),
        "queue": queue,
    }


def _safe_extract(targz_bytes: bytes, dest_dir: Path) -> None:
    """Extract the received .tar.gz into ``dest_dir`` (best-effort; guards against path traversal)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(targz_bytes), mode="r:gz") as tar:
            try:
                tar.extractall(dest_dir, filter="data")     # py3.12+: refuse unsafe members
            except TypeError:
                tar.extractall(dest_dir)
    except Exception as exc:  # noqa: BLE001 -- extraction is a convenience, bytes are already saved
        print(f"[ingest] WARN could not unpack bundle: {type(exc).__name__}: {exc}", flush=True)


def create_app(video_path: str = DEFAULT_VIDEO, out_dir: str = DEFAULT_OUT_DIR) -> FastAPI:
    app = FastAPI(title="Road Guard Mock Backend", version="1.0")
    video = Path(video_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _resolve(name: str) -> Path:
        # Serve the configured video by its own name; also allow siblings in the same folder.
        p = video if name == video.name else (video.parent / name)
        if not p.exists() or not p.is_file():
            raise HTTPException(status_code=404, detail=f"no such video: {name}")
        return p

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "serving": video.name, "video_exists": video.exists(),
                "out_dir": str(out)}

    @app.get("/video/{name}/meta")
    def video_meta(name: str) -> dict:
        return probe_video(str(_resolve(name))).to_dict()

    @app.get("/video/{name}")
    def get_video(name: str):
        return FileResponse(str(_resolve(name)), media_type="video/mp4", filename=name)

    @app.post("/dispatch")
    def dispatch(body: DispatchRequest, request: Request) -> dict:
        """Backend-initiated entry point: hand a job to the brain. The backend tells the brain where
        to PULL the video from (itself) and where to DELIVER the bundle (its own /ingest)."""
        base = str(request.base_url).rstrip("/")
        job = {
            "video_name": body.video or video.name,
            "source_base_url": base,
            "callback_url": base + "/ingest",
            "job_id": uuid.uuid4().hex[:12],
            "top_k_crossings": body.top_k_crossings,
            "with_synthetic": body.with_synthetic,
            "annotate": body.annotate,
        }
        resp = requests.post(body.brain_url.rstrip("/") + "/jobs", json=job, timeout=30)
        print(f"[dispatch] job {job['job_id']} -> brain {body.brain_url} ({resp.status_code})",
              flush=True)
        return {"dispatched": True, "brain_status": resp.status_code, "job": job,
                "brain_response": resp.json() if resp.content else None}

    @app.post("/ingest")
    async def ingest(bundle: UploadFile = File(...),
                     job_id: str = Form(None), source_video: str = Form(None)):
        data = await bundle.read()
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        stem = f"bundle_{ts}"
        dest = out / f"{stem}.tar.gz"
        dest.write_bytes(data)
        # Unpack into a browsable per-violation folder tree (one folder per vehicle/violation).
        unpacked = out / f"{stem}_unpacked"
        _safe_extract(data, unpacked)
        receipt = {
            "ok": True,
            "saved": str(dest),
            "unpacked": str(unpacked),
            "filename": bundle.filename,
            "job_id": job_id,                 # correlates this callback to the dispatched job
            "source_video": source_video,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "received_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            **_summarize_bundle(data),
        }
        (out / f"{stem}.receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        # Friendly server-side log so the operator watching Terminal 1 sees it land.
        print(f"[ingest] saved {dest.name} ({len(data):,} bytes) -- "
              f"{receipt.get('violation_count')} violations, "
              f"{receipt.get('n_files_in_bundle')} files; unpacked -> {unpacked.name}", flush=True)
        return JSONResponse(receipt)

    return app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Road Guard mock backend (serves a video, receives bundles).")
    ap.add_argument("--video", default=DEFAULT_VIDEO, help="raw source video to serve")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="where received bundles are saved")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)

    if not Path(args.video).exists():
        print(f"WARNING: video not found: {args.video} (the /video endpoints will 404)", flush=True)

    import uvicorn
    app = create_app(args.video, args.out_dir)
    print(f"Mock backend serving '{Path(args.video).name}' on http://{args.host}:{args.port}\n"
          f"  receiving bundles into: {args.out_dir}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
