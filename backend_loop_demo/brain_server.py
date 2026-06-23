"""
brain_server -- the brain as an ASYNC SERVICE the backend dispatches jobs to (backend-initiated).

This is the inverse of run_brain.py's client mode: instead of a human running the brain, a BACKEND
server hands it work. Flow (async, 202 + callback):

    backend  --POST /jobs {video_name, source_base_url, callback_url, job_id}-->  brain  (202)
    brain    --GET video + meta (pull, fail-fast integrity)-------------------->  backend
    brain    ...analyse -> annotate -> bundle (run_brain.run_once)...
    brain    --POST callback_url (the evidence bundle, in the folder format)--->  backend

The brain returns 202 immediately and does the (minutes-long, on a GPU) work in a background thread,
so the backend never blocks. Job state is queryable at GET /jobs/{job_id}.

Run (its own terminal)::

    python backend_loop_demo/brain_server.py --port 9000
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
import threading
import traceback
import uuid
from typing import Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend_loop_demo import run_brain


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


class JobRequest(BaseModel):
    """What the backend POSTs to dispatch one analysis job."""
    video_name: str
    source_base_url: str            # where the brain PULLS the video from (the backend)
    callback_url: str               # where the brain DELIVERS the finished bundle (backend ingest)
    job_id: Optional[str] = None
    top_k_crossings: int = 3
    with_synthetic: bool = True
    annotate: bool = True


def create_app() -> FastAPI:
    app = FastAPI(title="Road Guard Brain Service", version="1.0")
    jobs: dict = {}
    lock = threading.Lock()

    def _set(job_id: str, **fields) -> None:
        with lock:
            jobs.setdefault(job_id, {"job_id": job_id}).update(fields)

    def _process(job_id: str, req: JobRequest) -> None:
        _set(job_id, status="running", started_utc=_now())
        try:
            out = run_brain.run_once(
                req.source_base_url, video_name=req.video_name,
                top_k_crossings=req.top_k_crossings, with_synthetic=req.with_synthetic,
                annotate=req.annotate, ingest_url=req.callback_url, job_id=job_id)
            _set(job_id, status="done", finished_utc=_now(), delivered=out["ok"],
                 callback_status=out["status_code"], bundle_bytes=out["bundle_bytes"],
                 violation_count=out["manifest"].get("violation_count"), receipt=out["receipt"])
        except Exception as exc:  # noqa: BLE001 -- record the failure on the job, don't crash the server
            _set(job_id, status="error", finished_utc=_now(),
                 error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())

    @app.get("/health")
    def health() -> dict:
        with lock:
            return {"ok": True, "service": "brain", "jobs": len(jobs)}

    @app.post("/jobs", status_code=202)
    def submit_job(req: JobRequest) -> dict:
        """Accept a job, start processing in the background, return 202 immediately."""
        job_id = req.job_id or uuid.uuid4().hex[:12]
        _set(job_id, status="accepted", accepted_utc=_now(), video=req.video_name,
             source_base_url=req.source_base_url, callback_url=req.callback_url)
        threading.Thread(target=_process, args=(job_id, req), daemon=True).start()
        return {"job_id": job_id, "status": "accepted"}

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        with lock:
            if job_id not in jobs:
                raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
            return dict(jobs[job_id])

    @app.get("/jobs")
    def list_jobs() -> dict:
        with lock:
            return {"jobs": [dict(v) for v in jobs.values()]}

    return app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Road Guard brain service (async job processor).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    args = ap.parse_args(argv)

    import uvicorn
    print(f"Brain service listening on http://{args.host}:{args.port}  (POST /jobs to dispatch)",
          flush=True)
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
