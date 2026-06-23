"""
End-to-end integration test for the backend loop, run AUTOMATICALLY in one process.

It spins the mock backend up on an ephemeral localhost port in a background thread, then drives the
brain (run_brain.run_once) against it: pull DeNnDugXxP0.mp4 over HTTP -> fail-fast integrity check ->
analyse (real cached crossings + synthetic speeding/yellow) -> cut clips + grab frames + render the
speeding .docx -> bundle -> POST back. Then it asserts the bundle LANDED and is correctly structured.

OPT-IN (heavy: real ffmpeg cuts on a real 90s video). It is SKIPPED unless you ask for it, so the
normal fast unit-test loop stays untouched::

    # bash
    ROADGUARD_RUN_INTEGRATION=1 pytest backend_loop_demo/test_pipeline_integration.py -s
    # PowerShell
    $env:ROADGUARD_RUN_INTEGRATION=1; pytest backend_loop_demo/test_pipeline_integration.py -s

Also auto-skips if the source video or ffmpeg/ffprobe are missing.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import socket
import tarfile
import threading
import time
from pathlib import Path

import pytest

_VIDEO = Path.home() / "Desktop" / "DeNnDugXxP0.mp4"
_HAVE_FFMPEG = shutil.which("ffmpeg") and shutil.which("ffprobe")

pytestmark = pytest.mark.skipif(
    not os.environ.get("ROADGUARD_RUN_INTEGRATION"),
    reason="opt-in heavy e2e test: set ROADGUARD_RUN_INTEGRATION=1 to run")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _Server:
    """Run uvicorn in a daemon thread so the test process can talk to it over real HTTP."""

    def __init__(self, app, host: str, port: int):
        import uvicorn
        self._uv = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
        self._thread = threading.Thread(target=self._uv.run, daemon=True)
        self.base_url = f"http://{host}:{port}"

    def start(self, timeout: float = 10.0) -> None:
        self._thread.start()
        import requests
        deadline = time.time() + timeout
        while time.time() < deadline:
            if getattr(self._uv, "started", False):
                try:
                    if requests.get(self.base_url + "/health", timeout=1).status_code == 200:
                        return
                except Exception:  # noqa: BLE001 -- server still binding
                    pass
            time.sleep(0.05)
        raise RuntimeError("mock backend did not become ready in time")

    def stop(self) -> None:
        self._uv.should_exit = True
        self._thread.join(timeout=5)


@pytest.fixture
def backend(tmp_path):
    if not (_VIDEO.exists() and _HAVE_FFMPEG):
        pytest.skip(f"need {_VIDEO} and ffmpeg/ffprobe on PATH")
    from backend_loop_demo.mock_backend import create_app
    out_dir = tmp_path / "backend_received_evidence"
    server = _Server(create_app(str(_VIDEO), str(out_dir)), "127.0.0.1", _free_port())
    server.start()
    try:
        yield server, out_dir
    finally:
        server.stop()


def test_full_loop_lands_a_correctly_structured_bundle(backend):
    server, out_dir = backend
    from backend_loop_demo.run_brain import run_once

    result = run_once(server.base_url, top_k_crossings=3, with_synthetic=True)

    # ---- the HTTP transaction succeeded and the bundle physically LANDED -------------------
    assert result["ok"] and result["status_code"] in (200, 201)
    saved = list(out_dir.glob("*.tar.gz"))
    assert len(saved) == 1, f"expected exactly one bundle on disk, found {saved}"
    receipt = result["receipt"]
    assert receipt["ok"] and Path(receipt["saved"]).exists()
    assert receipt["sha256"]  # server fingerprinted what it received

    # ---- the manifest is ordered by the HARD TYPE TIERS we designed ------------------------
    violations = result["manifest"]["violations"]
    tiers = [v["tier"] for v in violations]
    assert tiers == sorted(tiers), f"queue must be tier-ordered, got {tiers}"
    types = [v["violation"] for v in violations]
    # every crossing precedes the speeder, which precedes the yellow-line event
    assert types.index("SOLID_LINE_CROSSING") < types.index("SPEEDING") < types.index("YELLOW_LINE_RIGHT")

    crossings = [v for v in violations if v["violation"] == "SOLID_LINE_CROSSING"]
    assert len(crossings) == 3
    # within the crossing tier: confidence DESC
    confs = [v["detector_confidence"] for v in crossings]
    assert confs == sorted(confs, reverse=True)
    # crossing carries its crossing confidence; speeding does NOT (null)
    assert all(v["crossing_solid_line_confidence"] is not None for v in crossings)
    speeders = [v for v in violations if v["violation"] == "SPEEDING"]
    assert speeders and speeders[0]["crossing_solid_line_confidence"] is None

    # ---- the agreed payload fields are present on the speeding row -------------------------
    sp = speeders[0]
    for fld in ("vehicle_id", "violation", "plate", "plate_score", "n_reads", "manual_review",
                "crossing_solid_line_confidence"):
        assert fld in sp
    assert sp["plate"] == "40-418-79" and sp["n_reads"] == 6 and sp["manual_review"] is False

    # ---- every violation shipped: annotated clip + 3 vehicle pics (+ plate crops) + 2 documents -
    for v in violations:
        ev = v["evidence"]
        assert ev["clip"] is not None and ev["clip"]["bytes"] > 0
        assert ev["clip"]["recompressed"] is True       # annotated clip is a deliberate re-encode
        assert len(ev["crops"]) == 3
        assert all("plate" in c for c in ev["crops"])    # a plate crop for each vehicle picture
        assert v["description"]                          # human-readable description present
        assert v["record_file"].endswith("violation.json") and v["summary_file"].endswith(".txt")
    assert sp["evidence"]["report"] is not None          # speeding also gets the speed .docx
    assert sp["evidence"]["report"]["file"].endswith(".docx")

    # ---- crack the .tar.gz the backend stored: manifest + per-violation docs + plate pics + docx
    blob = saved[0].read_bytes()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        names = set(tar.getnames())
        assert "manifest.json" in names
        assert len([n for n in names if n.endswith("violation.json")]) == len(violations)
        assert len([n for n in names if n.endswith(".txt")]) == len(violations)   # a doc per folder
        assert [n for n in names if n.endswith("_plate.png")], "plate crops must be present"
        docx_names = [n for n in names if n.endswith("report.docx")]
        assert docx_names, "speeding .docx must be in the archive"
        docx_bytes = tar.extractfile(docx_names[0]).read()

    # ---- backend auto-unpacked the bundle into a browsable per-violation folder tree ----------
    assert Path(receipt["unpacked"]).is_dir()
    # a .docx is a zip whose first local-file header is 'PK' and which contains word/document.xml
    assert docx_bytes[:2] == b"PK"
    with zipfile_open(docx_bytes) as z:
        assert "word/document.xml" in z.namelist()


def test_backend_initiated_async_dispatch(backend):
    """The BACKEND-INITIATED direction: backend dispatches a job to the brain SERVICE; the brain
    pulls the video, analyses, and delivers the bundle back to the backend's /ingest (async)."""
    server, out_dir = backend
    import requests
    from backend_loop_demo.brain_server import create_app as create_brain

    brain = _Server(create_brain(), "127.0.0.1", _free_port())
    brain.start()
    try:
        # backend dispatches: "analyse the served video on this brain"
        r = requests.post(server.base_url + "/dispatch",
                          json={"brain_url": brain.base_url, "top_k_crossings": 2}, timeout=30)
        assert r.status_code == 200
        dispatched = r.json()
        assert dispatched["brain_status"] == 202        # brain accepted asynchronously
        job_id = dispatched["job"]["job_id"]

        # the brain works in the background, then calls back -> poll its job record until it finishes
        # ("done" guarantees the callback returned, so the backend's receipt is fully written by now)
        deadline = time.time() + 180
        status = {}
        while time.time() < deadline:
            status = requests.get(f"{brain.base_url}/jobs/{job_id}", timeout=5).json()
            if status.get("status") in ("done", "error"):
                break
            time.sleep(0.5)
        assert status.get("status") == "done", f"job did not finish cleanly: {status}"
        assert 200 <= status["callback_status"] < 300

        # the delivered bundle is on disk and correlated to the dispatched job
        assert list(out_dir.glob("*.tar.gz")), "no bundle delivered to the backend"
        receipt = json.loads(next(iter(out_dir.glob("*.receipt.json"))).read_text(encoding="utf-8"))
        assert receipt["ok"] and receipt["job_id"] == job_id
        assert receipt["violation_count"] >= 1
    finally:
        brain.stop()


def zipfile_open(data: bytes):
    import zipfile
    return zipfile.ZipFile(io.BytesIO(data))
