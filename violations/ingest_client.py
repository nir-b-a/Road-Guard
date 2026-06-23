"""
ingest_client -- the APP-SIDE half of the backend handshake: pull a video, verify it survived the
download, then (after analysis) the brain POSTs the evidence bundle back (see violations.export).

Where the integrity boundary lives (the architecture we agreed on):
  * The BACKEND owns the authoritative reference fingerprint -- it holds the original video, so it
    computes the reference ``VideoMeta`` (via violations.video_integrity.probe_video) and exposes it.
  * The APP does a FAIL-FAST check after downloading: re-probe the local copy and compare against the
    backend's reference. This catches a transfer that lost fidelity (truncation, a re-encoding proxy)
    BEFORE the GPU burns minutes analysing a corrupted clip, and it localises the fault to this hop.
    It is NOT the final correctness check -- the backend re-verifies the returned evidence itself.

Pure stdlib + an injectable HTTP ``session`` (a ``requests.Session`` in production, a fake in tests)
and an injectable ``ffprobe`` runner, so the download/verify logic tests without a network or ffmpeg.
"""
from __future__ import annotations

import os
import shutil
from typing import Any, Callable, Optional

from violations.video_integrity import (VideoMeta, assert_hop, probe_video,
                                         _default_ffprobe_runner)


def _join(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def fetch_reference_meta(base_url: str, name: str, *, session: Any = None,
                         timeout: float = 30.0) -> VideoMeta:
    """GET the backend's reference ``VideoMeta`` for ``name`` (its authoritative fingerprint)."""
    if session is None:
        import requests
        session = requests
    url = _join(base_url, f"video/{name}/meta")
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return VideoMeta.from_dict(resp.json())


def download_video(base_url: str, name: str, dest_path: str, *, session: Any = None,
                   timeout: float = 300.0, chunk_size: int = 1 << 20) -> str:
    """Stream the raw video from the backend to ``dest_path`` (chunked -> flat memory). Returns the
    local path written."""
    if session is None:
        import requests
        session = requests
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)
    url = _join(base_url, f"video/{name}")
    resp = session.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        iterator = resp.iter_content(chunk_size=chunk_size) if hasattr(resp, "iter_content") \
            else [resp.content]
        for block in iterator:
            if block:
                f.write(block)
    return dest_path


def download_and_verify(base_url: str, name: str, dest_path: str, *, session: Any = None,
                        ffprobe_runner: Callable = _default_ffprobe_runner,
                        timeout: float = 300.0,
                        allow_recompress: bool = False) -> tuple[str, VideoMeta, VideoMeta]:
    """Download ``name`` then FAIL-FAST verify the local copy against the backend's reference.

    Steps:
      1. fetch the backend's reference ``VideoMeta``,
      2. stream the video to ``dest_path``,
      3. re-probe the local file and :func:`assert_hop` it against the reference (raises
         ``IntegrityError`` on resolution loss / frame drop / silent re-encode).

    A faithful transfer is byte-identical, so ``allow_recompress`` defaults to ``False`` (a sha
    mismatch == the download was tampered with / corrupted). Returns
    ``(local_path, reference_meta, local_meta)``.
    """
    reference = fetch_reference_meta(base_url, name, session=session, timeout=timeout)
    local_path = download_video(base_url, name, dest_path, session=session, timeout=timeout)
    local_meta = probe_video(local_path, compute_sha256=True, runner=ffprobe_runner)
    assert_hop(reference, local_meta, hop="backend->app download",
               allow_recompress=allow_recompress)
    return local_path, reference, local_meta


def copy_local_as_download(src_path: str, dest_path: str) -> str:
    """Test/demo convenience: simulate a download by copying a local file (skips HTTP)."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)
    shutil.copyfile(src_path, dest_path)
    return dest_path
