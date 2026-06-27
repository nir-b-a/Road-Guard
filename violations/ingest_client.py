"""
ingest_client -- the BRAIN-SIDE half of the backend handshake: pull the source video from a
one-time presigned URL, verify it survived the download, then (after analysis) the brain uploads
the evidence bundle (see violations.export: presigned PUT + webhook notify, or legacy multipart).

Architecture (Cloudflare presigned-URL model -- the one we now build to)
--------------------------------------------------------------------------
The Node backend NO LONGER stores or serves the media bytes. Videos + evidence live in Cloudflare
(R2), and the backend hands the brain a JOB PAYLOAD describing one unit of work:

    {
      "job_id":      "abc123",
      "video_url":   "https://<r2-bucket>.../DeNnDugXxP0.mp4?X-Amz-Signature=...",   # OPAQUE, one-time
      "video_meta":  { ...VideoMeta dict... },     # reference fingerprint, from the DB (Mongo)
      "upload_url":  "https://<r2-bucket>.../bundles/abc123.tar.gz?X-Amz-Signature=...",  # presigned PUT
      "notify_url":  "https://<backend>/internal/jobs/abc123/complete"   # lightweight webhook
    }

So this module:
  * downloads from a FULL, OPAQUE URL -- it does NOT build "{base}/video/{name}" or "{base}/.../meta"
    paths any more. A presigned URL is a single signed string you cannot append to.
  * takes the reference ``VideoMeta`` from the JOB PAYLOAD (``video_meta``), not from a ``/meta``
    endpoint. The backend is no longer the byte source, so its DB record is the source of truth for
    the fingerprint (the app computes it at capture, or it comes from R2's checksum).
  * still FAIL-FAST verifies the downloaded copy against that reference (resolution / fps / frame
    count / sha256) BEFORE the GPU burns minutes on a corrupted clip, and localises the fault to the
    download hop. It is NOT the final correctness check -- the backend re-verifies the returned bundle.

Pure stdlib + an injectable HTTP ``session`` (a ``requests.Session`` in production, a fake in tests)
and an injectable ``ffprobe`` runner, so the download/verify logic tests without a network or ffmpeg.
"""
from __future__ import annotations

import os
import shutil
from typing import Any, Callable, Optional

from violations.video_integrity import (VideoMeta, assert_hop, probe_video,
                                         _default_ffprobe_runner)


def reference_from_job(job: dict) -> VideoMeta:
    """The reference ``VideoMeta`` carried in the job payload (the DB's stored fingerprint).

    Accepts either ``job["video_meta"]`` (preferred) or a bare VideoMeta dict. Raises if absent --
    a job with no reference fingerprint cannot be integrity-checked, which we treat as a hard error
    rather than silently skipping the guard rail.
    """
    meta = job.get("video_meta") if "video_meta" in job else job
    if not meta:
        raise ValueError("job payload has no 'video_meta' reference fingerprint to verify against")
    return VideoMeta.from_dict(meta)


def download_from_url(url: str, dest_path: str, *, session: Any = None,
                      timeout: float = 300.0, chunk_size: int = 1 << 20) -> str:
    """Stream a video from a FULL, OPAQUE ``url`` (e.g. a Cloudflare R2 presigned GET) to
    ``dest_path``, chunked so memory stays flat on multi-GB clips. Returns the local path written.

    ``url`` is used verbatim -- no path is appended to it (a presigned URL is a single signed
    string; appending would break the signature).
    """
    if session is None:
        import requests                       # lazy: keep module importable without requests
        session = requests
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)
    resp = session.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        iterator = resp.iter_content(chunk_size=chunk_size) if hasattr(resp, "iter_content") \
            else [resp.content]
        for block in iterator:
            if block:
                f.write(block)
    return dest_path


def verify_download(local_path: str, reference: VideoMeta, *,
                    ffprobe_runner: Callable = _default_ffprobe_runner,
                    allow_recompress: bool = False, hop: str = "presigned download") -> VideoMeta:
    """Re-probe the downloaded copy and FAIL-FAST it against ``reference`` (raises ``IntegrityError``
    on resolution loss / frame drop / silent re-encode). Returns the local ``VideoMeta``."""
    local_meta = probe_video(local_path, compute_sha256=True, runner=ffprobe_runner)
    assert_hop(reference, local_meta, hop=hop, allow_recompress=allow_recompress)
    return local_meta


def download_and_verify_from_url(video_url: str, dest_path: str, reference: VideoMeta, *,
                                 session: Any = None,
                                 ffprobe_runner: Callable = _default_ffprobe_runner,
                                 timeout: float = 300.0,
                                 allow_recompress: bool = False
                                 ) -> tuple[str, VideoMeta, VideoMeta]:
    """Download from an OPAQUE ``video_url`` then fail-fast verify it against a ``reference``
    fingerprint supplied by the caller (from the job payload / DB).

    A faithful transfer is byte-identical, so ``allow_recompress`` defaults to ``False`` (a sha
    mismatch == the download was tampered with / corrupted). Returns
    ``(local_path, reference_meta, local_meta)``.
    """
    if not isinstance(reference, VideoMeta):
        reference = VideoMeta.from_dict(reference)         # tolerate a raw dict
    local_path = download_from_url(video_url, dest_path, session=session, timeout=timeout)
    local_meta = verify_download(local_path, reference, ffprobe_runner=ffprobe_runner,
                                 allow_recompress=allow_recompress)
    return local_path, reference, local_meta


def download_and_verify_from_job(job: dict, dest_path: str, *, session: Any = None,
                                 ffprobe_runner: Callable = _default_ffprobe_runner,
                                 timeout: float = 300.0,
                                 allow_recompress: bool = False
                                 ) -> tuple[str, VideoMeta, VideoMeta]:
    """One-call pull+verify for a backend JOB PAYLOAD: read ``video_url`` + ``video_meta`` from the
    job, download from the presigned URL, and fail-fast verify. This is the entry point the live
    pipeline (main.py --job-payload) and the MODE-B brain service use. Returns
    ``(local_path, reference_meta, local_meta)``.
    """
    video_url = job.get("video_url")
    if not video_url:
        raise ValueError("job payload has no 'video_url' (presigned GET) to download from")
    reference = reference_from_job(job)
    return download_and_verify_from_url(video_url, dest_path, reference, session=session,
                                        ffprobe_runner=ffprobe_runner, timeout=timeout,
                                        allow_recompress=allow_recompress)


def copy_local_as_download(src_path: str, dest_path: str) -> str:
    """Test/demo convenience: simulate a download by copying a local file (skips HTTP)."""
    os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)
    shutil.copyfile(src_path, dest_path)
    return dest_path
