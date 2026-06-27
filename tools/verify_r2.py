"""
verify_r2 -- end-to-end self-test for the Cloudflare R2 integration.

Proves the full automatic upload/download flow with your REAL credentials:

    1. connect + read current bucket usage (enforces the 3 GB cap accounting)
    2. presign a GET and a PUT URL (what the backend hands the app / brain)
    3. upload a small test object   (brain -> R2)
    4. download it back             (R2 -> brain)  and verify the bytes match
    5. round-trip the SAME object through the presigned URLs (pure HTTP, no boto3)
    6. delete the test object       (cleanup; leaves the bucket as it was)

Run it from any network that is NOT blocking the R2 data-plane. (Some
corporate / campus / ISP networks SNI-filter `*.r2.cloudflarestorage.com`
storage subdomains -- if every step fails at the TLS handshake, that is the
cause, not your credentials. Run it from Colab / a cloud box / a phone
hotspot instead.)

Credentials are read from the environment (or a local .env). Nothing is
hardcoded here. See .env.example for the variable names.

    python tools/verify_r2.py
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request

# Make the repo importable when run as `python tools/verify_r2.py`
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from violations.r2_client import R2Client, R2Config, _load_dotenv, StorageCapacityError


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def main() -> int:
    _load_dotenv()
    try:
        cfg = R2Config.from_env()
    except EnvironmentError as e:
        print(e)
        return 2

    client = R2Client(cfg)
    test_key = f"_selftest/roadguard-verify-{int(time.time())}.bin"
    payload = b"road-guard r2 self-test " + os.urandom(64)

    print(f"R2 endpoint : {cfg.endpoint}")
    print(f"R2 bucket   : {cfg.bucket}")
    print(f"Cap         : {cfg.max_bytes / 1e9:.1f} GB")
    print("-" * 60)

    # 1) connect + usage ---------------------------------------------------
    try:
        used, cap = client.check_capacity(len(payload))
        _ok(f"connected; current usage {used / 1e6:.2f} MB / {cap / 1e9:.1f} GB")
    except StorageCapacityError as e:
        _fail(f"bucket already at/over cap: {e}")
        return 1
    except Exception as e:
        _fail(f"could not reach R2 data-plane: {type(e).__name__}: {e}")
        print("\n  If this is a TLS handshake failure, this network is blocking")
        print("  the R2 storage subdomain. Re-run from Colab / another network.")
        return 1

    # 2) presign -----------------------------------------------------------
    try:
        get_url = client.presign_get(test_key, expires_in=600)
        put_url = client.presign_put(test_key, expires_in=600,
                                     content_type="application/octet-stream")
        _ok("generated presigned GET + PUT URLs")
    except Exception as e:
        _fail(f"presign failed: {type(e).__name__}: {e}")
        return 1

    # 3) upload (boto3) ----------------------------------------------------
    try:
        client.upload_bytes(payload, test_key)
        _ok(f"uploaded {len(payload)} bytes -> {test_key}")
    except Exception as e:
        _fail(f"upload failed: {type(e).__name__}: {e}")
        return 1

    # 4) download (boto3) + verify ----------------------------------------
    dest = os.path.join(_REPO, "_r2_selftest_download.bin")
    try:
        client.download_file(test_key, dest)
        got = open(dest, "rb").read()
        os.remove(dest)
        if got == payload:
            _ok("downloaded via boto3 and bytes match")
        else:
            _fail("downloaded bytes do NOT match what was uploaded")
            return 1
    except Exception as e:
        _fail(f"download failed: {type(e).__name__}: {e}")
        return 1

    # 5) presigned-URL HTTP round-trip (what the app / brain actually use) --
    try:
        req = urllib.request.Request(get_url, method="GET")
        with urllib.request.urlopen(req, timeout=30) as r:
            via_url = r.read()
        if via_url == payload:
            _ok("fetched the object through the presigned GET URL (bytes match)")
        else:
            _fail("presigned-GET bytes do NOT match")
            return 1
    except Exception as e:
        _fail(f"presigned-GET fetch failed: {type(e).__name__}: {e}")
        return 1

    # 6) cleanup -----------------------------------------------------------
    try:
        client.delete_object(test_key)
        _ok("deleted the test object (bucket restored to original state)")
    except Exception as e:
        _fail(f"cleanup delete failed (object {test_key} may linger): {e}")
        return 1

    print("-" * 60)
    print("ALL CHECKS PASSED -- automatic upload + download to Cloudflare R2 works.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
