"""
Tests for violations/r2_client.py — fully offline, no real network calls.

All S3 operations are mocked via unittest.mock so these run on any machine
without boto3 credentials or a real R2 bucket.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch, call

# Make sure repo root is on the path
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from violations.r2_client import R2Config, R2Client, StorageCapacityError, _load_dotenv


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _fake_cfg(max_bytes=3 * 1024 ** 3) -> R2Config:
    return R2Config(
        access_key_id="FAKE_KEY",
        secret_access_key="FAKE_SECRET",
        endpoint="https://fake.r2.cloudflarestorage.com",
        bucket="test-bucket",
        max_bytes=max_bytes,
    )


def _client_with_mock_s3(max_bytes=3 * 1024 ** 3):
    """Return (R2Client, mock_s3) with a pre-wired mock boto3 client."""
    mock_s3 = MagicMock()
    with patch("violations.r2_client._b3") as mock_b3:
        mock_b3.return_value.client.return_value = mock_s3
        client = R2Client(_fake_cfg(max_bytes))
    client._s3 = mock_s3   # keep reference after patch exits
    return client, mock_s3


def _make_paginator(pages: list[list[dict]]):
    """Build a mock paginator that yields `pages` (each a list of S3 object dicts)."""
    mock_pag = MagicMock()
    mock_pag.paginate.return_value = [
        {"Contents": p} for p in pages if p
    ] + ([{"Contents": []}] if not pages else [])
    return mock_pag


# --------------------------------------------------------------------------- #
# R2Config tests
# --------------------------------------------------------------------------- #
class TestR2Config(unittest.TestCase):

    def test_from_env_raises_on_missing_vars(self):
        clean = {k: v for k, v in os.environ.items()
                 if not k.startswith("CF_R2_")}
        with patch.dict(os.environ, clean, clear=True):
            with self.assertRaises(EnvironmentError) as ctx:
                R2Config.from_env()
            self.assertIn("CF_R2_ACCESS_KEY_ID", str(ctx.exception))

    def test_from_env_reads_all_vars(self):
        env = {
            "CF_R2_ACCESS_KEY_ID": "KEY",
            "CF_R2_SECRET_ACCESS_KEY": "SECRET",
            "CF_R2_ENDPOINT": "https://acct.r2.cloudflarestorage.com",
            "CF_R2_BUCKET": "my-bucket",
            "CF_R2_MAX_BYTES": "1073741824",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = R2Config.from_env()
        self.assertEqual(cfg.access_key_id, "KEY")
        self.assertEqual(cfg.bucket, "my-bucket")
        self.assertEqual(cfg.max_bytes, 1073741824)

    def test_endpoint_trailing_slash_stripped(self):
        env = {
            "CF_R2_ACCESS_KEY_ID": "K",
            "CF_R2_SECRET_ACCESS_KEY": "S",
            "CF_R2_ENDPOINT": "https://acct.r2.cloudflarestorage.com/",
            "CF_R2_BUCKET": "b",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = R2Config.from_env()
        self.assertFalse(cfg.endpoint.endswith("/"))


# --------------------------------------------------------------------------- #
# bucket_size_bytes + check_capacity
# --------------------------------------------------------------------------- #
class TestStorageAccounting(unittest.TestCase):

    def test_bucket_size_sums_all_pages(self):
        client, mock_s3 = _client_with_mock_s3()
        pag = MagicMock()
        pag.paginate.return_value = [
            {"Contents": [{"Size": 100}, {"Size": 200}]},
            {"Contents": [{"Size": 300}]},
        ]
        mock_s3.get_paginator.return_value = pag
        self.assertEqual(client.bucket_size_bytes(), 600)

    def test_bucket_size_empty_bucket(self):
        client, mock_s3 = _client_with_mock_s3()
        pag = MagicMock()
        pag.paginate.return_value = [{}]   # no Contents key
        mock_s3.get_paginator.return_value = pag
        self.assertEqual(client.bucket_size_bytes(), 0)

    def test_check_capacity_ok(self):
        client, mock_s3 = _client_with_mock_s3(max_bytes=1000)
        pag = MagicMock()
        pag.paginate.return_value = [{"Contents": [{"Size": 400}]}]
        mock_s3.get_paginator.return_value = pag
        current, cap = client.check_capacity(upload_bytes=500)
        self.assertEqual(current, 400)
        self.assertEqual(cap, 1000)

    def test_check_capacity_raises_over_limit(self):
        client, mock_s3 = _client_with_mock_s3(max_bytes=1000)
        pag = MagicMock()
        pag.paginate.return_value = [{"Contents": [{"Size": 900}]}]
        mock_s3.get_paginator.return_value = pag
        with self.assertRaises(StorageCapacityError):
            client.check_capacity(upload_bytes=200)   # 900 + 200 = 1100 > 1000

    def test_check_capacity_exactly_at_limit_ok(self):
        client, mock_s3 = _client_with_mock_s3(max_bytes=1000)
        pag = MagicMock()
        pag.paginate.return_value = [{"Contents": [{"Size": 800}]}]
        mock_s3.get_paginator.return_value = pag
        current, cap = client.check_capacity(upload_bytes=200)   # 800 + 200 == 1000 exactly
        self.assertEqual(current, 800)


# --------------------------------------------------------------------------- #
# Presigned URL generation
# --------------------------------------------------------------------------- #
class TestPresignedUrls(unittest.TestCase):

    def test_presign_get_calls_correct_operation(self):
        client, mock_s3 = _client_with_mock_s3()
        mock_s3.generate_presigned_url.return_value = "https://fake/get-url"
        url = client.presign_get("videos/clip.mp4", expires_in=1800)
        mock_s3.generate_presigned_url.assert_called_once_with(
            "get_object",
            Params={"Bucket": "test-bucket", "Key": "videos/clip.mp4"},
            ExpiresIn=1800,
        )
        self.assertEqual(url, "https://fake/get-url")

    def test_presign_put_calls_correct_operation(self):
        client, mock_s3 = _client_with_mock_s3()
        mock_s3.generate_presigned_url.return_value = "https://fake/put-url"
        url = client.presign_put("bundles/job123.tar.gz", expires_in=3600,
                                 content_type="application/x-tar")
        mock_s3.generate_presigned_url.assert_called_once_with(
            "put_object",
            Params={"Bucket": "test-bucket", "Key": "bundles/job123.tar.gz",
                    "ContentType": "application/x-tar"},
            ExpiresIn=3600,
        )
        self.assertEqual(url, "https://fake/put-url")


# --------------------------------------------------------------------------- #
# Upload / download
# --------------------------------------------------------------------------- #
class TestUploadDownload(unittest.TestCase):

    def test_upload_file_checks_cap_and_calls_s3(self):
        client, mock_s3 = _client_with_mock_s3(max_bytes=10 * 1024 ** 3)
        pag = MagicMock()
        pag.paginate.return_value = [{"Contents": [{"Size": 100}]}]
        mock_s3.get_paginator.return_value = pag

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(b"fake video data")
            tmp = f.name
        try:
            result = client.upload_file(tmp, "videos/test.mp4")
            mock_s3.upload_file.assert_called_once()
            call_args = mock_s3.upload_file.call_args
            self.assertEqual(call_args[0][1], "test-bucket")
            self.assertEqual(call_args[0][2], "videos/test.mp4")
            self.assertIn("test-bucket", result)
            self.assertIn("videos/test.mp4", result)
        finally:
            os.unlink(tmp)

    def test_upload_file_raises_when_over_cap(self):
        client, mock_s3 = _client_with_mock_s3(max_bytes=50)
        pag = MagicMock()
        pag.paginate.return_value = [{"Contents": [{"Size": 40}]}]
        mock_s3.get_paginator.return_value = pag

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"x" * 20)   # 40 + 20 = 60 > 50
            tmp = f.name
        try:
            with self.assertRaises(StorageCapacityError):
                client.upload_file(tmp, "videos/big.mp4")
        finally:
            os.unlink(tmp)

    def test_upload_bytes_skips_check_when_disabled(self):
        client, mock_s3 = _client_with_mock_s3(max_bytes=1)
        client.upload_bytes(b"a lot of data", "key", check_cap=False)
        mock_s3.upload_fileobj.assert_called_once()
        mock_s3.get_paginator.assert_not_called()

    def test_download_file(self):
        client, mock_s3 = _client_with_mock_s3()
        with tempfile.TemporaryDirectory() as d:
            dest = os.path.join(d, "out.mp4")
            result = client.download_file("videos/clip.mp4", dest)
            mock_s3.download_file.assert_called_once_with(
                "test-bucket", "videos/clip.mp4", dest)
            self.assertEqual(result, dest)


# --------------------------------------------------------------------------- #
# List / delete
# --------------------------------------------------------------------------- #
class TestListDelete(unittest.TestCase):

    def test_list_objects_returns_all_pages(self):
        client, mock_s3 = _client_with_mock_s3()
        from datetime import datetime
        pag = MagicMock()
        pag.paginate.return_value = [
            {"Contents": [{"Key": "a.mp4", "Size": 100, "LastModified": datetime(2024, 1, 1)}]},
            {"Contents": [{"Key": "b.tar.gz", "Size": 200, "LastModified": datetime(2024, 1, 2)}]},
        ]
        mock_s3.get_paginator.return_value = pag
        objects = client.list_objects()
        self.assertEqual(len(objects), 2)
        self.assertEqual(objects[0]["key"], "a.mp4")
        self.assertEqual(objects[1]["size"], 200)

    def test_delete_object(self):
        client, mock_s3 = _client_with_mock_s3()
        client.delete_object("videos/old.mp4")
        mock_s3.delete_object.assert_called_once_with(
            Bucket="test-bucket", Key="videos/old.mp4")

    def test_delete_objects_batches_correctly(self):
        client, mock_s3 = _client_with_mock_s3()
        mock_s3.delete_objects.return_value = {"Deleted": [{}] * 5}
        keys = [f"file{i}.mp4" for i in range(5)]
        n = client.delete_objects(keys)
        mock_s3.delete_objects.assert_called_once()
        self.assertEqual(n, 5)

    def test_delete_objects_empty_list(self):
        client, mock_s3 = _client_with_mock_s3()
        n = client.delete_objects([])
        mock_s3.delete_objects.assert_not_called()
        self.assertEqual(n, 0)


# --------------------------------------------------------------------------- #
# make_job_payload
# --------------------------------------------------------------------------- #
class TestJobPayload(unittest.TestCase):

    def test_make_job_payload_shape(self):
        client, mock_s3 = _client_with_mock_s3()
        mock_s3.generate_presigned_url.side_effect = [
            "https://fake/get", "https://fake/put"]
        payload = client.make_job_payload(
            job_id="job123",
            video_key="videos/clip.mp4",
            bundle_key="bundles/job123.tar.gz",
            notify_url="https://backend/internal/jobs/job123/complete",
            video_meta={"width": 1920, "height": 1080},
        )
        self.assertEqual(payload["job_id"], "job123")
        self.assertEqual(payload["video_url"], "https://fake/get")
        self.assertEqual(payload["upload_url"], "https://fake/put")
        self.assertEqual(payload["notify_url"],
                         "https://backend/internal/jobs/job123/complete")
        self.assertEqual(payload["object_key"], "bundles/job123.tar.gz")
        self.assertEqual(payload["video_meta"]["width"], 1920)
        # must have both presigned URLs
        self.assertIn("video_url", payload)
        self.assertIn("upload_url", payload)

    def test_make_job_payload_generates_two_presigned_urls(self):
        client, mock_s3 = _client_with_mock_s3()
        mock_s3.generate_presigned_url.return_value = "https://fake/url"
        client.make_job_payload("j1", "vk", "bk", "https://notify")
        self.assertEqual(mock_s3.generate_presigned_url.call_count, 2)
        ops = [c[0][0] for c in mock_s3.generate_presigned_url.call_args_list]
        self.assertIn("get_object", ops)
        self.assertIn("put_object", ops)


# --------------------------------------------------------------------------- #
# _load_dotenv
# --------------------------------------------------------------------------- #
class TestLoadDotenv(unittest.TestCase):

    def test_loads_key_value_pairs(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env",
                                        delete=False) as f:
            f.write("# comment\n")
            f.write("TEST_R2_KEY=hello\n")
            f.write("TEST_R2_VAL = world \n")
            tmp = f.name
        try:
            env_before = dict(os.environ)
            os.environ.pop("TEST_R2_KEY", None)
            os.environ.pop("TEST_R2_VAL", None)
            # Patch the path _load_dotenv looks for
            with patch("violations.r2_client.os.path.dirname",
                       return_value=os.path.dirname(tmp)):
                with patch("violations.r2_client.os.path.exists", return_value=True):
                    with patch("builtins.open",
                               unittest.mock.mock_open(
                                   read_data="TEST_R2_KEY=hello\nTEST_R2_VAL=world\n")):
                        _load_dotenv()
            self.assertEqual(os.environ.get("TEST_R2_KEY"), "hello")
        finally:
            os.unlink(tmp)
            os.environ.pop("TEST_R2_KEY", None)
            os.environ.pop("TEST_R2_VAL", None)


if __name__ == "__main__":
    unittest.main()
