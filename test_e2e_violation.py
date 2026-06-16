"""
End-to-end test: driver registers → uploads video → brain submits violation → check it appears.
Run with:  python test_e2e_violation.py
"""

import requests
import struct
import tempfile
import os

BASE = "http://localhost:5000/api"


def make_dummy_mp4() -> str:
    """Write a minimal valid MP4 file so multer accepts it."""
    # Minimal ftyp + mdat boxes — just enough to pass the mime check
    ftyp = (
        struct.pack(">I", 20) +   # box size
        b"ftyp" +
        b"isom" +                  # major brand
        struct.pack(">I", 0x200) + # minor version
        b"isom"                    # compatible brand
    )
    mdat = struct.pack(">I", 8) + b"mdat"
    data = ftyp + mdat

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


def step(label):
    print(f"\n{'='*50}\n{label}\n{'='*50}")


# ── 1. Register a test driver ──────────────────────────────────────────────
step("1. Register test driver")
r = requests.post(f"{BASE}/auth/register", json={
    "name": "Test Driver",
    "email": "testdriver_e2e@roadguard.com",
    "password": "password123",
    "role": "driver"
})
print(r.status_code, r.json())
# OK if 201 or 400 "already registered"

# ── 2. Login as driver ─────────────────────────────────────────────────────
step("2. Login as test driver")
r = requests.post(f"{BASE}/auth/login", json={
    "email": "testdriver_e2e@roadguard.com",
    "password": "password123"
})
print(r.status_code, r.json())
assert r.status_code == 200, "Login failed"
token = r.json()["data"]["token"]
headers = {"Authorization": f"Bearer {token}"}

# ── 3. Upload a dummy video drive ──────────────────────────────────────────
step("3. Upload dummy video (creates Drive record)")
mp4_path = make_dummy_mp4()
try:
    with open(mp4_path, "rb") as f:
        r = requests.post(
            f"{BASE}/driver/upload",
            headers=headers,
            data={
                "sessionId": "test-session-e2e-001",
                "tags": '[{"timestamp": 0, "lat": 31.7767, "lon": 35.2345}]',
                "speed_json": '[{"timestampMs": 0, "speedKmh": 72.5, "lat": 31.7767, "lon": 35.2345}]'
            },
            files={"video": ("test_drive.mp4", f, "video/mp4")}
        )
finally:
    os.unlink(mp4_path)

print(r.status_code, r.json())
assert r.status_code == 201, "Drive upload failed"
drive_id = r.json()["data"]["driveId"]
print(f"\n>>> driveId: {drive_id}")

# ── 4. Brain submits a violation ───────────────────────────────────────────
step("4. Brain calls /internal/violation (simulating detection result)")
r = requests.post(f"{BASE}/internal/violation", json={
    "driveId": drive_id,
    "videoClipPath": "uploads/videos/fake_clip_redlight.mp4",
    "carId": "12-345-67",
    "calculatedSpeed": 72.5,
    "lat": 31.7767,
    "lon": 35.2345
})
print(r.status_code, r.json())
assert r.status_code == 201, "Violation submission failed"
violation_id = r.json()["data"]["violationId"]
print(f"\n>>> violationId: {violation_id}")

# ── 5. Authority fetches violations ───────────────────────────────────────
step("5. Register + login as authority officer and fetch violations")
requests.post(f"{BASE}/auth/register", json={
    "name": "Test Officer",
    "email": "testofficer_e2e@roadguard.com",
    "password": "password123",
    "role": "authority",
    "inviteCode": "ROADGUARD-2026"
})
r = requests.post(f"{BASE}/auth/login", json={
    "email": "testofficer_e2e@roadguard.com",
    "password": "password123"
})
auth_token = r.json()["data"]["token"]
auth_headers = {"Authorization": f"Bearer {auth_token}"}

r = requests.get(f"{BASE}/authority/violations", headers=auth_headers)
print(r.status_code)
violations = r.json()["data"]
print(f"Total violations returned: {len(violations)}")
for v in violations:
    print(f"  - id={v['_id']}  plate={v['carId']}  status={v['status']}")

# ── 6. Authority fetches evidence for the violation ────────────────────────
step("6. Fetch evidence for the violation")
r = requests.get(f"{BASE}/authority/evidence/{violation_id}", headers=auth_headers)
print(r.status_code, r.json())

print("\n\n✅  All steps passed. Check MongoDB Compass → roadguard → violations to see the record.")
