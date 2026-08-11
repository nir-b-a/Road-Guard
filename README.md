# Road Guard

Road Guard is a dashcam-based traffic violation detection system for Israeli roads. A driver records footage with an Android app; a computer-vision pipeline automatically detects violations committed by other vehicles; a human authority reviewer approves or dismisses each finding via a web dashboard before any action is taken.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Violation Types](#violation-types)
5. [Custom-Trained Models](#custom-trained-models)
6. [Setup & Running](#setup--running)
7. [Tests](#tests)
8. [Further Reading](#further-reading)

---

## System Overview

The system is built around a **Human-in-the-Loop** design: the pipeline surfaces potential violations and a qualified reviewer always makes the final call. This means a false positive reaching the dashboard is far less harmful than a real violation being silently dropped — the pipeline is tuned for high recall, and the human filters the noise.
Road-Guard is a human in the loop (HITL) automated traffic violation detection system that integrates android dashcam devices,
a cloud based processing pipeline and a web authority dashboard to detect,surface and enforce traffic violations.

---

## Architecture

```
┌─────────────────┐          direct upload        ┌──────────────────┐
│   Android App   │ ──────────────────────────►  │  Cloudflare R2   │
│  (dashcam rec.) │                               │  (raw video +    │
└────────┬────────┘                               │   sensor CSVs)   │
         │ POST /upload/init                      └────────┬─────────┘
         │ POST /upload/complete                           │ download
         ▼                                                 ▼
┌─────────────────┐   GET /internal/next-job   ┌──────────────────┐
│  Node Backend   │ ◄─────────────────────────  │   GPU Worker     │
│ (Express + Mongo│   POST /internal/violation  │  (worker.py)     │
│  + Cloudflare)  │ ◄─────────────────────────  └────────┬─────────┘
└────────┬────────┘                                      │ runs
         │ REST API                              ┌────────▼─────────┐
         ▼                                       │   CV Pipeline    │
┌─────────────────┐                              │  (main.py)       │
│ Authority        │                              │  YOLO + OpenCV + │
│ Dashboard        │                              │  FastALPR        │
│ (React)         │                              └──────────────────┘
└─────────────────┘
```

---

## Components

### Android App (`application/`)
Kotlin app that records 1080p@30fps dashcam video while logging six synchronized sensor streams: gyroscope, GPS, gravity, linear acceleration, and camera intrinsics. After a drive, the user uploads the session directly to Cloudflare R2.
> Stack: Kotlin, CameraX 1.4.0, OkHttp, Google Play Services Location, Android API 26+

### Backend (`backend/`)
REST API that manages users, drives, and violations. Coordinates the upload flow (validates the 7-file session contract), exposes a job queue for the GPU worker, and serves the authority dashboard.
> Stack: Node.js, Express 4, MongoDB (Mongoose), Cloudflare R2, JWT auth

### GPU Worker (`worker.py`)
Runs on the GPU machine. Polls the backend every 5 seconds for queued drives, downloads the session files from R2, runs the full CV pipeline, uploads evidence clips back to R2, and reports violations to the backend. The YOLO model is loaded once and reused across all jobs.

### CV Pipeline (`main.py`, `Objects/`, `violations/`, `speed_estimation/`, `lpr/`)
The core of the system. Processes each video frame with two YOLO models (vehicle tracking + lane segmentation), detects violations, reads license plates, and assembles a ranked evidence bundle (annotated clips, vehicle photos, plate crops, `.docx` reports).
> Stack: Python 3.10, PyTorch + CUDA, Ultralytics YOLOv8/v11, OpenCV, FastALPR, SciPy, boto3

### Authority Dashboard (`frontend/roadguard-authority/`)
Web dashboard where authority reviewers watch violation clips (streamed directly from R2), view vehicle photos and plate reads, and verify or dismiss each finding.
> Stack: React 19, React Router v7, Tailwind CSS, Vite

---

## Violation Types

### Solid-Line Crossing
Detected when a vehicle's tracked bounding box crosses a solid lane marking. Uses a ghost-mask tracker to handle occlusion, a motion direction filter to suppress false positives from oncoming traffic, and a K-of-M temporal gate before a crossing is confirmed.

### Speeding
Reconstructed from pinhole camera geometry combined with ego-motion estimates from the Android sensor streams (gyroscope + GPS + gravity). A Kalman smoother filters the velocity estimate, which is then compared against the speed limit retrieved from the OpenStreetMap Overpass API.

#### how it looks inside the brain :
![Solid-line crossing violation](docs/violation_solid_line.png)
![License plate read](docs/lpr_plate_read.png)

### Yellow-Line / Shoulder Driving
Detected when a vehicle drives on the wrong side of a yellow lane boundary (shoulder or oncoming lane). Uses a two-sigmoid confidence scorer and an Israeli-road prior (yellow markings appear only on the far-right edge or median).

---

## Custom-Trained Models

Three models in this project were trained from scratch on custom datasets. See [`WEIGHTS.md`](WEIGHTS.md) for download instructions.

| Model file | What it detects | Used for |
|---|---|---|
| `israeli_plates.pt` | Israeli license plates | Feeding crops into FastALPR for plate reading |
| `weights/phase3_v3_yellowprotect.pt` | Lane markings (`solid`, `dashed`, `yellow`) | Solid-line and yellow-line violation detection |
| `models/tire_yolo11n.pt` | Vehicle tires | Stage-2 cascade: confirms solid-line crossings by detecting tires crossing the line |

The general vehicle tracker (`yolo11x.pt`) is an off-the-shelf Ultralytics model, auto-downloaded on first run.



---

## Setup & Running

### First-time setup (from a fresh clone)

Three commands take a new machine from nothing to a runnable pipeline. **Order matters:**
`fetch_models.py` imports `ultralytics`, so the dependencies must be installed first.

```bash
# 1. Clone
git clone <repo-url>
cd malshinon_master

# 2. Install Python dependencies  (Python 3.10, CUDA-capable GPU)
pip install -r requirements.txt

# 3. Fetch and verify every model weight
python tools/fetch_models.py
```

> **Step 2 caveat:** follow the numbered install steps *inside* `requirements.txt` — PyTorch
> with CUDA must be installed manually before the rest, or pip will pull a CPU-only build.

That is the whole model setup. **No weight is ever downloaded by hand, and no Roboflow or
other API key is needed to run the system** — only to retrain. Step 3 prints a line per model
and exits non-zero if anything required is missing or corrupt, so it doubles as a pre-demo
smoke check:

```
  [OK] lane_seg                 weights/phase3_v3_yellowprotect.pt     6.8 MB verified
  [OK] tire                     models/tire_yolo11n.pt                 5.5 MB verified
  [OK] plate_detector_israeli   israeli_plates.pt                      6.3 MB verified
  [OK] yolo11x                  yolo11x.pt                             114.6 MB verified
  All models present and verified.
```

Then pick a surface below: Docker for the whole stack, or *Python Pipeline* to run the
worker natively against a containerised backend.

### Quick start (Docker)

```bash
cp backend/.env.example backend/.env   # fill in JWT_SECRET and the R2_* values

docker compose up --build                 # MongoDB + REST API + authority dashboard
docker compose --profile cpu up --build   # ... + the CV worker, CPU-only
docker compose --profile gpu up --build   # ... + the CV worker, CUDA 11.8
```

- Dashboard → <http://localhost:4173>
- API → <http://localhost:5000/api> (health: `/api/health`)

| Command | What you get | Cost |
|---------|--------------|------|
| `docker compose up` | Mongo + API + dashboard. Enough to browse the product. | ~400 MB, no extra host setup |
| `--profile cpu` | The above **plus** the full CV pipeline, end to end. Processes clips perhaps 20–40× slower than GPU. | ~2.9 GB image |
| `--profile gpu` | The above at real throughput. | ~10 GB image, **needs an NVIDIA driver + the NVIDIA Container Toolkit** (on Windows: Docker Desktop with the WSL2 backend) |

The worker sits behind a profile so a plain `docker compose up` stays a fast three-container
demo with no GPU prerequisites. Run **at most one** worker profile at a time — both claim from
the same job queue.

Startup is ordered by health, not by luck: the API waits for a real `mongosh` ping, and the
worker waits for `/api/health` to report `mongo: connected` before it starts polling.

**Model weights are not baked into the images.** The 114 MB `yolo11x.pt` and the FastALPR ONNX
cache live in a named `models-cache` volume that `worker/entrypoint.sh` populates on first
start (~9 s) and links on every start after (~1 s). Only the three small committed weights
ship in the layer.

To run the worker natively against the containerised backend instead, start the stack without
a profile and see *Python Pipeline* below. Note that `worker.env` points at
`http://localhost:5001`; the containerised backend publishes **5000**.

#### Container layout

| File | Role |
|------|------|
| [`docker-compose.yml`](docker-compose.yml) | The whole stack. Profiles `cpu` / `gpu` gate the worker. |
| [`backend/Dockerfile`](backend/Dockerfile) | Node 22 alpine, `npm ci --omit=dev`. |
| [`frontend/roadguard-authority/Dockerfile`](frontend/roadguard-authority/Dockerfile) | Two-stage: `vite build`, then `vite preview` over the static bundle. |
| [`worker/Dockerfile.cpu`](worker/Dockerfile.cpu) | `python:3.10-slim` + torch 2.1.2 from the CPU wheel index. |
| [`worker/Dockerfile.gpu`](worker/Dockerfile.gpu) | `nvidia/cuda:11.8-cudnn8-runtime` + torch 2.1.2+cu118. |
| [`worker/entrypoint.sh`](worker/entrypoint.sh) | Links the cached weights in from the volume, fetches what is missing, links the result back out. |
| [`worker/verify_env.py`](worker/verify_env.py) | Build-time gate: asserts the torch flavour, the headless-OpenCV substitution, and that every pipeline module imports. Run as the last layer, so a broken dependency set fails the **build**. |
| [`requirements-worker.txt`](requirements-worker.txt) | Container dependency set — headless OpenCV, no PaddleOCR, no UnLanedet extras. |
| [`.dockerignore`](.dockerignore) | Allowlist. Trims the worker build context from ~500 MB to ~19 MB. |

#### Testing the stack

```bash
pytest tests/test_docker_stack.py                    # static + daemon checks (~5 s)
pytest tests/test_docker_stack.py --rundocker-build  # + build the image and assert on it
pytest tests/test_docker_stack.py -m "not docker"    # no daemon needed
cd backend && npm test                               # includes the /api/health contract
```

### Model weights

```bash
python tools/fetch_models.py           # fetch anything missing, then verify
python tools/fetch_models.py --check   # verify only, no network (offline / CI / pre-demo)
python tools/fetch_models.py --list    # inventory: which models, and where each comes from
```

Every model falls into one of three groups, which is why no manual download step exists:

| Group | Models | How you get it |
|-------|--------|----------------|
| **Ours, committed** | `weights/phase3_v3_yellowprotect.pt`, `models/tire_yolo11n.pt`, `israeli_plates.pt` | Arrives with the clone — each is under 7 MB. `.gitignore` excludes `*.pt` then re-includes exactly these three. |
| **Third-party, auto-downloaded** | `yolo11x.pt` (114 MB) | Ultralytics fetches it on first use; `fetch_models.py` pins the working directory so it lands where `Constants.YOLO_VERSION` expects. |
| **Third-party, self-caching** | FastALPR plate detector + OCR (ONNX) | `fast_alpr` pulls both into its own cache when the reader is first constructed. |

Checksums are enforced asymmetrically on purpose: a mismatch on **our** committed weights is a
hard failure (those bytes must never change), while a third-party mismatch is only a warning,
since upstream can legitimately re-cut a release.

If a committed weight is reported missing or corrupt, restore it with
`git checkout -- <path>` rather than re-downloading.

See [WEIGHTS.md](WEIGHTS.md) for per-model detail and
[docs/EXTERNAL_COMPONENTS.md](docs/EXTERNAL_COMPONENTS.md) for provenance — who trained each
model, on what data, under which licence.

### Python Pipeline

**Prerequisites:** Python 3.10, CUDA-capable GPU, and *First-time setup* above completed
(dependencies installed, `python tools/fetch_models.py` green).

```bash
# Start the worker — polls the backend and processes drives automatically
python worker.py
```


### Backend

**Prerequisites:** Node.js 18+, MongoDB instance, Cloudflare R2 bucket.

```bash
cd backend
npm install
cp .env.example .env   # fill in Mongo URI, R2 credentials, JWT secret
npm run dev            # development (nodemon)
node server.js         # production
```

### Frontend

**Prerequisites:** Node.js 18+, backend running.

```bash
cd frontend/roadguard-authority
npm install
npm run dev
```

### Android App

Open `application/` in Android Studio. Update `BASE_URL` in `PostDriveActivity.kt` to point to your backend. Build and deploy via Gradle (`assembleDebug` or `assembleRelease`). Requires Android API 26+.



---

## Further Reading

| Document | Contents |
|---|---|
| [`BRAIN_ARCHITECTURE.md`](BRAIN_ARCHITECTURE.md) | Stage-by-stage walkthrough of the CV pipeline: models, detection logic, output files, CLI flags |
| [`docs/ROADGUARD_OVERVIEW.md`](docs/ROADGUARD_OVERVIEW.md) | Full design document: philosophy, building blocks, lessons learned, results, roadmap |
| [`BRAIN_BACKEND_INTEGRATION_HANDOFF.txt`](BRAIN_BACKEND_INTEGRATION_HANDOFF.txt) | Integration contract: bundle structure, API contract, DB schema, upload flow |
| [`WEIGHTS.md`](WEIGHTS.md) | Model weight download guide |
| [`docs/EXTERNAL_COMPONENTS.md`](docs/EXTERNAL_COMPONENTS.md) | Full disclosure of every external library, open-source project, pretrained model, dataset and service — with licences, and what we built ourselves |
