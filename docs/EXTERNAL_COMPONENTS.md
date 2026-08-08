# External Components Disclosure — Road Guard

**Purpose.** This document declares *every* external component used in Road Guard: third-party
libraries, open-source projects, pretrained model weights, datasets, and hosted services. It exists
to satisfy the project requirement that all external components be reported explicitly.

**How to read the Origin column**

| Marker | Meaning |
|---|---|
| **External** | Third-party. We use it as published; we did not write it. |
| **In-house** | Written by the Road Guard team from scratch. |
| **Derived** | Third-party code or weights that we modified, patched, or fine-tuned. The base is external; the delta is ours. **These are called out individually below.** |

Versions were read from the actual installed environment (`importlib.metadata`, `node_modules`,
`libs.versions.toml`), not from the manifests, so they reflect what really runs.

---

## 1. Pretrained models and weights

The most important table here: these are trained artefacts, and three of the six running models are
third-party.

| Model | Architecture | Role in pipeline | Origin |
|---|---|---|---|
| `yolo11x.pt` (~109 MB) | YOLO11-X | Vehicle + traffic-light detection; the main tracker's detector | **External** — Ultralytics, pretrained on COCO |
| BoT-SORT (`botsort.yaml`) | Kalman + ReID tracker | Assigns persistent `track_id` across frames | **External** — via Ultralytics (see §2 for our patch) |
| `weights/phase3_v3_yellowprotect.pt` (6.8 MB) | YOLOv8-seg | Lane segmentation: solid white, yellow, dashed, road surface | **Derived** — fine-tuned in-house from a YOLOv8-seg base on our Roboflow-curated dataset |
| `models/tire_yolo11n.pt` (5.5 MB) | YOLO11n | Tire detection for the Stage-2 solid-line cascade | **Derived** — fine-tuned in-house from a YOLO11n base on an external Roboflow tire dataset |
| `yolo-v9-t-384-license-plate-end2end` | YOLOv9-t (ONNX) | Locates the licence plate within a vehicle crop | **External** — bundled by `fast_alpr`, auto-downloaded on first use |
| `global-plates-mobile-vit-v2-model` | MobileViT-v2 (ONNX) | OCR of the located plate | **External** — bundled by `fast_alpr`, auto-downloaded on first use |

Present in the repository but **not** on the live path:

| Model | Status |
|---|---|
| `israeli_plates.pt` (6.3 MB) | **Derived** — our fine-tuned Israeli-plate locator. Used only by the alternative `PaddleOCRDetectorReader`; FastALPR is the active reader. |
| `clrernet_model_best_culane.pth` (~264 MB) | **External** — CLRerNet model zoo, pretrained on CULane. Used only by the `tools/test_corun_video.py` comparison tool, never by the production pipeline. |

Run `python tools/fetch_models.py --list` for the machine-readable version of this table, and
`--check` to verify every weight against its SHA-256.

---

## 2. Open-source projects and algorithms

Whole codebases and published algorithms we build on. This is the highest-sensitivity category.

| Component | Version | License | Role | Origin |
|---|---|---|---|---|
| **Ultralytics YOLO** | 8.4.75 | **AGPL-3.0** | Detection, segmentation and tracking framework; the backbone of the CV pipeline | **External** |
| **BoT-SORT** | (via Ultralytics) | AGPL-3.0 | Multi-object tracking algorithm | **Derived** — see below |
| **FastALPR** | 0.4.0 | MIT | End-to-end licence-plate detection + OCR | **External** |
| **UnLanedet** | git clone (not on PyPI) | see upstream repo | Deep-learning lane-detection framework, evaluated for the lane model | **External** |
| **CLRerNet** | model zoo weight | see upstream repo | Lane-detection network used for comparison in `tools/test_corun_video.py` | **External** |
| **PaddleOCR** | 2.7.3 (optional, not installed by default) | Apache-2.0 | Text recognition for the alternative plate-reader path | **External** |
| **FFmpeg** | system binary | LGPL-2.1 / GPL-2.0 | H.264 re-encode with `+faststart` for browser-playable evidence clips | **External** |

### ⚠️ Declared modification: `speed_estimation/botsort_patch.py`

We **monkey-patch Ultralytics' BoT-SORT implementation at runtime**. This is derived work on an
AGPL-3.0 codebase and is declared explicitly:

1. **Kalman-filter stability fix** — floating-point error accumulates in the covariance matrix until
   eigenvalues go negative and Cholesky decomposition fails. Our patch catches the `cho_factor`
   failure and projects the matrix to the nearest positive-definite one (symmetrise, then clip
   eigenvalues).
2. **Scale-aware ReID cutoff** — BoT-SORT extracts appearance features from the detection crop. For
   distant vehicles that crop is ~30×20 px, too little signal for reliable matching, so the combined
   ReID+IoU score drops below threshold and the track is dropped and re-IDed. Our patch disables ReID
   below `REID_MIN_HEIGHT` and falls back to IoU-only matching.

The patched symbols are `ultralytics.trackers.bot_sort.BOTSORT` and
`ultralytics.trackers.utils.matching`. The algorithm is theirs; the two fixes are ours.

---

## 3. Python CV worker

Runtime: **Python 3.10**, CUDA 11.8.

| Library | Version | License | Role |
|---|---|---|---|
| torch | 2.1.2+cu118 | BSD-3-Clause | Deep-learning runtime; GPU inference |
| torchvision | 0.16.2+cu118 | BSD-3-Clause | Vision ops and transforms backing Ultralytics |
| ultralytics | 8.4.75 | **AGPL-3.0** | YOLO detection / segmentation / tracking |
| opencv-python | 4.10.0.84 | Apache-2.0 | Frame decode, drawing, Laplacian sharpness scoring |
| numpy | 1.26.4 | BSD-3-Clause | Array maths throughout |
| scipy | 1.13.1 | BSD-3-Clause | Savitzky-Golay smoothing, Kalman/RTS, least-squares camera-height solve |
| pandas | 2.2.2 | BSD-3-Clause | Reads the dashcam sensor CSVs |
| matplotlib | 3.10.9 | PSF-based | Per-vehicle speed/distance diagnostic plots |
| lapx | 0.9.4 | MIT | Linear-assignment solver used by the BoT-SORT/ByteTrack matching step |
| pillow | 10.4.0 | HPND | Image I/O |
| onnxruntime | 1.23.2 | MIT | Runs the two FastALPR ONNX models |
| fast_alpr | 0.4.0 | MIT | Licence-plate detection + OCR pipeline |
| boto3 | 1.43.36 | Apache-2.0 | S3 client for Cloudflare R2 upload/download |
| requests | 2.28.1 | Apache-2.0 | HTTP to the job server and the Overpass API |
| pyyaml | 6.0.2 | MIT | Tracker/model config parsing |
| tqdm | 4.66.5 | MPL-2.0 AND MIT | Progress bars |
| scikit-learn | 1.5.1 | BSD-3-Clause | Clustering/metrics in dataset tooling |
| scikit-image | 0.25.2 | BSD-3-Clause | Image metrics in dataset tooling |
| shapely | 2.1.2 | BSD-3-Clause | Polygon geometry for lane contours |
| imgaug | 0.4.0 | MIT | Training-time augmentation (UnLanedet path) |
| albumentations | 1.4.10 | MIT | Training-time augmentation (UnLanedet path) |
| yt-dlp | 2026.3.17 | Unlicense | Downloads the public dashcam clips used as test data |

---

## 4. Backend — Node.js / Express REST API

Runtime: **Node.js 20 LTS**.

| Package | Version | License | Role |
|---|---|---|---|
| express | 4.22.1 | MIT | HTTP routing and middleware |
| mongoose | 8.23.0 | MIT | MongoDB ODM: schemas, validation, queries |
| jsonwebtoken | 9.0.3 | MIT | Stateless JWT auth for driver and authority roles |
| bcryptjs | 2.4.3 | MIT | Password hashing |
| @aws-sdk/client-s3 | 3.1075.0 | Apache-2.0 | S3 API client against Cloudflare R2 |
| @aws-sdk/s3-request-presigner | 3.1075.0 | Apache-2.0 | Mints the short-lived presigned PUT/GET URLs |
| cors | 2.8.6 | MIT | Cross-origin access for the dashboard |
| dotenv | 16.6.1 | BSD-2-Clause | Loads configuration and secrets from `.env` |
| express-rate-limit | 7.5.1 | MIT | Per-IP request cap |
| express-async-errors | 3.1.1 | ISC | Propagates async route errors to the handler |
| axios | 1.13.6 | MIT | Outbound HTTP |
| jest | 30.3.0 | MIT | Test runner *(dev)* |
| supertest | 7.2.2 | MIT | HTTP assertions against the Express app *(dev)* |
| mongodb-memory-server | 11.0.1 | MIT | Ephemeral MongoDB for tests *(dev)* |
| nodemon | 3.1.14 | MIT | Dev auto-reload *(dev)* |

---

## 5. Frontend — authority web dashboard

| Package | Version | License | Role |
|---|---|---|---|
| react / react-dom | 19.2.4 | MIT | UI framework |
| react-router-dom | 7.13.1 | MIT | Client-side routing |
| axios | 1.13.6 | MIT | REST calls to the backend |
| vite | 8.0.1 | MIT | Build tool and dev server |
| @vitejs/plugin-react | 6.0.1 | MIT | React fast-refresh integration |
| tailwindcss | 3.4.19 | MIT | Utility-first CSS framework — **all dashboard styling** |
| postcss | 8.5.8 | MIT | CSS transform pipeline for Tailwind |
| autoprefixer | 10.4.27 | MIT | Vendor prefixing |
| eslint (+ `@eslint/js`, react-hooks, react-refresh, globals) | 9.39.4 | MIT | Linting *(dev)* |
| @types/react, @types/react-dom | 19.2.x | MIT | Type definitions *(dev)* |

---

## 6. Android driver application

Language: **Kotlin 2.0.21** (not Java). Build: Android Gradle Plugin 8.10.1, Gradle 8.11.1.
`minSdk` 26, `compileSdk`/`targetSdk` 35, JVM target 11.

| Dependency | Version | License | Role |
|---|---|---|---|
| androidx.camera:camera-core / camera2 / lifecycle / video / view | 1.4.0 | Apache-2.0 | **CameraX** — the video recording pipeline |
| com.squareup.okhttp3:okhttp | 4.12.0 | Apache-2.0 | HTTP client; uploads files to R2 via presigned PUT |
| com.google.android.gms:play-services-location | 21.3.0 | Google APIs ToS | GPS / fused location for the speed and position streams |
| com.google.android.material:material | 1.12.0 | Apache-2.0 | Material Components UI |
| androidx.core:core-ktx | 1.13.1 | Apache-2.0 | Kotlin extensions on the Android framework |
| androidx.appcompat:appcompat | 1.7.0 | Apache-2.0 | Backwards-compatible activities and theming |
| androidx.constraintlayout:constraintlayout | 2.1.4 | Apache-2.0 | Layout engine |
| androidx.activity:activity-ktx | 1.9.0 | Apache-2.0 | Activity result / lifecycle APIs |
| androidx.recyclerview:recyclerview | 1.3.2 | Apache-2.0 | Notification and drive lists |
| androidx.cardview:cardview | 1.0.0 | Apache-2.0 | Card UI containers |
| junit:junit | 4.13.2 | EPL-1.0 | Unit tests *(test)* |
| androidx.test.ext:junit | 1.2.1 | Apache-2.0 | Instrumented tests *(test)* |
| androidx.test.espresso:espresso-core | 3.6.1 | Apache-2.0 | UI tests *(test)* |

Also used from the Android framework itself (no dependency entry): `SensorManager` for gravity,
gyroscope and linear-acceleration streams, and `Camera2` intrinsics via CameraX interop.

---

## 7. Datasets

| Dataset | Origin | Use |
|---|---|---|
| **COCO** | **External** (Microsoft) | Pretraining behind `yolo11x.pt` — we did not train on it directly, but the detector's weights derive from it |
| **CULane** | **External** | Pretraining behind the CLRerNet comparison weight |
| **Roboflow tire dataset** | **External** source, curated in-house | Base data for `tire_yolo11n.pt`. We collapsed its two label classes into a single `tire` class after finding ~93% of the second class were real tires |
| **Road Guard lane dataset** | **Derived** — external dashcam footage, labelled in-house via Roboflow | Training data for `phase3_v3_yellowprotect.pt` |
| **Public dashcam clips (YouTube)** | **External** source material, downloaded with `yt-dlp` | Validation and test set for the violation pipeline |
| **Team dashcam recordings** | **In-house** | End-to-end integration testing |

---

## 8. Hosted services and infrastructure

| Service | Role | Origin |
|---|---|---|
| **Cloudflare R2** | S3-compatible object storage for all video and evidence; presigned PUT/GET keeps media off the app server | **External** |
| **MongoDB / MongoDB Atlas** | Document store for Drive, Violation, User and Notification records | **External** |
| **Overpass API** (`overpass-api.de`, `overpass.kumi.systems`, `overpass.private.coffee`) | OpenStreetMap speed-limit lookup by GPS coordinate, for the speeding rule | **External** |
| **Roboflow** | Dataset hosting, labelling and augmentation for our two fine-tuned models | **External** |
| **NVIDIA CUDA 11.8 / cuDNN** | GPU acceleration for inference | **External** |

---

## 9. What we built ourselves

Stated for contrast, so the boundary between external and in-house work is unambiguous.

| Component | Description |
|---|---|
| Solid-line crossing detection (Stage 1) | Geometric contact test between tracked vehicles and segmented lane contours, with a K-of-M temporal filter |
| **Stage-2 cascade** | Precision filter over Stage-1: tire-contact geometry, axle-vector straddle, plate ground-projection proxy, heading estimation, curve gating, side-switch tracking. `violations/cascade/` |
| Ego-speed estimation | Fuses GPS, gravity, gyroscope, linear acceleration and camera intrinsics into a per-frame speed signal |
| Speeding rule | K-of-N sliding window over a 1.5 s buffer against the Overpass-derived limit |
| Yellow-line right-of-way rule | Signed-side test against the segmented yellow line |
| Violation priority model | Lexicographic type-tiered ranking (solid-line → speeding → other → yellow), confidence ordering within a tier only |
| Evidence pipeline | In-memory rolling buffer of the sharpest vehicle crops, annotated clip extraction, per-violation evidence bundle with SHA-256 integrity |
| Video integrity checks | ffprobe fingerprint at ingest, hop validation at every pipeline boundary |
| Backend REST API | All endpoints, the drive state machine, job queue and watchdog |
| Authority dashboard | All screens and components |
| Android driver app | All screens, recording and upload logic |
| BoT-SORT patches | Two fixes on the external tracker — see §2 |

---

## 10. Licence note

Most dependencies are permissive (MIT, BSD, Apache-2.0) and carry only attribution obligations,
satisfied by this document.

**One exception is worth stating explicitly: Ultralytics YOLO is AGPL-3.0**, a copyleft licence.
Road Guard is an academic project and is not distributed as a commercial product, so the AGPL's
source-availability obligation is met by the project repository. Any future commercial deployment
would require either an Ultralytics commercial licence or replacing the detector. `torch`,
`opencv-python` and every other CV dependency are permissively licensed and impose no such
condition.

FFmpeg is invoked as an external binary (not linked), and standard builds are LGPL-2.1; this is the
usual arrangement for tools that shell out to `ffmpeg`.
