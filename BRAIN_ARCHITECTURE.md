# Road Guard — Brain Architecture

How a dashcam video flows through the pipeline end-to-end: which files run, which models are loaded, and what comes out.

---

## Entry point

```
python main.py <video.mp4> [flags]
```

The backend sends a job payload (`--job-payload`) containing a Cloudflare presigned URL. `main.py` pulls + verifies the clip, runs the full pipeline, then pushes the evidence bundle back.

---

## Models loaded

| Model | Weight file | What it does |
|-------|------------|--------------|
| **Vehicle detector** | `yolo11x.pt` (auto-downloaded) | Detects + tracks cars, trucks, buses, motorcycles, traffic lights every frame using BoT-SORT |
| **Lane-seg model** | `weights/phase3_v3_yellowprotect.pt` | YOLOv8-seg, classifies lane markings as `solid`, `dashed`, or `yellow` per frame |
| **Plate detector** | `israeli_plates.pt` | FastALPR — detects + reads Israeli license plates from vehicle crops |
| **Tire model** *(Stage-2, pending wiring)* | `models/tire_yolo11n.pt` | YOLOv11n — confirms a vehicle's tires straddle a solid line (FP filter for solid-line crossing) |

---

## Pipeline stages (in order)

### 1. Frame loop — vehicle tracking
**File:** `main.py:processFrame` → `Objects/World.py`

Every frame:
- `yolo11x.pt` runs vehicle detection + BoT-SORT tracking → assigns stable `track_id` per vehicle
- Vehicle bboxes fed into `World` (the in-memory vehicle state tracker)
- Each vehicle crop offered to `EvidenceCollector` (rolling top-K sharpest frames, no re-read needed later)

### 2. Lane segmentation (same frame loop)
**Files:** `main.py:seg_lanes` → `tools/ghost_mask.py`

Every frame (if `--no-yellow` is not set):
- `phase3_v3_yellowprotect.pt` runs on the raw frame → returns lane polygons with class + confidence
- Optical-flow ego-shift (`tools/ghost_mask.estimate_ego_shift`) compensates for camera movement
- Result stored in `seg_frames` list: `{frame, vehicles, lanes, shift}` — shared by ALL violation rules below

### 3. Speed estimation (post-loop)
**Files:** `speed_estimation/speed_estimator.py`, `speed_estimation/ego_yaw.py`, `speed_estimation/overspeed.py`

- Requires ego pose (Android sensor CSVs or simulation telemetry)
- Reconstructs world-frame speed per vehicle per frame via pinhole geometry + Kalman smoothing
- Without pose data: speed estimation is skipped (no Android CSVs → no speeding detection)

### 4. Violation detection (post-loop)

#### 4a. Solid-line crossing
**Files:** `main.py:evaluate_crossing` → `tools/ghost_mask.compute_verdict_timeline` → `tools/motion_filter.merge_events`

- Walks `seg_frames`, checks each vehicle's contact point against `solid` lane polygons
- Ghost-mask TTL (0.5 s) handles line occlusion under the vehicle
- K-of-M filter + 3 s cooldown → one `ViolationEvent(SOLID_LINE_CROSSING)` per incident
- **Stage-2 cascade** (`violations/cascade/`) will add tire + geometry confirmation once wired in

#### 4b. Yellow-line (shoulder) driving
**Files:** `main.py:evaluate_yellow_line` → `tools/shoulder_violation.py`

- Same `seg_frames` cache, checks contact against `yellow` lane class
- Two-sigmoid confidence scorer (distance × angle), 0.30 oncoming down-weight
- Speed gate: uses world-frame speed if available, motion proxy otherwise
- K-of-M + age + side-switch guards → `ViolationEvent(YELLOW_LINE_CROSSING)` per incident

#### 4c. Speeding
**Files:** `speed_estimation/overspeed.py`

- Android path: compares per-vehicle world speed to ego GPS speed + margin
- Returns `OverspeedEvent` → converted to `ViolationEvent(SPEEDING)`

### 5. Evidence collection
**Files:** `main.py:run_evidence_and_report` → `lpr/evidence.py` → `lpr/reader.py`

For every `ViolationEvent`:
- Read plate ONCE per vehicle from the in-memory crop buffer (`EvidenceCollector`)
- `israeli_plates.pt` (FastALPR) OCRs the best crop → plate string + confidence
- Unknown plates written as `"UNKNOWN - manual review"` (recall-first: never dropped)
- Writes `{video}_violations.csv` + evidence crop images

### 6. Backend export
**Files:** `main.py:export_and_push_violations` → `violations/export.py` → `violations/ingest_client.py`

- Builds prioritised violation bundle: **Solid > Speeding > other > Yellow** (lexicographic tier, confidence within tier)
- Per violation: 10 s annotated clip + 3 best-picture crops + speeding `.docx` (stdlib OOXML)
- Cloudflare path: presigned PUT of `.tar.gz` + webhook notify → backend marks job complete

---

## Output files (written next to the source video)

| File | Contents |
|------|---------|
| `{name}_violations.csv` | All violations: type, frame, confidence, plate, speed |
| `{name}_evidence/` | Plate crops + best vehicle pictures per violation |
| `{name}_violations_bundle.tar.gz` | Full evidence bundle sent to backend |
| `{name}_annotated.mp4` | Annotated video (red boxes, captions) |
| `{name}_vehicles.csv` | Per-track distance + speed summary |
| `{name}_perframe.csv` | Per-frame: n_vehicles, max_speed, yellow_conf |
| `{name}_tracks.csv` | Per-frame bboxes for all tracked vehicles |

---

## Key flags

| Flag | Effect |
|------|--------|
| `--job-payload <json>` | Worker mode: pull video from Cloudflare presigned URL, push bundle back |
| `--push-url <url>` | Legacy: POST bundle to Node backend directly |
| `--no-yellow` | Skip lane-seg model entirely (faster, no yellow/solid violation detection) |
| `--no-evidence` | Skip LPR / plate reading |
| `--lane-weights <path>` | Override lane-seg weight path (default: `weights/phase3_v3_yellowprotect.pt`) |
| `--model <path>` | Override vehicle detector (default: `yolo11x.pt`) |
| `--benchmark` | CSV-only fast mode, no annotated video |

---

## Minimal standalone run (no backend, no GPS)

```bash
python main.py path/to/video.mp4
```
Runs vehicle tracking + lane-seg + solid/yellow line detection + LPR. Outputs CSV + annotated video. No speeding detection (needs GPS/telemetry).
