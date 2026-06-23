# Backend integration loop — local end-to-end demo

Proves the full **brain ⇄ backend** HTTP transaction on one machine, with no cloud and no database:
the backend serves a raw video, the brain pulls it, analyses it, and POSTs a prioritised evidence
bundle back, which lands on disk.

```
 ┌────────────────────────┐                         ┌──────────────────────────────┐
 │  mock_backend.py        │                         │  run_brain.py ("the brain")  │
 │  (FastAPI, Terminal 1)  │                         │  (Terminal 2)                │
 ├────────────────────────┤                         ├──────────────────────────────┤
 │ GET /video/{name}/meta ─┼──── reference VideoMeta ─► 1. PULL + fail-fast verify   │
 │ GET /video/{name}      ─┼──── raw .mp4 bytes ──────►    (ingest_client)           │
 │                         │                         │ 2. ANALYSE (cached crossings  │
 │                         │                         │    + synthetic speeding/yellow│
 │                         │                         │ 3. clip(10s pre/5s post)+3 pics│
 │                         │                         │    + speeding .docx           │
 │ POST /ingest           ◄┼──── violations.tar.gz ──┤ 4. BUNDLE (tiered manifest)   │
 │   saves bundle+receipt  │                         │ 5. PUSH                       │
 └────────────┬───────────┘                         └──────────────────────────────┘
              ▼
   ~/Desktop/backend_received_evidence/bundle_<ts>.tar.gz   (+ .receipt.json)
```

## What the bundle contains

A top-level `manifest.json` (ranked, see below) plus **one folder per vehicle/violation**
(`v{id}_{TYPE}_f{frame}/`), each self-describing like the reference submission layout:

```
v1_SOLID_LINE_CROSSING_f149/
  clip.mp4                              # the 10s clip (10s pre/5s post), ANNOTATED:
                                        #   red box on the violating vehicle + caption describing it
  v1_SOLID_LINE_CROSSING.txt            # human-readable document (key=value) for THIS violation
  violation.json                        # machine-readable record (same fields the manifest lists)
  v1_SOLID_LINE_CROSSING_0_f149.png     # vehicle picture 1  ┐
  v1_SOLID_LINE_CROSSING_0_f149_plate.png # its plate crop   ├ 3 pictures, each with a plate crop
  v1_SOLID_LINE_CROSSING_1_f150.png     # vehicle picture 2  │
  v1_SOLID_LINE_CROSSING_1_f150_plate.png                   │
  v1_SOLID_LINE_CROSSING_2_f.. .png     # vehicle picture 3  ┘
  ...
v901_SPEEDING_f1200/
  ...same... + report.docx              # SPEEDING also gets a .docx describing the calculated speed
```

The backend **auto-unpacks** each received bundle into `<bundle>_unpacked/` so you can browse this
tree directly (no manual `tar` needed).

Each `violation.txt` / `violation.json` / manifest row carries the agreed payload fields:
`vehicle_id`, `violation`, `plate`, `plate_score`, `n_reads`, `manual_review`,
`crossing_solid_line_confidence` (the crossing confidence on a solid-line row; `null` on others),
plus a human-readable `description`.

## Priority ordering (hard type tiers — lexicographic)

The type dominates absolutely; confidence only orders **within** a tier:

| tier | type                  | ordered within by              |
|-----:|-----------------------|--------------------------------|
| 0    | `SOLID_LINE_CROSSING` | confidence ↓                   |
| 1    | `SPEEDING`            | confidence ↓ (margin-over-limit) |
| 2    | *(any other type)*    | confidence ↓                   |
| 3    | `YELLOW_LINE_RIGHT`   | chronological (no confidence)  |

Plate confidence is **never** a ranking signal — it is evidence only.

## Honesty notes (read before you trust the demo)

- **Crossings are REAL** — the top-3 by confidence from the cached prior analysis of
  `DeNnDugXxP0.mp4` (18 solid-line crossings, 0 plates). No GPU re-run is performed.
- **Speeding + Yellow are SYNTHETIC, clearly flagged** (`"_synthetic": true` in `details`). This
  clip has no GPS (so no speeding) and no yellow-shoulder event, so they are injected purely to
  exercise tiers 1 & 3 and the `.docx`. Pass `--no-synthetic` to ship crossings only.
- The **plate row on the speeding event** (`40-418-79`, score `0.971`, 6 reads) is demo data on the
  synthetic event — the real crossings read `UNKNOWN → manual review`.
- The `.docx` is generated with **stdlib only** (no `python-docx` needed).

---

## Run it — two terminals

**Terminal 1 — start the backend** (serves the video, receives bundles):

```bash
python backend_loop_demo/mock_backend.py \
    --video ~/Desktop/DeNnDugXxP0.mp4 \
    --out-dir ~/Desktop/backend_received_evidence \
    --port 8000
```

> Windows PowerShell: use `$HOME\Desktop\DeNnDugXxP0.mp4` (or an absolute path) for `--video`.

**Terminal 2 — run the brain** (pull → analyse → bundle → push):

```bash
python backend_loop_demo/run_brain.py --base-url http://127.0.0.1:8000
```

You should see, in Terminal 2:

```
[pull]   downloaded DeNnDugXxP0.mp4 -> integrity OK (1280x720 @ 30fps, 2700 frames)
[analyse] 5 violations (3 real crossings + synthetic)
[bundle]  5 violations, ... files, ... bytes
          queue order: [('SOLID_LINE_CROSSING', ...), ('SPEEDING', 901, ...), ('YELLOW_LINE_RIGHT', 951, ...)]
[push]    POST /ingest -> 200 (ok=True)
```

…and in Terminal 1 a `[ingest] saved bundle_<ts>.tar.gz ...` line.

**Verify the drop** (the backend already unpacked it for you):

```bash
ls -la ~/Desktop/backend_received_evidence/
# browse the unpacked per-violation folders:
ls -R ~/Desktop/backend_received_evidence/bundle_*_unpacked/
cat  ~/Desktop/backend_received_evidence/bundle_*_unpacked/v*/v*_*.txt
cat  ~/Desktop/backend_received_evidence/bundle_*.receipt.json
```

## Backend-initiated flow (two servers, async)

The above is *brain-initiated* (you run the brain, it pulls + pushes). For the realistic shape where
the **backend dispatches work to the brain** and the brain calls back, the brain runs as a SERVICE:

```
backend  --POST /jobs {video_name, source_base_url, callback_url, job_id}-->  brain (202 Accepted)
brain    --GET video + meta (pull, fail-fast integrity)------------------->  backend
brain    ...analyse -> annotate -> bundle...
brain    --POST callback_url (the evidence bundle)----------------------->  backend (/ingest)
```

It is asynchronous: the brain returns `202` immediately and calls back when done (real analysis is
minutes of GPU, so the backend never blocks). Three steps:

```bash
# Terminal 1 — backend (serves video, receives bundles, can dispatch)
python backend_loop_demo/mock_backend.py --port 8000

# Terminal 2 — the brain SERVICE (waits for jobs)
python backend_loop_demo/brain_server.py --port 9000

# Terminal 3 — tell the backend to dispatch the job to the brain
curl -X POST http://127.0.0.1:8000/dispatch \
     -H "Content-Type: application/json" \
     -d '{"brain_url": "http://127.0.0.1:9000"}'
```

Then watch the job and the drop:

```bash
# job_id is in the /dispatch response; poll the brain:
curl http://127.0.0.1:9000/jobs/<job_id>
ls -R ~/Desktop/backend_received_evidence/      # the bundle lands here when the brain calls back
```

The bundle the backend receives is correlated to the dispatched job via `job_id` (in the receipt).

## Run it — automatically (one command, in-process server)

```bash
# bash
ROADGUARD_RUN_INTEGRATION=1 pytest backend_loop_demo/test_pipeline_integration.py -s
```
```powershell
# PowerShell
$env:ROADGUARD_RUN_INTEGRATION=1; pytest backend_loop_demo/test_pipeline_integration.py -s
```

Two tests run: the **client-initiated** loop (brain pulls + pushes) and the **backend-initiated**
async loop (backend dispatches to the brain service, brain calls back). Both start their servers on
ephemeral ports, run the whole pipeline, and assert the bundle landed, is tier-ordered, and carries
clip + 3 pics + plate crops + docs (+ `.docx` on the speeder). They are **opt-in** (the env var) so
the normal fast unit-test loop (`pytest violations/tests`) is untouched.
