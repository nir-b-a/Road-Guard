# Road Guard — System Overview & Design Story

*Detecting traffic violations by other vehicles, from a single Israeli dashcam.*

---

## 1. The mission

Road Guard watches dashcam footage and flags when **another vehicle** — never the camera car —
commits a traffic violation, packages a short evidence clip, and (in deployment) forwards it to
the authorities. It runs on a modest GPU (GTX 1060), uses only classical computer vision on top
of two neural detectors, and is built to be **explainable**: every alert can be traced to a
concrete geometric reason.

The guiding principle throughout is **recall-first**: *missing a real violation is far worse than
raising a false one.* Almost every design decision below falls out of that single rule — but with
one deliberate exception (the "moving" gate, see §6), because falsely accusing a parked car is its
own kind of harm.

---

## 2. Design philosophy

Four commitments shape the whole system:

- **Recall-first.** We never hard-reject a candidate violation on a fragile signal. Mechanisms may
  only *raise the bar* (demand more persistence, deeper overlap), scaled by evidence — so a real
  violation always has a path to fire.
- **Cache-once / sweep-cheap.** The expensive neural inference runs **once** per clip and is cached
  to JSON (`outputs/violation_cache/<clip>.json`: per frame, the vehicles, the lane polygons + class
  + confidence, and the ego motion). Every experiment after that — parameter sweeps, new logic, new
  violation types — runs in *seconds* on the cache, with no GPU. This is why we could iterate so fast.
- **Image-space only.** The lane segmentation model is trained on raw dashcam frames. We refuse to
  warp the input (no bird's-eye / inverse-perspective transform) because that would wreck the
  classifier. All geometry is post-processing on the model's *output*, in the original image.
- **Explainable & cheap.** Polynomials, optical flow, mask arithmetic — nothing a human can't trace,
  nothing that won't run in real time on a 1060.

---

## 3. The pipeline, end to end

```
  frame ─► [YOLOv8-seg lanes]     ─► reconcile fragments ─► per-line IDs ─► tighten ─┐
        └► [YOLOv8 + ByteTrack]   ─► vehicles (bbox, id, contact)                    │
        └► [LK optical flow]      ─► ego shift (dx,dy)                               │
                                                                                     ▼
   surface (solid lines + islands) ◄─ LaneMemory (fills the line under a car) ◄── ghost
                                                                                     │
                       per-vehicle VERDICT (which line, which side, on it?) ◄────────┘
                                                                                     │
        MotionDirectionFilter (oncoming P) ─► dynamic-K + directional far-side shift  │
                                                                                     ▼
                K consecutive frames ─► per-track cooldown ─► EVENT ─► ±5s evidence clip
```

Two neural models do the *seeing*; everything else is classical CV doing the *reasoning*.

---

## 4. The building blocks (and why each exists)

### Lane reconciliation & per-line identity — `ghost_mask.py`
The segmentation model emits noisy, fragmented blobs. `reconcile_components` dilates and
connected-components them into **physical lines**, assigning each a class by majority pixel area.
`LaneLineTracker` then gives every physical line a **persistent ID** across frames, with:
- **class-vote hysteresis** — a historically-white line resists flipping to "yellow" on a few
  low-confidence frames (kills the class-flicker that plagued early versions);
- an **Israeli road prior** — yellow is expected only on the far-right/median edge, so center/left
  yellow votes are heavily discounted;
- an **age** — how long this line has persisted, used for validity gates.

### Polynomial centreline & distance-adaptive tightening
Raw masks are fat and blobby — catastrophically so near the vanishing point, where a 2-px error is
meters on the ground and neighbouring lines merge into mush. `_fit_line` fits a **robust 2nd-degree
centreline** `x = a·y² + b·y + c` (residual-trimmed, so it ignores points where masks merge), and
`_raster_line` redraws a thin ribbon whose **width shrinks toward the horizon**
`W(y)=max(w_min, w_alpha·(y−y_vp))`. This is "Phase 1." Evaluating the curve *locally at a vehicle's
row* makes it robust even when the global fit can't capture a complex S-curve.

### The Ghost Mask — occlusion, take one
A violating vehicle physically **covers the line it's sitting on**, exactly when we need to see it.
`GhostMaskTracker` snapshots the violation surface under a vehicle the moment its tires touch it,
binds that "ghost" to the vehicle's track ID, and carries it forward by compensating for the
camera's motion (Lucas–Kanade optical flow → an affine shift). A wide central "verdict line" then
judges against the ghost while the real paint is hidden. TTL ≈ 0.5 s, so a stale snapshot can't keep
firing forever.

### LaneMemory — occlusion, take two (the ghost, generalized) — `LaneMemory`
The ghost remembers *pixels*; `LaneMemory` remembers each line's **fitted polynomial per ID**, and
re-injects it **only where a vehicle occludes the road**. Its safety is the **negative-update gate**:
the instant a remembered line shows *bare asphalt* (visible, not under a car) with no detection, it
is **hard-killed** — so we never hallucinate a line across a junction, a dashed gap, or where the
paint genuinely ended. (On the white-line violation this turned out inert — see the lesson in §6 —
but it is *the* enabling mechanism for the shoulder violation, where occlusion is intrinsic.)

### MotionDirectionFilter — direction from motion alone — `motion_filter.py`
The hardest false positive is an **oncoming car** in the opposing lane whose tires, in 2-D
projection, land right on the line between us. Static geometry *cannot* tell that apart from a
same-direction car drifting onto the line. So we decide direction from **motion**: per track,
- **ego-compensated vertical velocity** `V_y = ((cy_now − cy_old) − Σdy_ego)/bbox_height` — oncoming
  cars slide down the image faster than the road does;
- a **Y-origin anchor** — oncoming tracks are born near the vanishing point and descend.

These fuse into an **oncoming probability `P ∈ [0,1]`**. Measured on real data, genuine line-drivers
have median `V_y ≈ 0` (they move *with* the road); oncoming traffic sits well positive — clean
separation that geometry alone never gave us.

### The two false-positive layers (recall-safe by construction)
- **Dynamic-K** — a track's required consecutive-hit time stretches as `K·(1+α·P)`. Same-direction
  violators (P≈0) keep the base K; oncoming cars must persist implausibly long, so they drop out.
- **Directional far-side shift** — for an oncoming-looking car, the verdict only counts if the line
  overlaps the car's *far-from-ego* side (a mere near-edge graze doesn't), scaled by P.

Because both scale with P, a genuine same-direction violator is untouched — recall is preserved *by
construction*. Together they took highway false positives from **6.7 → 1.1 per minute at full recall.**

### Side-of-line test
"Which side of a line is a vehicle on?" — evaluate the line polynomial at the car's contact row and
compare the car's centre to it, with a **margin that scales with bbox width** (a free depth proxy, so
the threshold tightens automatically with distance). This test *failed* for "car **on** the white
line" (a line-driver's centre straddles the line ambiguously) — but it is exactly right for "car to
the **right of** the yellow line," where clear one-sidedness *is* the offense.

### Persistence, cooldown, evidence
A violation is an **event**: the verdict must hold **K consecutive frames**; a **3-second per-track
cooldown** then merges flickery repeats into one *incident* (the right unit for a police alert); and
each incident is exported as a **±5 s clip — raw (evidence) + overlaid (operator triage)**.

---

## 5. The violations

Each violation is a *recomposition* of the blocks above — the line **classifier** is what turns a
maneuver into an offense (crossing a dashed line is legal; crossing a solid one is not).

| Violation | How it composes the primitives | Status |
|-----------|-------------------------------|--------|
| **Vehicle on a solid white line** | surface (tightened solid) + ghost/memory + verdict + motion FP-control + K/cooldown/evidence | ✅ validated |
| **Vehicle on a traffic island** | island region (eroded) added to the surface | ✅ |
| **Right-of-yellow shoulder driving** | right-edge yellow line (via LaneMemory, survives the car covering it) + width-scaled side test + looming "moving" gate + oncoming suppression + K=3 s | 🟡 built, needs clips |
| **Wrong-way / oncoming in our lane** | high P + our-lane corridor + close + no-divider-between | 🟡 built, needs clips |

The two 🟡 detectors are **additive modules** (`shoulder_violation.py`, `wrong_way.py`) that touch
no existing logic — they run clean and don't false-fire on clean footage, but their *catch* paths
and thresholds are unvalidated until real clips exist.

---

## 6. What we learned (the honest part)

The most valuable findings were the things that *didn't* work — they told us where the real limits are.

- **Geometry can't determine direction.** A hard "side-of-line" gate cut highway FPs ~50% but
  **deleted a real catch** (a portrait-video violator whose centre sat just past the line, identical
  to an oncoming car). Lesson: for "on the line," *motion* is the only honest direction signal — which
  led to the MotionDirectionFilter, the actual win.
- **The horizon-proximity prior did nothing.** Intuitive ("far overlaps are unreliable"), but the
  data showed our oncoming FPs happen **low in the frame** (close cars), not near the horizon. Killed it.
- **Tightening (Phase 1) and memory (Phase 2) were metric washes — for opposite reasons.** Tightening
  had no FPs left to clean (motion already removed them); memory recovered no recall because our
  remaining misses are **"never-detected," not "occluded."** Both are kept (tightening as the clean
  centreline substrate; memory as real-world occlusion insurance and the enabler of the shoulder
  violation), but neither moved the numbers on *our* clips.
- **We hit the ceiling of image-space post-processing.** False positives are solved (0.9/min); the
  remaining missed violations are **detector blind spots** — the model simply never outputs the line.
  No amount of clever geometry conjures a line that was never detected. The next real gain is on the
  **model** side (dual-threshold re-cache, or relabel→retrain).
- **The "moving" gate is the one place we lean precision, not recall.** Falsely accusing a *parked*
  car (sent to police) is worse than missing a crawler — a deliberate, scoped exception to recall-first.

---

## 7. Results

| metric | naive baseline | + motion stack | + Phase 3 (ship) |
|--------|---------------:|---------------:|-----------------:|
| Highway false positives | 6.7 /min | 1.1 /min | **0.9 /min** |
| Violation recall (TP) | 7.3 | 7.3 | **7.3 (full)** |

The motion + geometry stack is the headline: **~7× fewer false positives at zero recall cost.**
The ship config: `far_bias=0.75, alpha=5, K_base=0.10`, ghost `TTL=0.5 s`, tightening
`w_alpha=0.037 / w_min=5`, `3 s` cooldown, ±5 s evidence clips.

---

## 8. Roadmap

**Now / productionize**
- Wire the validated ship config into `main.py` (it still lives in the `tools/` experiment scripts).

**The real recall lever — the detector**
- Remaining white-line misses are never-detected by the seg model. Fix via **dual-threshold**
  (re-cache at `conf=0.15`, accept faint masks only where they intersect the LaneMemory history) or
  **relabel→retrain**. This pays off across *every* violation type.

**Validate the two new violations**
- Collect clips (shoulder driving + **parked cars**; a wrong-way clip + a **legitimate-oncoming**
  clip), cache them, and tune thresholds the same data-driven way we tuned the motion signal.

**Polish**
- A **debug overlay** that draws the exact tested surface + trigger source per frame — both to nail
  the one unexplained "phantom island" case and as the explainability layer for police evidence.
- Eyeball and dial the line width.

**The merge — with the sibling speed-detector project**
- Lock a per-track `distance(track)` / `velocity(track)` interface so integration is a swap.
- Then: a **confidence score** on every alert (degrades with distance + lateral offset; annotate for
  triage, never suppress); replace the interim looming "moving" gate with **real world-velocity**; and
  unlock **tailgating** and **stopped-in-live-lane**, while consuming **speeding** from the sibling.

**Future violation ideas**
- Lane straddling (riding a dashed line), failure-to-keep-right (needs lane indexing), erratic weaving.

---

*See `docs/ROADGUARD_HANDOFF.md` for the terse, prioritized task list. Run everything with the
`roadguard-dl` conda python.*
