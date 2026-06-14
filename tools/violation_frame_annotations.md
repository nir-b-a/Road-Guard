# Violation Frame Annotations — confidence-score validation clips

Ground-truth, human-labeled violation windows for the clips downloaded for the
**violation-confidence-score** work. Frame indices are **1-indexed into the trimmed
downloaded clip** (`tests_videos/frames/<id>/f<NNNNNN>.jpg`, where `f000001` is the
first frame of the trimmed segment).

This doc is the **superset** record of every violation we annotated, across **all
violation types**. Only the **solid-white-line crossing** windows are mirrored into
`tools/violation_labels.json` (the only type the crossing harness scores). Other types
(solid-yellow right-lane, traffic-island) are recorded here for **future** logic.

`seconds = (frame - 1) / fps`. For clips with an original-timeline offset, the original
YouTube time = clip seconds + offset.

## Clips

| ID | Source | Trim | Orig offset | Res | FPS | Frames | Plates |
|----|--------|------|-------------|-----|-----|--------|--------|
| DeNnDugXxP0 | youtube.com/watch?v=DeNnDugXxP0 | 0:00–1:30 | none | 1280×720 | 30.00 | 2700 | blurred → N/A |
| 58zh0ZyUTwU | youtube.com/watch?v=58zh0ZyUTwU | 0:10–0:30 | **+0:10** | 1920×1080 | 29.97 | 600 | blurred → N/A |
| mVvcgAA0rJ8 | youtube.com/watch?v=mVvcgAA0rJ8 | 0:00–0:22 | none | 1280×720 | 23.98 | 528 | blurred → N/A |
| Aps9SxIsVl8 | youtube.com/watch?v=Aps9SxIsVl8 | 0:00–0:41 | none | 1920×1080 | 23.98 | 984 | blurred → N/A |

Violation types: `solid_white` = vehicle on/crossing solid white line (scored by
`crossing_violation_test.py`, in `violation_labels.json`). `solid_yellow_right` =
vehicle driving in the right onto/over a solid yellow lane (logic NOT built). `island` =
traffic-island crossing (logic NOT built / different from rule B).

---

## DeNnDugXxP0 (30 fps)

| Frames | Type | In labels.json | Notes |
|--------|------|----------------|-------|
| 68–151 | solid_white | ✅ | visible whole range; model expected to fire ~f68–100 |
| 170–240 | solid_white | ✅ | |
| 255–310 | solid_white | ✅ | |
| 620–680 | solid_white | ✅ | |
| 1320–1380 | solid_white | ✅ | |
| **1440–1465** | **solid_yellow_right** | ❌ (future) | vehicle driving in the right over solid yellow lane. See TODO below. |
| 1880–1939 | solid_white | ✅ | |
| **2318–2400** | **island** | ❌ (future) | traffic-island road crossing |
| 2624–2696 | solid_white | ✅ | |

## mVvcgAA0rJ8 (23.98 fps)

| Frames | Type | In labels.json | Notes |
|--------|------|----------------|-------|
| 70–108 | solid_white | ✅ | |
| 201–221 | solid_white | ✅ | |
| 344–395 | solid_white | ✅ | |
| 500–520 | solid_white | ✅ | |

## Aps9SxIsVl8 (23.98 fps)

| Frames | Type | In labels.json | Notes |
|--------|------|----------------|-------|
| 58–72 | solid_white | ✅ | |
| **388–444** | **island** | ❌ (future) | traffic-island crossing |
| 687–715 | solid_white | ✅ | |
| 788–800 | solid_white | ✅ | |

## 58zh0ZyUTwU (29.97 fps, orig offset +0:10)

No solid-white-line violations → **not added to `violation_labels.json`**. Both events
are future types; frames are clip-relative (1-indexed into the trimmed clip).

| Frames | Type | In labels.json | Notes |
|--------|------|----------------|-------|
| **48–85** | **island** | ❌ (future) | traffic-island crossing |
| **439–541** | **solid_yellow_right** | ❌ (future) | driving in the right of a solid yellow line |

---

## Future violation types (logic not yet built)

### solid_yellow_right — vehicle driving in the right onto a solid yellow lane
- Example: DeNnDugXxP0 f1440–1465.
- Requires **vehicle velocity** (waiting on the velocity signal from teammate's distance/motion work) to confirm the vehicle is moving/driving, not parked.
- Requires confirming the vehicle is **on the right side** — Tal believes this detection is
  already implemented; **TODO: verify the right-side detection logic exists and works.**
- Not scored by the current crossing harness (rule B = solid white only).

### island — traffic-island crossing
- Example: DeNnDugXxP0 f2318–2400.
- Distinct from rule B; needs its own crossing logic against the `traffic_island` class.
