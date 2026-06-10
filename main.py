"""
Step 8 (polished): end-to-end violation pipeline with init-zoom + dashboard fixes.

  * LaneDetector  — classical CV pipeline with temporal smoothing + dead reckoning
  * YOLO (yolov8m) — vehicle tracking with persistent IDs
  * CrossingMonitor — flags vehicles whose bbox intersects a solid lane line
  * Spatial gate — drops YOLO boxes whose bottom edge sits below y=1000
                   (filters out the ego-vehicle dashboard/hood detection)
  * Writer hard-locked to 1920x1080, frame.copy() used as render canvas
"""

from pathlib import Path

import cv2
from ultralytics import YOLO

from Constants import DetectClass
from line_crossing.line_detector import LaneDetector
from line_crossing.crossing_detector import CrossingMonitor


INPUT_VIDEO = Path(
    "tests_videos/raw_videos/crossing_solid_line/"
    "0SmdindPVEY_Crossing_solid_white_line_-_Dashcam.f299.mp4"
)
OUTPUT_VIDEO = Path("final_polished_test.avi")
MAX_FRAMES   = 300

OUTPUT_WIDTH  = 1920
OUTPUT_HEIGHT = 1080

YOLO_WEIGHTS = "yolov8m.pt"
YOLO_CONF    = 0.5

# Drop any YOLO bbox whose bottom edge sits below y=1000 — that's where the
# ego-vehicle dashboard/hood sits in this 1080p mount, and YOLO sometimes
# hallucinates a "car" on those reflections/edges.
EGO_VEHICLE_MIN_Y = 1000

LANE_COLOR_BGR      = (0, 0, 255)
LANE_THICKNESS      = 15
BBOX_COLOR_BGR      = (0, 255, 0)
BBOX_THICKNESS      = 4
VIOLATION_COLOR_BGR = (0, 0, 255)
VIOLATION_THICKNESS = 10


def _filter_ego_vehicle(bbox: tuple[int, int, int, int]) -> bool:
    """Return True if bbox should be kept (i.e. it's NOT the ego dashboard region)."""
    return bbox[3] <= EGO_VEHICLE_MIN_Y


def main():
    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise SystemExit(f"failed to open {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO), fourcc, fps, (OUTPUT_WIDTH, OUTPUT_HEIGHT),
    )
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"failed to open writer for {OUTPUT_VIDEO}")

    print(f"input: {INPUT_VIDEO.name}  @ {fps:.2f} fps")
    print(f"output: {OUTPUT_VIDEO}  (XVID, {OUTPUT_WIDTH}x{OUTPUT_HEIGHT}, up to {MAX_FRAMES} frames)")

    yolo     = YOLO(YOLO_WEIGHTS)
    detector = LaneDetector()
    monitor  = CrossingMonitor()

    written        = 0
    new_violations = 0

    while written < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break

        # Defensive: enforce exact output resolution regardless of source.
        if frame.shape[0] != OUTPUT_HEIGHT or frame.shape[1] != OUTPUT_WIDTH:
            frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))

        # Render canvas is an independent copy so nothing the detectors do can
        # bleed back into the output (even though they don't mutate today).
        canvas = frame.copy()

        try:
            lanes = detector.detect_lanes(frame)
        except Exception as exc:
            print(f"frame {written}: detect_lanes failed ({exc})")
            lanes = {}

        results = yolo.track(
            frame,
            persist=True,
            verbose=False,
            classes=DetectClass.Vehicle_Classes,
            conf=YOLO_CONF,
        )

        vehicles: list[tuple[int, tuple[int, int, int, int]]] = []
        boxes = results[0].boxes if results else None
        if boxes is not None:
            for box in boxes:
                if box.id is None:
                    continue
                vid = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bbox = (x1, y1, x2, y2)
                if not _filter_ego_vehicle(bbox):
                    continue
                vehicles.append((vid, bbox))

        for vid, bbox in vehicles:
            if monitor.update(vid, bbox, lanes):
                new_violations += 1
                print(f"[VIOLATION] vehicle {vid} crossed solid line at frame {written}")

        # ---- render onto canvas (NOT onto frame passed into detectors) ----
        for x1, y1, x2, y2 in lanes.values():
            cv2.line(canvas, (x1, y1), (x2, y2), LANE_COLOR_BGR, LANE_THICKNESS)

        any_violator_visible = False
        for vid, (x1, y1, x2, y2) in vehicles:
            if monitor.is_violator(vid):
                any_violator_visible = True
                cv2.rectangle(
                    canvas, (x1, y1), (x2, y2),
                    VIOLATION_COLOR_BGR, VIOLATION_THICKNESS,
                )
                cv2.putText(
                    canvas, f"VIOLATION  ID {vid}",
                    (x1, max(y1 - 20, 40)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, VIOLATION_COLOR_BGR, 5,
                )
            else:
                cv2.rectangle(
                    canvas, (x1, y1), (x2, y2),
                    BBOX_COLOR_BGR, BBOX_THICKNESS,
                )
                cv2.putText(
                    canvas, f"ID {vid}",
                    (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, BBOX_COLOR_BGR, 2,
                )

        if any_violator_visible:
            # Massive banner stretched across the top of the frame.
            cv2.putText(
                canvas, "!!! SOLID LINE VIOLATION !!!",
                (80, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 3.5, VIOLATION_COLOR_BGR, 10,
            )

        writer.write(canvas)
        written += 1

    cap.release()
    writer.release()

    print(
        f"done: wrote {written} frames, "
        f"{new_violations} unique vehicles flagged for crossing the solid line"
    )


if __name__ == "__main__":
    main()
