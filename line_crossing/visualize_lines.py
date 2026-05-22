"""
Extract one frame from a video, run solid-line detection, save result as an image.

Usage (run from project root):
    python line_crossing/visualize_lines.py <video_path> [frame_number]

Example:
    python line_crossing/visualize_lines.py tests_videos/raw_videos/crossing_solid_line/0SmdindPVEY_Crossing_solid_white_line_-_Dashcam.f299.mp4 80

Output: saved as <video_stem>_lines_frame<N>.jpg in the project root.
Red   = solid line (do not cross)
Green = frame centre reference
"""

import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from line_crossing.line_detector import detect_solid_lines


def draw_solid_lines(frame, detected: dict) -> object:
    out = frame.copy()
    h, w = frame.shape[:2]
    # Faint centre reference
    cv2.line(out, (w // 2, 0), (w // 2, h), (0, 200, 0), 1)
    for label, (x1, y1, x2, y2) in detected.items():
        cv2.line(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(out, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    return out


def main():
    if len(sys.argv) < 2:
        print("Usage: python line_crossing/visualize_lines.py <video_path> [frame_number]")
        sys.exit(1)

    video_path = sys.argv[1]
    frame_num  = int(sys.argv[2]) if len(sys.argv) > 2 else 80

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Could not read frame {frame_num} (video has {total} frames)")
        sys.exit(1)

    detected   = detect_solid_lines(frame)
    annotated  = draw_solid_lines(frame, detected)

    out_path = Path(video_path).stem + f"_lines_frame{frame_num}.jpg"
    cv2.imwrite(out_path, annotated)

    print(f"Frame {frame_num}/{total}")
    print(f"Detected solid lines: {list(detected.keys()) or 'none'}")
    for label, coords in detected.items():
        print(f"  {label}: {coords}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
