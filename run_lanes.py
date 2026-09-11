"""
run_lanes.py -- the LANE half of Road Guard on its own: lane markings + lane violations, no speed.

`main.py <video>` is the speed-estimation entry point. This is its counterpart for the lane
rules: point it at a video (or a folder of videos) and it will

  * detect and track the vehicles,
  * segment every lane marking on every frame with weights/phase3_v3_yellowprotect.pt,
  * flag SOLID-WHITE-LINE CROSSINGS (with the Stage-2 tire confirmation) and
    YELLOW-LINE / SHOULDER driving,
  * read the offender's license plate,
  * write an annotated video WITH THE DETECTED LANES DRAWN ON IT, one annotated clip per
    violation, and the full violation records,

and it will NOT estimate anyone's speed: no speeding rule, no speed plots, and no call to the
OpenStreetMap Overpass API. Nothing is uploaded anywhere and nothing is deleted -- every output
is written next to the input, exactly like worker_offline.py (which this is a shortcut for).

    python run_lanes.py videos\\clips                  # every video in the folder
    python run_lanes.py videos\\clips\\drive1.mp4       # one video
    python run_lanes.py videos\\sessions --force       # redo sessions already processed

Every worker_offline.py flag still applies, e.g. `--out-root`, `--no-bundle`, `--max-frames`,
`--lane-conf 0.35`, `--no-stage2`. See docs/HOW_TO_RUN_LANES.md.

Equivalent to:  python worker_offline.py <path> --lanes-only
"""
import sys

import worker_offline

if __name__ == "__main__":
    if "--lanes-only" not in sys.argv:
        sys.argv.insert(1, "--lanes-only")
    sys.exit(worker_offline.main_cli())
