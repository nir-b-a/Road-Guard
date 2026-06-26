"""
Road Guard -- exhaustive grid search over a SIGMOID-family confidence formula (+ OpenCV demo).

The confidence is STRICTLY the product of two downward sigmoids (no arbitrary formulas, no linear
models). Only the 4 parameters are searched:

    conf(distance, angle_off) = sigmoid_down(distance; d50, k_d) * sigmoid_down(angle_off; a50, k_a)
        sigmoid_down(x; x0, k) = 1 / (1 + exp( k * (x - x0) ))     # 1 for small x, ->0 for large x

Objective (in-sample, by design):

    S = sum_TP conf  -  0.3 * sum_FP conf

The steepness ranges (k_d, k_a) are capped so the curve stays GRADED: a very steep sigmoid
degenerates into a 0/1 step, so capping keeps scores smooth. The search reports the % of scores
stuck at the extremes as proof.

Run:
    python tools/sigmoid_search.py                 # grid search on cached events, print top 5
    python tools/sigmoid_search.py --render outputs/violation_compare/sigmoid_demo.mp4   # + demo
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

DATA_DEFAULT = "outputs/violation_compare/confidence_samples.json"
BEST_DEFAULT = "outputs/violation_compare/sigmoid_best.json"
FP_WEIGHT = 0.3

# grid (exact ranges from the spec)
D50 = (5.0, 60.0, 1.0)      # distance midpoint  (start, stop, step)
K_D = (0.05, 0.40, 0.01)    # distance steepness (capped -> graded)
A50 = (0.10, 0.90, 0.02)    # angle midpoint
K_A = (2.0, 12.0, 0.5)      # angle steepness    (capped -> graded)


# --------------------------------------------------------------------------- #
def _grid(spec):
    lo, hi, step = spec
    return np.arange(lo, hi + step / 2, step)


def sigmoid_down(x, x0, k):
    return 1.0 / (1.0 + np.exp(np.clip(k * (x - x0), -50, 50)))


def conf_vec(distance, angle_off, d50, k_d, a50, k_a):
    return sigmoid_down(distance, d50, k_d) * sigmoid_down(angle_off, a50, k_a)


# --------------------------------------------------------------------------- #
def load_events(path: str):
    with open(path) as fh:
        rows = json.load(fh)
    d = np.array([r["feat"]["distance"] for r in rows], float)
    a = np.array([r["feat"]["angle_off"] for r in rows], float)
    tp = np.array([bool(r["tp"]) for r in rows])
    return d, a, tp


def make_dummy(seed: int = 0):
    """53 TP (closer + centred) + 240 FP (farther + off-centre) in the real feature ranges."""
    rng = np.random.default_rng(seed)
    d_tp = np.clip(rng.normal(14, 8, 53), 5, 60)
    a_tp = np.clip(np.abs(rng.normal(0.0, 0.22, 53)), 0, 1)
    d_fp = np.clip(rng.normal(26, 14, 240), 5, 60)
    a_fp = np.clip(np.abs(rng.normal(0.35, 0.28, 240)), 0, 1)
    return (np.concatenate([d_tp, d_fp]), np.concatenate([a_tp, a_fp]),
            np.concatenate([np.ones(53, bool), np.zeros(240, bool)]))


# --------------------------------------------------------------------------- #
def grid_search(distance, angle_off, is_tp, fp_w=FP_WEIGHT):
    """Vectorized exhaustive sweep. Returns sorted list of (S, d50, k_d, a50, k_a, sumTP, sumFP)."""
    d50g, kdg = (m.ravel() for m in np.meshgrid(_grid(D50), _grid(K_D), indexing="ij"))
    a50g, kag = (m.ravel() for m in np.meshgrid(_grid(A50), _grid(K_A), indexing="ij"))
    Nd, Na = d50g.size, a50g.size

    # distance factor per (d50,k_d) combo per event  ->  DF (Nd, E)
    DF = sigmoid_down(distance[None, :], d50g[:, None], kdg[:, None])
    AF = sigmoid_down(angle_off[None, :], a50g[:, None], kag[:, None])   # (Na, E)

    tp = is_tp.astype(float)
    fp = (~is_tp).astype(float)
    sumTP = (DF * tp) @ AF.T          # (Nd, Na)  = sum over TP of conf
    sumFP = (DF * fp) @ AF.T          # (Nd, Na)  = sum over FP of conf
    S = sumTP - fp_w * sumFP

    print(f"[grid] {Nd} x {Na} = {Nd * Na:,} parameter combinations swept")
    order = np.argsort(S.ravel())[::-1]
    out = []
    for idx in order[:5]:
        i, j = divmod(int(idx), Na)
        out.append((float(S.flat[idx]), float(d50g[i]), float(kdg[i]),
                    float(a50g[j]), float(kag[j]), float(sumTP[i, j]), float(sumFP[i, j])))
    return out


def report(top, distance, angle_off, is_tp):
    nTP, nFP = int(is_tp.sum()), int((~is_tp).sum())
    print(f"\n  reference:  all-conf=1 -> S = {nTP - FP_WEIGHT * nFP:.1f}    perfect(TP=1,FP=0) -> S = {nTP}")
    print(f"\n--- TOP 5 (ranked by S = sum_TP conf - {FP_WEIGHT}*sum_FP conf) ---")
    print(f"  {'#':>2}  {'S':>7}  {'d50':>5} {'k_d':>5} {'a50':>5} {'k_a':>5}   "
          f"{'TPconf':>6} {'FPconf':>6}  {'%stuck@0/1':>10}")
    for n, (s, d50, kd, a50, ka, stp, sfp) in enumerate(top, 1):
        c = conf_vec(distance, angle_off, d50, kd, a50, ka)
        stuck = 100.0 * ((c < 0.01) | (c > 0.99)).mean()
        print(f"  {n:>2}  {s:7.3f}  {d50:5.1f} {kd:5.3f} {a50:5.2f} {ka:5.1f}   "
              f"{stp / max(nTP,1):6.3f} {sfp / max(nFP,1):6.3f}  {stuck:9.1f}%")
    return top[0]


# --------------------------------------------------------------------------- #
def render_demo(params, out_path, fps=30, seconds=9, W=1280, H=720):
    """Synthetic proof-of-gradient clip: a car approaches (distance 60->5) while weaving
    left/right (angle_off 0..0.95). Box color goes green(high)->red(low) and the conf bar +
    live readouts show the score changing SMOOTHLY frame to frame."""
    import cv2
    d50, k_d, a50, k_a = params
    frames = int(fps * seconds)
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for f in range(frames):
        t = f / (frames - 1)
        distance = 60.0 - 55.0 * t                       # approach
        s = float(np.sin(2 * np.pi * 3 * t))             # signed lateral weave
        angle_off = abs(s) * 0.95
        conf = float(conf_vec(np.array([distance]), np.array([angle_off]), d50, k_d, a50, k_a)[0])
        color = (0, int(255 * conf), int(255 * (1 - conf)))   # BGR: green high -> red low

        frame = np.full((H, W, 3), 55, np.uint8)
        for y in range(0, H, 40):                        # dashed center reference
            cv2.line(frame, (W // 2, y), (W // 2, y + 20), (0, 170, 170), 2)

        bh = int(np.interp(distance, [5, 60], [340, 70])); bw = int(bh * 0.85)
        cx = int(W / 2 + s * (W / 2 - 170)); cy = int(H * 0.55)
        cv2.rectangle(frame, (cx - bw // 2, cy - bh // 2), (cx + bw // 2, cy + bh // 2), color, 5)
        cv2.putText(frame, f"conf={conf:.2f}", (cx - bw // 2, max(cy - bh // 2 - 12, 30)),
                    FONT, 1.0, color, 2, cv2.LINE_AA)

        cv2.rectangle(frame, (0, 0), (430, 150), (30, 30, 30), -1)
        cv2.putText(frame, f"distance  = {distance:5.1f} m", (15, 42), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"angle_off = {angle_off:4.2f}", (15, 82), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"conf      = {conf:4.2f}", (15, 122), FONT, 0.9, color, 2, cv2.LINE_AA)

        bx1, by1, bx2, by2 = 40, H - 55, W - 40, H - 28           # confidence bar
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (80, 80, 80), -1)
        cv2.rectangle(frame, (bx1, by1), (int(bx1 + (bx2 - bx1) * conf), by2), color, -1)
        cv2.putText(frame, f"sigmoid: d50={d50:.0f} k_d={k_d:.3f} a50={a50:.2f} k_a={k_a:.1f}",
                    (bx1, by1 - 10), FONT, 0.6, (210, 210, 210), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    print(f"   demo video -> {out_path}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Grid search over the two-sigmoid confidence formula.")
    ap.add_argument("--data", default=DATA_DEFAULT)
    ap.add_argument("--dummy", action="store_true", help="use synthetic data instead of the cached file")
    ap.add_argument("--save", default=BEST_DEFAULT, help="write winning params to this JSON")
    ap.add_argument("--render", metavar="MP4", help="also render the proof-of-gradient demo to this path")
    ap.add_argument("--seconds", type=float, default=9.0)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    if args.dummy or not os.path.isfile(args.data):
        d, a, tp = make_dummy()
        print(f"[data] dummy: {int(tp.sum())} TP / {int((~tp).sum())} FP")
    else:
        d, a, tp = load_events(args.data)
        print(f"[data] {args.data}: {int(tp.sum())} TP / {int((~tp).sum())} FP")

    top = grid_search(d, a, tp)
    s, d50, kd, a50, ka, stp, sfp = report(top, d, a, tp)

    payload = {"form": "sigmoid_down(distance;d50,k_d) * sigmoid_down(angle_off;a50,k_a)",
               "params": {"d50": d50, "k_d": kd, "a50": a50, "k_a": ka},
               "fp_weight": FP_WEIGHT, "S": s,
               "tp_conf": stp / max(int(tp.sum()), 1), "fp_conf": sfp / max(int((~tp).sum()), 1)}
    with open(args.save, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\n[saved] winning params -> {args.save}")

    if args.render:
        print(f"\n[render] proof-of-gradient demo ...")
        render_demo((d50, kd, a50, ka), args.render, fps=args.fps, seconds=args.seconds)


if __name__ == "__main__":
    main()
