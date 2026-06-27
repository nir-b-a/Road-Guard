"""
Road Guard -- brute-force / genetic search for the best confidence-scoring formula.

Searches the space of mathematical formulas conf = f(distance, angle_off) in [0,1] and
returns the one that maximizes the objective (maximized IN-SAMPLE by design, as intended):

    S = SUM(conf over TP)  -  0.1 * SUM(conf over FP)

Mechanism: a lightweight genetic-programming loop over random expression trees built from
numpy primitives (sigmoid, exp, log1p, sin, tanh, powers, +-*/, min/max). Every formula is
evaluated VECTORIZED over the whole dataset, so 10k-100k formulas score in seconds. A string
memo skips re-evaluating duplicate formulas; elites carry their fitness between generations.

Run (dummy data, instant):   python tools/formula_search.py
Plug in real cached samples:  python tools/formula_search.py --real outputs/violation_compare/confidence_samples.json --normalize
Crank the search:             python tools/formula_search.py --pop 4000 --gens 40 --seed 7
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np

# --------------------------------------------------------------------------- #
# 1. DUMMY DATA  (53 TP + 240 FP)  -- replace with --real to use your cache
# --------------------------------------------------------------------------- #
def make_dummy(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """53 TP + 240 FP with two features in ~[0,1]. TP cars sit closer + more centred,
    so the two classes partially separate -- just like the real data."""
    rng = np.random.default_rng(seed)
    # TP: small distance (close), small angle_off (centred)
    d_tp = np.clip(rng.normal(0.30, 0.15, 53), 0, 1)
    a_tp = np.clip(np.abs(rng.normal(0.00, 0.20, 53)), 0, 1)
    # FP: larger distance (far), larger angle_off (off-centre)
    d_fp = np.clip(rng.normal(0.60, 0.20, 240), 0, 1)
    a_fp = np.clip(np.abs(rng.normal(0.40, 0.25, 240)), 0, 1)
    distance = np.concatenate([d_tp, d_fp])
    angle_off = np.concatenate([a_tp, a_fp])
    is_tp = np.concatenate([np.ones(53, bool), np.zeros(240, bool)])
    return distance, angle_off, is_tp


def load_real(path: str, normalize: bool):
    """Load [{'feat': {'distance':..,'angle_off':..}, 'tp': bool}, ...] (confidence_samples.json).
    Returns (distance, angle_off, is_tp, norm) where norm carries the min/max used so the SAME
    normalization can be reproduced at render time (None when --normalize is off)."""
    with open(path) as fh:
        rows = json.load(fh)
    distance = np.array([r["feat"]["distance"] for r in rows], float)
    angle_off = np.array([r["feat"]["angle_off"] for r in rows], float)
    is_tp = np.array([bool(r["tp"]) for r in rows])
    norm = None
    if normalize:                                   # min-max each feature to [0,1]
        norm = {"distance": [float(distance.min()), float(distance.max())],
                "angle_off": [float(angle_off.min()), float(angle_off.max())]}
        distance = (distance - norm["distance"][0]) / (norm["distance"][1] - norm["distance"][0] + 1e-12)
        angle_off = (angle_off - norm["angle_off"][0]) / (norm["angle_off"][1] - norm["angle_off"][0] + 1e-12)
    return distance, angle_off, is_tp, norm


# --------------------------------------------------------------------------- #
# 2. PRIMITIVES  (all numerically guarded: no nan/inf escapes)
# --------------------------------------------------------------------------- #
def _sig(x):  return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

UNARY = {
    "neg":    lambda x: -x,
    "sig":    _sig,
    "sigd":   lambda x: 1.0 - _sig(x),                       # sigmoid going DOWN
    "expneg": lambda x: np.exp(-np.clip(np.abs(x), 0, 50)),  # decay in (0,1]
    "gauss":  lambda x: np.exp(-np.clip(x * x, 0, 50)),      # bump in (0,1]
    "log1p":  lambda x: np.log1p(np.abs(x)),
    "sin":    lambda x: np.sin(x),
    "tanh":   lambda x: np.tanh(x),
    "sq":     lambda x: np.clip(x * x, -1e6, 1e6),
    "sqrt":   lambda x: np.sqrt(np.abs(x)),
    "abs":    lambda x: np.abs(x),
}
BINARY = {
    "add":  lambda x, y: x + y,
    "sub":  lambda x, y: x - y,
    "mul":  lambda x, y: x * y,
    "min":  lambda x, y: np.minimum(x, y),
    "max":  lambda x, y: np.maximum(x, y),
    "pdiv": lambda x, y: x / (np.abs(y) + 1e-3),             # protected division
}
UNARY_KEYS, BINARY_KEYS = list(UNARY), list(BINARY)

UNARY_STR = {
    "neg": "(-{0})", "sig": "sigmoid({0})", "sigd": "sigmoid_down({0})",
    "expneg": "exp(-|{0}|)", "gauss": "gauss({0})", "log1p": "log1p(|{0}|)",
    "sin": "sin({0})", "tanh": "tanh({0})", "sq": "({0})^2", "sqrt": "sqrt(|{0}|)",
    "abs": "|{0}|",
}
BINARY_STR = {
    "add": "({0} + {1})", "sub": "({0} - {1})", "mul": "({0} * {1})",
    "min": "min({0}, {1})", "max": "max({0}, {1})", "pdiv": "({0} / (|{1}|+1e-3))",
}

# tree node forms:  ('d',) ('a',) ('c',value) ('u',op,child) ('b',op,left,right)


def rand_const() -> float:
    r = random.random()
    if r < 0.40: return round(random.uniform(0, 1), 3)       # feature-scale offsets
    if r < 0.70: return round(random.uniform(-5, 5), 3)
    if r < 0.90: return round(random.uniform(-30, 30), 2)    # steep-sigmoid gains
    return float(random.randint(0, 5))


def random_tree(depth: int):
    if depth <= 0 or random.random() < 0.30:                 # grow a terminal
        if random.random() < 0.70:
            return ("d",) if random.random() < 0.5 else ("a",)
        return ("c", rand_const())
    if random.random() < 0.40:                               # unary node
        return ("u", random.choice(UNARY_KEYS), random_tree(depth - 1))
    return ("b", random.choice(BINARY_KEYS),                 # binary node
            random_tree(depth - 1), random_tree(depth - 1))


def ev(n, d, a):
    k = n[0]
    if k == "d": return d
    if k == "a": return a
    if k == "c": return np.full_like(d, n[1])
    if k == "u": return UNARY[n[1]](ev(n[2], d, a))
    return BINARY[n[1]](ev(n[2], d, a), ev(n[3], d, a))


def to_str(n) -> str:
    k = n[0]
    if k == "d": return "distance"
    if k == "a": return "angle_off"
    if k == "c": return str(n[1])
    if k == "u": return UNARY_STR[n[1]].format(to_str(n[2]))
    return BINARY_STR[n[1]].format(to_str(n[2]), to_str(n[3]))


def size(n) -> int:
    k = n[0]
    if k in ("d", "a", "c"): return 1
    if k == "u": return 1 + size(n[2])
    return 1 + size(n[2]) + size(n[3])


def all_nodes(n) -> list:
    if n[0] == "u": return [n] + all_nodes(n[2])
    if n[0] == "b": return [n] + all_nodes(n[2]) + all_nodes(n[3])
    return [n]


def _rebuild(n, ctr, target, f):
    """Replace the subtree at pre-order index `target` with f(subtree)."""
    idx = ctr[0]; ctr[0] += 1
    if idx == target:
        ctr[0] = idx + size(n)                               # reserve descendant indices
        return f(n)
    if n[0] == "u": return ("u", n[1], _rebuild(n[2], ctr, target, f))
    if n[0] == "b":
        l = _rebuild(n[2], ctr, target, f)
        r = _rebuild(n[3], ctr, target, f)
        return ("b", n[1], l, r)
    return n


def replace_random(tree, f):
    return _rebuild(tree, [0], random.randrange(size(tree)), f)


# --------------------------------------------------------------------------- #
# 3. GENETIC OPERATORS
# --------------------------------------------------------------------------- #
MAX_NODES = 35


def _cap(tree):
    while size(tree) > MAX_NODES:                            # hoist a subtree to fight bloat
        tree = random.choice(all_nodes(tree))
    return tree


def mutate(tree):
    r = random.random()
    if r < 0.55:                                             # subtree mutation
        out = replace_random(tree, lambda _: random_tree(random.randint(1, 3)))
    elif r < 0.85:                                           # constant jitter
        out = replace_random(tree, lambda s: ("c", round(s[1] * (1 + random.gauss(0, 0.3)) + random.gauss(0, 0.2), 3))
                             if s[0] == "c" else s)
    else:                                                    # hoist (shrink)
        out = random.choice(all_nodes(tree))
    return _cap(out)


def crossover(a, b):
    donor = random.choice(all_nodes(b))
    return _cap(replace_random(a, lambda _: donor))


# --------------------------------------------------------------------------- #
# 4. FITNESS  (the objective S)
# --------------------------------------------------------------------------- #
def confidence(tree, d, a) -> np.ndarray:
    with np.errstate(all="ignore"):
        v = ev(tree, d, a)
    v = np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(v, 0.0, 1.0)


def score(tree, d, a, is_tp) -> float:
    try:
        c = confidence(tree, d, a)
        s = float(c[is_tp].sum() - 0.1 * c[~is_tp].sum())
        return s if np.isfinite(s) else -1e9
    except Exception:
        return -1e9


def apply_saved(saved: dict, distance, angle_off) -> np.ndarray:
    """Evaluate a saved winning formula on RAW distance/angle_off -> confidence in [0,1].
    Re-applies the exact min/max normalization the search used, so render scores match the search."""
    d = np.asarray(distance, float)
    a = np.asarray(angle_off, float)
    norm = saved.get("normalize")
    if norm:
        d = (d - norm["distance"][0]) / (norm["distance"][1] - norm["distance"][0] + 1e-12)
        a = (a - norm["angle_off"][0]) / (norm["angle_off"][1] - norm["angle_off"][0] + 1e-12)
    return confidence(saved["formula"], d, a)


# --------------------------------------------------------------------------- #
# 5. SEARCH LOOP
# --------------------------------------------------------------------------- #
def search(d, a, is_tp, pop_size, gens, elite, tourn, seed):
    random.seed(seed)
    memo: dict[str, float] = {}

    def fit(tree):
        key = to_str(tree)
        if key not in memo:
            memo[key] = score(tree, d, a, is_tp)
        return memo[key]

    pop = [random_tree(random.randint(2, 4)) for _ in range(pop_size)]
    pop = [(t, fit(t)) for t in pop]
    hof: dict[str, tuple[float, tuple]] = {}                 # string -> (S, tree)

    def remember(t, s):
        hof[to_str(t)] = (s, t)

    for g in range(gens):
        pop.sort(key=lambda p: p[1], reverse=True)
        for t, s in pop[:elite]:
            remember(t, s)
        best = pop[0][1]
        print(f"  gen {g:3d}   best S = {best:7.3f}   unique formulas tried = {len(memo)}")

        def tournament():
            return max(random.sample(pop, tourn), key=lambda p: p[1])[0]

        children = [p[0] for p in pop[:elite]]               # elitism
        while len(children) < pop_size:
            r = random.random()
            if r < 0.55:
                children.append(crossover(tournament(), tournament()))
            elif r < 0.90:
                children.append(mutate(tournament()))
            else:
                children.append(random_tree(random.randint(2, 4)))   # fresh blood
        pop = [(t, fit(t)) for t in children]

    pop.sort(key=lambda p: p[1], reverse=True)
    for t, s in pop[:elite]:
        remember(t, s)
    return hof, len(memo)


# --------------------------------------------------------------------------- #
def report(label, tree, s, d, a, is_tp):
    c = confidence(tree, d, a)
    print(f"\n{label}")
    print(f"  S            = {s:.4f}")
    print(f"  avg TP conf  = {c[is_tp].mean():.4f}")
    print(f"  avg FP conf  = {c[~is_tp].mean():.4f}")
    print(f"  size (nodes) = {size(tree)}")
    print(f"  formula      = {to_str(tree)}")


def main():
    ap = argparse.ArgumentParser(description="Genetic search for the best confidence formula.")
    ap.add_argument("--real", metavar="JSON", help="load confidence_samples.json instead of dummy data")
    ap.add_argument("--normalize", action="store_true", help="min-max each feature to [0,1] (use with --real)")
    ap.add_argument("--pop", type=int, default=2000)
    ap.add_argument("--gens", type=int, default=25)
    ap.add_argument("--elite", type=int, default=30)
    ap.add_argument("--tourn", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", metavar="JSON", help="write the winning formula (+ normalization) to this path")
    args = ap.parse_args()

    if args.real:
        d, a, is_tp, norm = load_real(args.real, args.normalize)
        print(f"[data] real: {args.real}  ({is_tp.sum()} TP / {(~is_tp).sum()} FP)"
              f"{'  [min-max normalized]' if args.normalize else ''}")
    else:
        d, a, is_tp = make_dummy(args.seed)
        norm = None
        print(f"[data] dummy: {is_tp.sum()} TP / {(~is_tp).sum()} FP")

    nTP, nFP = int(is_tp.sum()), int((~is_tp).sum())
    print(f"  reference:  all-conf=1 -> S = {nTP - 0.1 * nFP:.2f}   perfect(TP=1,FP=0) -> S = {nTP:.2f}\n")

    hof, n_unique = search(d, a, is_tp, args.pop, args.gens, args.elite, args.tourn, args.seed)

    ranked = sorted(hof.values(), key=lambda x: x[0], reverse=True)
    print(f"\n=== SEARCH DONE: {n_unique} unique formulas evaluated ===")
    print("\n--- top 5 ---")
    for i, (s, t) in enumerate(ranked[:5], 1):
        c = confidence(t, d, a)
        print(f"  #{i}  S={s:7.3f}  TPconf={c[is_tp].mean():.3f} FPconf={c[~is_tp].mean():.3f}  {to_str(t)}")
    report(">>> WINNER <<<", ranked[0][1], ranked[0][0], d, a, is_tp)

    if args.save:
        best_s, best_t = ranked[0]
        c = confidence(best_t, d, a)
        payload = {"formula": best_t, "formula_str": to_str(best_t), "normalize": norm,
                   "features": ["distance", "angle_off"], "S": best_s,
                   "tp_conf": float(c[is_tp].mean()), "fp_conf": float(c[~is_tp].mean())}
        with open(args.save, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n[saved] winning formula -> {args.save}")


if __name__ == "__main__":
    main()
