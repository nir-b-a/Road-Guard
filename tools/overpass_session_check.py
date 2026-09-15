#!/usr/bin/env python3
"""
overpass_session_check.py -- run ONLY the speed-limit (Overpass) step of the pipeline on one
session, and write down what went in, what came back, and which limit every frame got.

It makes the same calls main.py makes for an Android session, with the same arguments:

    overspeed.build_ego_track(frames.csv, gps.csv)           -> {frame: (lat, lon, bearing)}
    speed_limit_lookup.get_track_speed_limits(track_points)  -> {frame: km/h or None}

and only WRAPS them to record: the HTTP call (query sent, raw body received) and the local road
match (`_resolve`). No pipeline file is modified; the limits in the report are the ones the
pipeline itself returned.

Run from malshinon/:
    python tools/overpass_session_check.py <session_dir>
    python tools/overpass_session_check.py <session_dir> --replay <session_dir>/overpass_check/2_response_1.json

`--replay` feeds a saved Overpass response back in instead of calling the network, so the
matching can be re-checked without querying Overpass again.

Outputs (default <session_dir>/overpass_check/):
    1_track_points.csv     INPUT   the per-frame GPS points handed to the lookup
    1_route.csv            INPUT   the thinned polyline the route query is built from
    2_request_N.txt        INPUT   endpoint + exact Overpass query of HTTP attempt N
    2_response_N.json      OUTPUT  the raw body Overpass returned for attempt N
    2_http_attempts.csv    OUTPUT  one row per attempt: status, seconds, bytes, remark, error
    3_ways.csv             OUTPUT  every way in the response the lookup used
    4_limit_per_frame.csv  DERIVED the limit of every frame + the road it was matched to
    4_limit_segments.csv   DERIVED consecutive frames with the same limit and road
    summary.txt            the above in a few lines (also printed)
"""
import argparse
import csv
import json
import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # malshinon/
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import requests  # noqa: E402

from speed_estimation import ego_yaw, overspeed  # noqa: E402
from speed_estimation import speed_limit_lookup as sll  # noqa: E402


# --------------------------------------------------------------------------- #
# Recording wrappers
# --------------------------------------------------------------------------- #
class _ReplayResponse:
    """Stands in for a requests.Response built from a saved body (--replay)."""
    status_code = 200

    def __init__(self, body: str):
        self.text = body

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        pass


def _key(lat, lon, bearing):
    """The dedup key get_track_speed_limits uses, so frames can be joined to their match."""
    d = sll.CACHE_COORD_DECIMALS
    return (round(lat, d), round(lon, d), None if bearing is None else round(bearing / 45.0) % 8)


class Recorder:
    def __init__(self, out_dir: str, replay_body: str | None = None):
        self.out_dir = out_dir
        self.replay_body = replay_body
        self.attempts: list[dict] = []
        self.ok_json: dict | None = None     # parsed body of the response the lookup used
        self.ways: list[dict] | None = None  # what _run_overpass returned
        self.resolved: dict = {}             # key -> (lat, lon, bearing, max_dist_m, limit)

    def _http(self, method: str, endpoint: str, **kw):
        n = len(self.attempts) + 1
        query = (kw.get("data") or kw.get("params") or {}).get("data", "")
        with open(os.path.join(self.out_dir, f"2_request_{n}.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"{method} {endpoint}\n\n{query}\n")
        row = {"attempt": n, "method": method, "endpoint": endpoint,
               "replay": self.replay_body is not None, "status": "", "seconds": "",
               "bytes": "", "remark": "", "error": ""}
        self.attempts.append(row)

        t0 = time.time()
        try:
            if self.replay_body is not None:
                resp = _ReplayResponse(self.replay_body)
            else:
                resp = getattr(requests, method.lower())(endpoint, **kw)
        except requests.RequestException as e:
            row["seconds"] = round(time.time() - t0, 2)
            row["error"] = f"{type(e).__name__}: {e}"
            raise
        row["seconds"] = round(time.time() - t0, 2)
        row["status"] = resp.status_code
        body = resp.text or ""
        row["bytes"] = len(body.encode("utf-8"))
        with open(os.path.join(self.out_dir, f"2_response_{n}.json"), "w", encoding="utf-8") as fh:
            fh.write(body)
        try:
            parsed = json.loads(body)
        except ValueError:
            row["error"] = "body is not JSON"
        else:
            if isinstance(parsed, dict):
                row["remark"] = parsed.get("remark", "")
                if resp.status_code == 200:
                    self.ok_json = parsed
        return resp

    def install(self) -> None:
        """Patch the lookup module's own names; everything else in it runs unchanged."""
        sll.requests = types.SimpleNamespace(
            get=lambda url, **kw: self._http("GET", url, **kw),
            post=lambda url, **kw: self._http("POST", url, **kw),
            RequestException=requests.RequestException)

        orig_run = sll._run_overpass

        def run_overpass(query, timeout, **kw):
            self.ways = orig_run(query, timeout, **kw)
            return self.ways

        sll._run_overpass = run_overpass

        orig_resolve = sll._resolve

        def resolve(lat, lon, bearing, ways, max_dist_m=None):
            limit = orig_resolve(lat, lon, bearing, ways, max_dist_m=max_dist_m)
            self.resolved[_key(lat, lon, bearing)] = (lat, lon, bearing, max_dist_m, limit)
            return limit

        sll._resolve = resolve


# --------------------------------------------------------------------------- #
# Which road won, and why
# --------------------------------------------------------------------------- #
def _limit_of_way(tags: dict) -> tuple[int, str]:
    """The limit a way yields when matched, and where that number comes from."""
    tag = sll._parse_maxspeed(tags.get("maxspeed"))
    if tag is not None:
        return tag, "maxspeed tag"
    hwy = tags.get("highway")
    if hwy in sll.ISRAEL_DEFAULTS:
        return sll.ISRAEL_DEFAULTS[hwy], f"default for highway={hwy}"
    return sll.URBAN_DEFAULT, f"URBAN_DEFAULT (highway={hwy} not in ISRAEL_DEFAULTS)"


def explain(lat, lon, bearing, ways, max_dist_m) -> dict:
    """Score every way exactly like speed_limit_lookup._resolve and keep the best two, so the
    report can name the winning road, its distance/direction penalty, and the runner-up."""
    cands = []
    for i, way in enumerate(ways):
        if way["tags"].get("highway") not in sll.DRIVABLE or len(way["geometry"]) < 2:
            continue
        dist, seg_brg = sll._way_distance_and_bearing(lat, lon, way["geometry"])
        pen = 0.0
        if bearing is not None and seg_brg is not None:
            pen = sll._direction_penalty(bearing, seg_brg, sll._oneway_direction(way["tags"]))
        cands.append((dist + (pen / 180.0) * sll.DIRECTION_WEIGHT_M, dist, pen, i))
    cands.sort(key=lambda c: c[0])          # stable: equal scores keep way order, like _resolve

    if not cands:
        return {"limit": None, "source": "no drivable road in the response"}
    score, dist, pen, i = cands[0]
    out = {"way": i, "distance_m": dist, "penalty_deg": pen, "score": score}
    if max_dist_m is not None and dist > max_dist_m:
        out.update(limit=None, source=f"best road {dist:.0f} m away > {max_dist_m:.0f} m radius")
    else:
        out["limit"], out["source"] = _limit_of_way(ways[i]["tags"])
    if len(cands) > 1:
        r_score, r_dist, _, r_i = cands[1]
        out.update(runner_up=r_i, runner_up_distance_m=r_dist, runner_up_score=r_score,
                   runner_up_limit=_limit_of_way(ways[r_i]["tags"])[0])
    return out


# --------------------------------------------------------------------------- #
def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:   # -sig: Excel shows Hebrew
        wr = csv.writer(fh)
        wr.writerow(header)
        wr.writerows(rows)


def _r(x, nd=1):
    return "" if x is None else round(x, nd)


def main():
    ap = argparse.ArgumentParser(description="Run the pipeline's Overpass speed-limit step on one "
                                             "session and record its input, output and result.")
    ap.add_argument("session", help="session folder with frames.csv + gps.csv (or a file inside it)")
    ap.add_argument("--out", default=None, help="output folder (default <session>/overpass_check)")
    ap.add_argument("--replay", default=None, metavar="RESPONSE_JSON",
                    help="use this saved Overpass response instead of the network")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")    # road names are Hebrew; don't die on a cp1252 console

    session = os.path.abspath(args.session)
    if not os.path.isdir(session):
        session = os.path.dirname(session)
    frames_csv = os.path.join(session, "frames.csv")
    gps_csv = os.path.join(session, "gps.csv")
    missing = [p for p in (frames_csv, gps_csv) if not os.path.isfile(p)]
    if missing:
        sys.exit(f"[overpass-check] missing {', '.join(missing)} -- the pipeline would skip speeding")
    out_dir = os.path.abspath(args.out or os.path.join(session, "overpass_check"))
    os.makedirs(out_dir, exist_ok=True)
    replay_body = None
    if args.replay:
        with open(args.replay, encoding="utf-8") as fh:
            replay_body = fh.read()

    # ---- 1. INPUT, built exactly as main.py -> overspeed.flag_overspeed_vehicles builds it ----
    ego_track = overspeed.build_ego_track(frames_csv, gps_csv)
    if not ego_track:
        sys.exit("[overpass-check] no GPS track from frames.csv + gps.csv -- no limits would be looked up")
    track_points = [(frame, fix[0], fix[1], fix[2] if len(fix) > 2 else None)
                    for frame, fix in ego_track.items()]
    frame_ts = ego_yaw.load_frame_timestamps(frames_csv)
    t0 = min(frame_ts.values())

    def t_of(frame):
        return (frame_ts[frame] - t0) / 1e9 if frame in frame_ts else None

    with open(gps_csv, encoding="utf-8") as fh:
        n_fixes = sum(1 for _ in csv.DictReader(fh))
    _write_csv(os.path.join(out_dir, "1_track_points.csv"),
               ["frame", "time_s", "lat", "lon", "bearing_deg"],
               [(f, _r(t_of(f), 3), lat, lon, _r(b)) for f, lat, lon, b in sorted(track_points)])
    query_desc = "bounding-box query (GET)"
    if sll.USE_ROUTE_QUERY:
        route, spacing = sll._route_for_track(track_points)
        _write_csv(os.path.join(out_dir, "1_route.csv"), ["point", "lat", "lon"],
                   [(i, lat, lon) for i, (lat, lon) in enumerate(route)])
        corridor = sll.DEFAULT_RADIUS_M + sll.DIRECTION_WEIGHT_M + spacing
        query_desc = (f"route query (POST): polyline of {len(route)} points, spacing {spacing:.0f} m, "
                      f"corridor {corridor:.0f} m")

    # ---- 2. OUTPUT: the same call flag_overspeed_vehicles makes (no extra kwargs) ----------
    rec = Recorder(out_dir, replay_body)
    rec.install()
    sll.clear_cache()                       # what worker_common.reset_speed_limit_lookup does per drive
    started = time.time()
    limits = sll.get_track_speed_limits(track_points)
    elapsed = time.time() - started
    _write_csv(os.path.join(out_dir, "2_http_attempts.csv"),
               ["attempt", "method", "endpoint", "replay", "status", "seconds", "bytes", "remark", "error"],
               [[a[k] for k in ("attempt", "method", "endpoint", "replay", "status", "seconds",
                                "bytes", "remark", "error")] for a in rec.attempts])

    # ---- 3. the ways the lookup worked with -------------------------------------------------
    ways = rec.ways or []
    raw_ways = [el for el in (rec.ok_json or {}).get("elements", []) if el.get("type") == "way"]
    ids_ok = len(raw_ways) == len(ways)

    def osm_id(i):
        return raw_ways[i].get("id", "") if ids_ok and i is not None else ""

    # ---- 4. DERIVED: join every frame to its match ------------------------------------------
    explained, mismatches = {}, 0
    for key, (lat, lon, brg, max_d, limit) in rec.resolved.items():
        ex = explain(lat, lon, brg, ways, max_d)
        if ex["limit"] != limit:
            ex["source"] += f"  [MISMATCH: pipeline returned {limit}]"
            mismatches += 1
        explained[key] = ex

    frame_rows, matched_frames, source_counts = [], {}, {}
    for f, lat, lon, brg in sorted(track_points):
        ex = explained.get(_key(lat, lon, brg)) if ways else None
        ex = ex or {"source": "no ways returned (lookup failed or empty response)"}
        i = ex.get("way")
        tags = ways[i]["tags"] if i is not None else {}
        r_i = ex.get("runner_up")
        r_tags = ways[r_i]["tags"] if r_i is not None else {}
        if i is not None and ex.get("limit") is not None:
            matched_frames[i] = matched_frames.get(i, 0) + 1
        src = ex["source"].split("  [")[0]
        source_counts[src] = source_counts.get(src, 0) + 1
        frame_rows.append([f, _r(t_of(f), 3), lat, lon, _r(brg), limits.get(f), ex["source"],
                           osm_id(i), tags.get("highway", ""), tags.get("name", ""),
                           tags.get("maxspeed", ""), tags.get("oneway", ""),
                           _r(ex.get("distance_m")), _r(ex.get("penalty_deg")), _r(ex.get("score")),
                           osm_id(r_i), r_tags.get("highway", ""), r_tags.get("name", ""),
                           ex.get("runner_up_limit", ""), _r(ex.get("runner_up_distance_m")),
                           _r(ex.get("runner_up_score"))])
    _write_csv(os.path.join(out_dir, "4_limit_per_frame.csv"),
               ["frame", "time_s", "lat", "lon", "bearing_deg", "limit_kmh", "source",
                "osm_way_id", "highway", "name", "maxspeed_tag", "oneway",
                "distance_m", "direction_penalty_deg", "score",
                "runner_up_way_id", "runner_up_highway", "runner_up_name", "runner_up_limit_kmh",
                "runner_up_distance_m", "runner_up_score"], frame_rows)

    way_rows = []
    for i, w in enumerate(ways):
        t = w["tags"]
        drivable = t.get("highway") in sll.DRIVABLE
        lim, src = _limit_of_way(t) if drivable else ("", "never matched (not a drivable highway class)")
        oid = osm_id(i)
        way_rows.append([i, oid, t.get("highway", ""), t.get("name", ""), t.get("name:en", ""),
                         t.get("maxspeed", ""), t.get("oneway", ""), drivable,
                         lim, src, len(w["geometry"]), matched_frames.get(i, 0),
                         f"https://www.openstreetmap.org/way/{oid}" if oid else ""])
    _write_csv(os.path.join(out_dir, "3_ways.csv"),
               ["index", "osm_way_id", "highway", "name", "name_en", "maxspeed_tag", "oneway",
                "drivable", "limit_if_matched_kmh", "limit_source", "n_points", "frames_matched",
                "osm_url"], way_rows)

    segments = []                           # consecutive frames with the same limit + road
    for row in frame_rows:
        sig = (row[5], row[8], row[9], row[6].split("  [")[0])
        if segments and segments[-1]["sig"] == sig:
            segments[-1].update(end_frame=row[0], end_s=row[1], n=segments[-1]["n"] + 1)
        else:
            segments.append({"sig": sig, "start_frame": row[0], "end_frame": row[0],
                             "start_s": row[1], "end_s": row[1], "n": 1})
    _write_csv(os.path.join(out_dir, "4_limit_segments.csv"),
               ["start_frame", "end_frame", "start_s", "end_s", "n_frames", "limit_kmh",
                "highway", "name", "source"],
               [[s["start_frame"], s["end_frame"], s["start_s"], s["end_s"], s["n"], *s["sig"]]
                for s in segments])

    # ---- summary ------------------------------------------------------------------------------
    n_frames = len(track_points)
    n_limit = sum(1 for p in track_points if limits.get(p[0]) is not None)
    lines = [
        f"session   : {session}",
        f"input     : {n_frames} frames with a GPS point (from {n_fixes} GPS fixes); "
        f"bearing used on {sum(1 for p in track_points if p[3] is not None)}",
        f"query     : {query_desc}",
    ]
    if sll.DISABLE_NETWORK:
        lines.append("NOTE      : speed_limit_lookup.DISABLE_NETWORK is True -- no request was made")
    for a in rec.attempts:
        lines.append(f"http #{a['attempt']:<3}: {a['method']} {a['endpoint']} -> "
                     f"{a['status'] or 'no response'} in {a['seconds']} s, {a['bytes'] or 0} bytes"
                     f"{' (replay)' if a['replay'] else ''}{'  ERROR ' + a['error'][:120] if a['error'] else ''}")
        if a["remark"]:
            lines.append(f"  remark  : {a['remark']}  <- Overpass reported a problem; the pipeline ignores this field")
    if not rec.attempts and not sll.DISABLE_NETWORK:
        lines.append("http      : no request made")
    lines += [
        f"ways      : {len(ways)} in the response, {sum(1 for w in ways if w['tags'].get('highway') in sll.DRIVABLE)} drivable"
        + ("" if ids_ok or not ways else "  (could not align OSM ids with the raw response)"),
        f"matched   : {len(rec.resolved)} distinct GPS positions matched locally",
        f"limits    : {n_limit}/{n_frames} frames got a limit, {n_frames - n_limit} got none",
        "sources   : " + ", ".join(f"{k}: {v}" for k, v in sorted(source_counts.items(), key=lambda kv: -kv[1])),
    ]
    if n_limit == 0:
        lines.append("RESULT    : NO frame has a limit -> the pipeline prints 'No vehicles exceeded the "
                     "speed limit' and reports no speeding for this session")
    lines.append(f"segments  : {len(segments)} (limit, road) stretches:")
    for s in segments[:25]:
        lim, hwy, name, src = s["sig"]
        lines.append(f"  {s['start_s']:>8} - {s['end_s']:<8} s  {str(lim) + ' km/h' if lim is not None else 'none':>9}  "
                     f"{hwy or '-'} {name!r}  ({src})")
    if len(segments) > 25:
        lines.append(f"  ... {len(segments) - 25} more in 4_limit_segments.csv")
    if mismatches:
        lines.append(f"check     : MISMATCH on {mismatches} position(s) -- this tool's re-scoring disagrees "
                     f"with _resolve; trust limit_kmh, not the road columns")
    elif rec.resolved:
        lines.append(f"check     : this tool's road re-scoring agrees with the pipeline on all "
                     f"{len(rec.resolved)} positions")
    lines.append(f"elapsed   : {elapsed:.1f} s   outputs -> {out_dir}")

    text = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
