"""
Road speed-limit lookup from GPS, via the public Overpass API (OpenStreetMap).

Given an ego GPS fix (lat, lon, and optionally a travel bearing) this finds the
road the vehicle is on and returns its speed limit in km/h:

    get_speed_limit(lat, lon)                 -> 50            (km/h) or None
    get_speed_limit(lat, lon, bearing=190.0)  -> 90            (direction-aware)
    get_speed_limits_for_track([...])         -> {frame: kmh}  (batch, cached)

Resolution order for the matched road:
    1. explicit `maxspeed` tag  (handles "80", "50 km/h", "30 mph", "IL:urban").
    2. Israel legal default for the road class  (motorway 110, residential 50, ...).

Data source: OpenStreetMap via the *public* Overpass API -- free, no key, ODbL
licensed. It enforces fair-use limits (~10k requests/day, heavy users throttled
first), so we (a) cache by rounded coordinate so repeated/nearby fixes don't
re-query, (b) space real network calls out by a minimum interval, and (c) fail
over between mirror endpoints. For batch / offline processing of many videos,
swap this for a local Geofabrik extract (same OSM data, no rate limit).

NOTE: the Israeli default tables below mirror overpass_api_test.py. They are the
legal fallbacks for *untagged* roads; verify them against current Israeli law and
the OSM "Default speed limits" wiki before relying on them for alerts.
"""

import math
import time
import requests


# ============================================================================
# Configuration
# ============================================================================

# Public Overpass mirrors, tried in order on failure. No API key required.
OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
)

# Overpass etiquette: identify the client.
USER_AGENT = "malshinon-speed-limit/1.0 (student project; OSM Overpass)"

# OSM highway classes we treat as drivable roads (others -- footway, cycleway,
# path, steps -- are ignored so a sidewalk near the car can't win the match).
DRIVABLE = frozenset({
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "living_street", "service",
    "motorway_link", "trunk_link", "primary_link",
    "secondary_link", "tertiary_link",
})

# Implicit, country-coded maxspeed tags -> km/h. Generic "CC:NN" (e.g. "IL:30")
# is parsed numerically, so only the *named* zones need to live here.
ZONE_LIMITS = {
    "IL:urban": 50,
    "IL:rural": 90,
    "IL:trunk": 90,
    "IL:motorway": 110,
}

# Israel legal defaults by road class, used ONLY when a road has no maxspeed tag.
# Anything not listed falls back to URBAN_DEFAULT (mirrors overpass_api_test.py).
ISRAEL_DEFAULTS = {
    "motorway": 110,
    "trunk": 90,
    "primary": 80,
    "secondary": 80,
    "residential": 50,
    "living_street": 30,
}
URBAN_DEFAULT = 50          # km/h, used when the road class isn't in the table

# Matching parameters.
DEFAULT_RADIUS_M = 30.0     # search this far around the GPS fix for a road
DIRECTION_WEIGHT_M = 60.0   # how much a wrong travel direction "costs" in metres;
                            # big enough to push the opposite carriageway behind
                            # the correct one even when it's a touch closer.

# Politeness / robustness.
HTTP_TIMEOUT_S = 25
MIN_REQUEST_INTERVAL_S = 1.0   # minimum spacing between real network requests
MAX_RETRIES_PER_ENDPOINT = 2
CACHE_COORD_DECIMALS = 4       # ~11 m cache granularity
MPH_TO_KMH = 1.609344

# Module-level query cache + throttle state.
_CACHE: dict[tuple, int | None] = {}
_last_request_ts = 0.0

# Circuit breaker: the public Overpass mirrors are sometimes slow/down. After
# this many CONSECUTIVE total failures (all endpoints exhausted) we stop hitting
# the network for the rest of the process, so a batch can't hang on a dead API.
# Reset by clear_cache(). A single success re-arms it.
_FAILURE_LIMIT = 3
_consecutive_failures = 0
_overpass_down = False


# ============================================================================
# Geometry helpers (small distances -> equirectangular metres is plenty)
# ============================================================================

_EARTH_R = 6_371_000.0


def _local_xy(lat: float, lon: float, lat0: float) -> tuple[float, float]:
    """Equirectangular metres for (lat, lon) about origin latitude lat0."""
    x = math.radians(lon) * _EARTH_R * math.cos(math.radians(lat0))
    y = math.radians(lat) * _EARTH_R
    return x, y


def _point_segment_dist_m(plat, plon, alat, alon, blat, blon) -> float:
    """Distance (m) from point P to segment A-B, all in lat/lon."""
    px, py = _local_xy(plat, plon, plat)
    ax, ay = _local_xy(alat, alon, plat)
    bx, by = _local_xy(blat, blon, plat)
    abx, aby = bx - ax, by - ay
    ab2 = abx * abx + aby * aby
    if ab2 == 0.0:
        t = 0.0
    else:
        t = ((px - ax) * abx + (py - ay) * aby) / ab2
        t = max(0.0, min(1.0, t))
    cx, cy = ax + t * abx, ay + t * aby
    return math.hypot(px - cx, py - cy)


def _bearing_deg(lat1, lon1, lat2, lon2) -> float:
    """Initial bearing from point 1 to point 2 (degrees, 0=North, clockwise)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return math.degrees(math.atan2(y, x)) % 360.0


def _angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings, in [0, 180] degrees."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def _way_distance_and_bearing(plat, plon,
                              geometry: list[tuple[float, float]]
                              ) -> tuple[float, float | None]:
    """Closest distance (m) from P to a way polyline, and that segment's bearing."""
    best_dist = float("inf")
    best_brg: float | None = None
    for (alat, alon), (blat, blon) in zip(geometry, geometry[1:]):
        d = _point_segment_dist_m(plat, plon, alat, alon, blat, blon)
        if d < best_dist:
            best_dist = d
            best_brg = _bearing_deg(alat, alon, blat, blon)
    return best_dist, best_brg


# ============================================================================
# Tag parsing
# ============================================================================

_NON_NUMERIC = {"none", "no", "signals", "variable", "walk", "unknown"}


def _parse_maxspeed(value: str | None) -> int | None:
    """
    OSM `maxspeed` string -> km/h int, or None if absent / unlimited / unparseable.

    Handles: "80", "50 km/h", "30 mph", named zones ("IL:urban"), and generic
    country-coded numerics ("IL:30").
    """
    if not value:
        return None
    value = value.strip()

    if value in ZONE_LIMITS:                 # "IL:urban" etc.
        return ZONE_LIMITS[value]
    if ":" in value:                          # generic "CC:NN" -> NN ("IL:30")
        tail = value.split(":")[-1]
        if tail.isdigit():
            return int(tail)

    v = value.lower()
    if v in _NON_NUMERIC:                      # "none" (unlimited) etc.
        return None

    digits = ""                               # leading number of "30 mph" / "50 km/h"
    for ch in v:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    if not digits:
        return None

    kmh = int(digits)
    if "mph" in v:
        kmh = round(kmh * MPH_TO_KMH)
    return kmh


def _oneway_direction(tags: dict) -> int:
    """+1 if the way runs in node order, -1 if reversed, 0 if bidirectional."""
    ow = str(tags.get("oneway", "")).strip().lower()
    if ow in ("yes", "true", "1"):
        return 1
    if ow in ("-1", "reverse"):
        return -1
    # Motorways are implicitly oneway in OSM.
    if tags.get("highway") in ("motorway", "motorway_link"):
        return 1
    return 0


def _direction_penalty(travel_bearing: float,
                       seg_bearing: float,
                       oneway: int) -> float:
    """
    How badly the road's allowed travel direction disagrees with where the
    vehicle is heading, in degrees [0, 180]. 0 = aligned.

    For a oneway road only the legal direction counts (so the opposite
    carriageway of a divided highway gets a large penalty). For a two-way road
    either direction is fine, so we compare to the line modulo 180 degrees.
    """
    if oneway == 1:
        return _angle_diff(travel_bearing, seg_bearing)
    if oneway == -1:
        return _angle_diff(travel_bearing, (seg_bearing + 180.0) % 360.0)
    return min(_angle_diff(travel_bearing, seg_bearing),
               _angle_diff(travel_bearing, (seg_bearing + 180.0) % 360.0))


# ============================================================================
# Overpass query
# ============================================================================

def _throttle() -> None:
    """Block until at least MIN_REQUEST_INTERVAL_S has passed since the last call."""
    global _last_request_ts
    wait = MIN_REQUEST_INTERVAL_S - (time.time() - _last_request_ts)
    if wait > 0:
        time.sleep(wait)
    _last_request_ts = time.time()


def _run_overpass(query: str, timeout: int) -> list[dict]:
    """
    Run one Overpass query string, returning its ways (with inline geometry).

    Empty list on total failure (all endpoints exhausted). Fails over between
    mirrors and retries transient errors (429 rate-limit / 5xx / timeout). A
    circuit breaker (_overpass_down) short-circuits once the API looks dead, so a
    batch of lookups can't spend minutes re-failing against an unreachable server.
    """
    global _consecutive_failures, _overpass_down

    if _overpass_down:                # API already declared dead this run -> skip fast
        return []

    headers = {"User-Agent": USER_AGENT}
    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(MAX_RETRIES_PER_ENDPOINT):
            _throttle()
            try:
                resp = requests.get(endpoint, params={"data": query},
                                    headers=headers, timeout=timeout)
                if resp.status_code in (429, 502, 503, 504):
                    time.sleep(2.0 * (attempt + 1))      # back off, then retry
                    continue
                resp.raise_for_status()
                _consecutive_failures = 0                 # success re-arms the breaker
                return _parse_elements(resp.json())
            except (requests.RequestException, ValueError):
                continue                                  # try next attempt/endpoint

    _consecutive_failures += 1
    if _consecutive_failures >= _FAILURE_LIMIT:
        _overpass_down = True
        print(f"[speed-limit] Overpass unreachable {_consecutive_failures}x in a row "
              f"-> giving up speed-limit lookups for this run")
    else:
        print("[speed-limit] all Overpass endpoints failed")
    return []


def _query_overpass(lat: float, lon: float, radius_m: float,
                    timeout: int) -> list[dict]:
    """Drivable ways within radius_m of one GPS fix (single-point query)."""
    query = (
        f"[out:json][timeout:{timeout}];"
        f'way(around:{radius_m},{lat},{lon})["highway"];'
        f"out geom;"
    )
    return _run_overpass(query, timeout)


def _query_overpass_bbox(min_lat: float, min_lon: float,
                         max_lat: float, max_lon: float,
                         timeout: int) -> list[dict]:
    """Every drivable way inside a bounding box, in ONE query (for whole-track lookups)."""
    query = (
        f"[out:json][timeout:{timeout}];"
        f'way({min_lat},{min_lon},{max_lat},{max_lon})["highway"];'
        f"out geom;"
    )
    return _run_overpass(query, timeout)


def _parse_elements(data: dict) -> list[dict]:
    """Pull ways with inline geometry out of an Overpass JSON response."""
    ways = []
    for el in data.get("elements", []):
        if el.get("type") != "way":
            continue
        geom = [(g["lat"], g["lon"]) for g in el.get("geometry", [])]
        ways.append({"tags": el.get("tags", {}), "geometry": geom})
    return ways


# ============================================================================
# Selection + public API
# ============================================================================

def _resolve(lat: float, lon: float, bearing: float | None,
             ways: list[dict], max_dist_m: float | None = None) -> int | None:
    """Pick the best-matching way and return its speed limit (km/h) or None.

    When `max_dist_m` is given (the bbox-batch path, where `ways` aren't
    pre-filtered by an `around` query), a match whose nearest point is farther
    than that is rejected -- preserving the single-point query's "no road within
    radius -> None" behaviour.
    """
    best_score = float("inf")
    best_dist = float("inf")
    best_tags: dict | None = None
    best_class: str | None = None

    for way in ways:
        tags = way["tags"]
        hwy = tags.get("highway")
        if hwy not in DRIVABLE:
            continue
        geom = way["geometry"]
        if len(geom) < 2:
            continue

        dist, seg_brg = _way_distance_and_bearing(lat, lon, geom)
        penalty = 0.0
        if bearing is not None and seg_brg is not None:
            penalty = _direction_penalty(bearing, seg_brg, _oneway_direction(tags))

        score = dist + (penalty / 180.0) * DIRECTION_WEIGHT_M
        if score < best_score:
            best_score, best_dist, best_tags, best_class = score, dist, tags, hwy

    if best_tags is None:
        return None
    if max_dist_m is not None and best_dist > max_dist_m:
        return None

    explicit = _parse_maxspeed(best_tags.get("maxspeed"))
    if explicit is not None:
        return explicit
    return ISRAEL_DEFAULTS.get(best_class, URBAN_DEFAULT)


def get_speed_limit(lat: float, lon: float,
                    bearing: float | None = None,
                    radius_m: float = DEFAULT_RADIUS_M,
                    *,
                    use_cache: bool = True,
                    timeout: int = HTTP_TIMEOUT_S) -> int | None:
    """
    Speed limit (km/h) of the road at a GPS fix, or None if no road is found.

    Args:
        lat, lon:  GPS position (WGS84 degrees).
        bearing:   optional travel direction (degrees, 0=North, clockwise). When
                   given, divided-highway carriageways and parallel roads are
                   disambiguated by direction. Omit it (or pass None) if you only
                   have position -- matching then falls back to nearest road.
        radius_m:  search radius around the fix.
        use_cache: reuse results for coordinates seen before (~11 m granularity).
        timeout:   per-request HTTP timeout (also the Overpass server timeout).

    Returns:
        Speed limit in km/h (explicit tag, else Israeli class default), or None
        when no drivable road lies within `radius_m` / the API is unreachable.
    """
    key = None
    if use_cache:
        key = (round(lat, CACHE_COORD_DECIMALS),
               round(lon, CACHE_COORD_DECIMALS),
               None if bearing is None else round(bearing / 45.0) % 8)
        if key in _CACHE:
            return _CACHE[key]

    ways = _query_overpass(lat, lon, radius_m, timeout)
    result = _resolve(lat, lon, bearing, ways)

    if use_cache and key is not None:
        _CACHE[key] = result
    return result


def get_speed_limits_for_track(points, **kwargs) -> dict[int, int | None]:
    """
    Speed limit per frame for a GPS track.

    Args:
        points:  iterable of (frame, lat, lon) or (frame, lat, lon, bearing).
        kwargs:  forwarded to get_speed_limit (radius_m, use_cache, timeout).

    Returns:
        {frame -> speed_limit_kmh or None}. Caching makes repeated/nearby fixes
        (slow or stationary stretches) essentially free.
    """
    out: dict[int, int | None] = {}
    for p in points:
        frame, lat, lon = p[0], p[1], p[2]
        bearing = p[3] if len(p) > 3 else None
        out[frame] = get_speed_limit(lat, lon, bearing, **kwargs)
    return out


def get_speed_limits_sampled(points, *, step: int = 25, **kwargs) -> list[int | None]:
    """
    Speed limits along a drive, SAMPLING the GPS track instead of querying every fix.

    `points` is the SAME iterable get_speed_limits_for_track takes -- (frame, lat, lon)
    or (frame, lat, lon, bearing). GPS is logged at several Hz, so consecutive fixes sit
    on the same road a few metres apart; we look up only every `step`-th point (the last
    fix is always included so the end of the drive is represented). The real work --
    per-point Overpass lookup, the ~11 m coordinate cache, direction handling and the
    Israeli class defaults -- is delegated to get_speed_limit via get_speed_limits_for_track,
    so nothing here is duplicated.

    Returns the speed limit (km/h, or None where no road / API was unreachable) for each
    SAMPLED point, in track order.

    Design note: this is a thin WRAPPER, not a refactor. get_speed_limits_for_track already
    iterates points and is cache-backed; "sample the track" only changes WHICH points are
    queried, so subsetting the input and reusing that function is the smallest correct change.
    """
    pts = list(points)
    if not pts:
        return []
    if step < 1:
        step = 1

    sampled = pts[::step]
    if (len(pts) - 1) % step != 0:          # keep the final fix even if step skipped it
        sampled.append(pts[-1])

    limits_by_frame = get_speed_limits_for_track(sampled, **kwargs)
    return [limits_by_frame[p[0]] for p in sampled]


def get_track_speed_limits(points,
                           *,
                           radius_m: float = DEFAULT_RADIUS_M,
                           timeout: int = HTTP_TIMEOUT_S,
                           bbox_margin_deg: float = 0.002) -> dict[int, int | None]:
    """
    Speed limit per frame for a WHOLE GPS track, using ONE Overpass query.

    Same input as get_speed_limits_for_track -- (frame, lat, lon) or
    (frame, lat, lon, bearing). Instead of a network request per point, this
    fetches every drivable way in the track's bounding box once, then resolves
    each frame's limit LOCALLY against that road set. That turns hundreds of slow
    public-API calls into a single one (the fix for long clips appearing to hang).

    Args:
        radius_m:         a frame with no road within this distance gets None.
        timeout:          per-request HTTP/Overpass timeout.
        bbox_margin_deg:  padding added around the track's lat/lon extent (~0.002
                          deg ~= 220 m) so roads just off the path are included.

    Returns:
        {frame -> speed_limit_kmh or None}. If the single bbox query fails (or the
        circuit breaker has tripped), every frame maps to None.
    """
    pts = [p for p in points]
    if not pts:
        return {}

    lats = [p[1] for p in pts]
    lons = [p[2] for p in pts]
    m = bbox_margin_deg
    ways = _query_overpass_bbox(min(lats) - m, min(lons) - m,
                                max(lats) + m, max(lons) + m, timeout)

    out: dict[int, int | None] = {}
    local: dict[tuple, int | None] = {}     # dedup near-identical fixes (cheap, no network)
    for p in pts:
        frame, lat, lon = p[0], p[1], p[2]
        bearing = p[3] if len(p) > 3 else None
        if not ways:
            out[frame] = None
            continue
        key = (round(lat, CACHE_COORD_DECIMALS),
               round(lon, CACHE_COORD_DECIMALS),
               None if bearing is None else round(bearing / 45.0) % 8)
        if key not in local:
            local[key] = _resolve(lat, lon, bearing, ways, max_dist_m=radius_m)
        out[frame] = local[key]
    return out


def clear_cache() -> None:
    """Drop the in-memory query cache and re-arm the Overpass circuit breaker."""
    global _consecutive_failures, _overpass_down
    _CACHE.clear()
    _consecutive_failures = 0
    _overpass_down = False


# ============================================================================
# Manual smoke test
# ============================================================================

if __name__ == "__main__":
    print("no bearing :", get_speed_limit(32.0668, 34.9049))
    #print("no bearing :", get_speed_limit(32.0677, 34.9032, bearing=0.0))
    #print("heading N  :", get_speed_limit(32.091, 34.864, bearing=0.0))
    #print("heading S  :", get_speed_limit(32.091, 34.864, bearing=180.0))
