"""Route Telemetry: F1-style corner analysis for NFL route runners.

Every route is treated like a lap and every route break like a corner:
entry speed -> braking -> apex (minimum speed) -> exit acceleration.
All distances are yards, speeds yd/s, accelerations yd/s^2 (or g).
"""
import numpy as np
import pandas as pd

HZ = 10                 # tracking frames per second
G = 10.73               # 1 g in yd/s^2
HALF = 5                # half-width of the corner window (0.5 s)
TRACE = 15              # frames kept either side of the apex for overlays
MIN_TURN_DEG = 35       # below this the route has no real corner (go/seam)
MIN_ENTRY_SPEED = 3.5   # yd/s; ignores standing pivots
SKIP_FRAMES = 8         # ignore release moves in the first 0.8 s

SNAP_EVENTS = {"ball_snap", "autoevent_ballsnap"}
RELEASE_EVENTS = {"pass_forward", "autoevent_passforward"}
END_EVENTS = {"pass_arrived", "pass_outcome_caught", "pass_outcome_incomplete",
              "pass_outcome_interception", "pass_outcome_touchdown",
              "qb_sack", "qb_strip_sack", "tackle", "out_of_bounds"}

CORNER_BINS = [MIN_TURN_DEG, 70, 115, 181]
CORNER_LABELS = ["Sweeper (slant/post)", "90-degree (in/out/dig)", "Hairpin (curl/comeback)"]


def normalize_direction(df):
    """Flip left-moving plays so every offense attacks toward +x."""
    left = df["playDirection"].eq("left").to_numpy()
    df["x"] = np.where(left, 120.0 - df["x"], df["x"])
    df["y"] = np.where(left, 160.0 / 3 - df["y"], df["y"])
    df["dir"] = np.where(left, (df["dir"] + 180.0) % 360.0, df["dir"])
    return df


def _smooth(v, k=3):
    """Centered moving average that keeps the array length."""
    if len(v) < k:
        return v.astype(float)
    pad = k // 2
    padded = np.concatenate([np.repeat(v[0], pad), v, np.repeat(v[-1], pad)])
    return np.convolve(padded, np.ones(k) / k, mode="valid")


def kinematics(s, dir_deg):
    """Speed, heading (rad, unwrapped), longitudinal and lateral acceleration."""
    s = _smooth(np.asarray(s, float))
    heading = _smooth(np.unwrap(np.deg2rad(np.asarray(dir_deg, float))))
    a_lon = np.gradient(s) * HZ                 # speeding up (+) / braking (-)
    a_lat = s * np.gradient(heading) * HZ       # v * omega = cornering force
    return s, heading, a_lon, a_lat


def find_corner(s, heading, a_lon, a_lat):
    """Locate the route break and measure it like a race-car corner.

    Returns a dict of corner metrics, or None when the route has no corner
    (e.g. a go route) or the tracking ends before the exit.
    """
    n = len(s)
    if n < SKIP_FRAMES + 2 * HALF + 1:
        return None
    centres = np.arange(HALF, n - HALF)
    turn = np.abs(heading[2 * HALF:] - heading[:-2 * HALF])   # turn over 1.0 s
    # speed carried into each candidate corner (max over the previous 1.5 s)
    entry = pd.Series(s).rolling(TRACE, min_periods=1).max().to_numpy()[centres]
    ok = (centres >= SKIP_FRAMES) & (entry >= MIN_ENTRY_SPEED)
    if not ok.any():
        return None
    c = centres[ok][np.argmax(turn[ok])]
    if np.rad2deg(turn[c - HALF]) < MIN_TURN_DEG:
        return None

    lo, hi = c - HALF, c + HALF + 1
    apex = lo + int(np.argmin(s[lo:hi]))
    if apex + HALF >= n or apex - HALF < 0:
        return None

    e0 = max(apex - TRACE, 0)
    i_entry = e0 + int(np.argmax(s[e0:apex + 1]))
    v_entry, v_apex, v_exit = s[i_entry], s[apex], s[apex + HALF]
    window = slice(apex - HALF, apex + HALF + 1)
    covered = s[window][:-1].sum() / HZ                      # yards run in the 1 s corner
    turn_deg = float(np.rad2deg(abs(heading[apex + HALF] - heading[apex - HALF])))

    return {
        "apex": apex,
        "turn_deg": min(turn_deg, 180.0),
        "v_entry": v_entry,
        "v_apex": v_apex,
        "v_exit": v_exit,
        "speed_retained": v_apex / v_entry,
        "exit_accel": (v_exit - v_apex) / (HALF / HZ),
        "brake_dist": s[i_entry:apex].sum() / HZ,
        "peak_brake_g": float(-a_lon[i_entry:apex + 1].min() / G),
        "peak_lat_g": float(np.abs(a_lat[window]).max() / G),
        "yards_lost": v_entry * (2 * HALF / HZ) - covered,   # vs holding entry speed
    }


def speed_trace(s, apex):
    """Speed from 1.5 s before to 1.5 s after the apex (NaN where missing)."""
    out = np.full(2 * TRACE + 1, np.nan)
    lo, hi = apex - TRACE, apex + TRACE + 1
    a, b = max(lo, 0), min(hi, len(s))
    out[a - lo:b - lo] = s[a:b]
    return out


def pairwise_distance(rx, ry, dx, dy):
    """Distance from every receiver to every defender, per frame: (receivers, defenders, frames)."""
    return np.hypot(rx[:, None, :] - dx[None, :, :], ry[:, None, :] - dy[None, :, :])


def _first_frame(events, names, after=None):
    hit = events[events.isin(names)]
    if after is not None:
        hit = hit[hit.index > after]
    return int(hit.index[0]) if len(hit) else None


def analyze_play(play, route_ids, cover_ids):
    """Corner metrics for every route runner on one (direction-normalized) play.

    play: tracking rows for one play. Returns (route rows, g-g samples).
    """
    events = play.drop_duplicates("frameId").set_index("frameId")["event"]
    snap = _first_frame(events, SNAP_EVENTS)
    if snap is None:
        return [], []
    release = _first_frame(events, RELEASE_EVENTS, after=snap)
    end = _first_frame(events, END_EVENTS, after=snap) or int(events.index.max())
    frames = np.arange(snap, end + 1)
    if len(frames) < SKIP_FRAMES + 2 * HALF + 1:
        return [], []

    live = play[play["frameId"].between(snap, end)]
    nf = len(frames)

    def grids(ids):
        """(ids kept, {col: array[player, frame]}) for players with complete tracking."""
        sub = live[live["nflId"].isin(ids)].sort_values(["nflId", "frameId"])
        counts = sub.groupby("nflId")["frameId"].size()
        keep = counts.index[counts == nf].to_numpy()
        sub = sub[sub["nflId"].isin(keep)]
        return keep, {c: sub[c].to_numpy(float).reshape(len(keep), nf) for c in ("x", "y", "s", "dir")}

    rids, r = grids(route_ids)
    if len(rids) == 0:
        return [], []
    dids, d = grids(cover_ids)
    dist = pairwise_distance(r["x"], r["y"], d["x"], d["y"])
    sep = dist.min(axis=1) if len(dids) else np.full(r["x"].shape, np.nan)   # nearest defender, any
    rel = None if release is None or release > end else release - snap

    rows, gg = [], []
    for i, pid in enumerate(rids):
        s_raw, d_raw = r["s"][i], r["dir"][i]
        if np.isnan(s_raw).any() or np.isnan(d_raw).any():
            continue
        s, heading, a_lon, a_lat = kinematics(s_raw, d_raw)
        gg.append(pd.DataFrame({"nflId": pid, "s": s, "sep": sep[i],
                                "a_lon_g": a_lon / G, "a_lat_g": a_lat / G}))
        row = {"nflId": pid, "route_frames": len(s), "max_speed": s.max(),
               "sep_release": sep[i, rel] if rel is not None else np.nan,
               "has_corner": False}
        corner = find_corner(s, heading, a_lon, a_lat)
        if corner:
            a = corner.pop("apex")
            x = r["x"][i]
            row.update(corner, has_corner=True,
                       apex_time=a / HZ,
                       depth=x[a] - x[0],
                       sep_in=sep[i, a - HALF], sep_apex=sep[i, a], sep_out=sep[i, a + HALF],
                       sep_gain=sep[i, a + HALF] - sep[i, a - HALF],
                       thrown_before_apex=bool(rel is not None and rel < a),
                       trace=speed_trace(s, a))
            if len(dids):
                # same-defender view: lock onto whoever is closest at corner entry and follow him
                j = int(np.argmin(dist[i, :, a - HALF]))
                row.update(def_id=dids[j],
                           sep_gain_same=dist[i, j, a + HALF] - dist[i, j, a - HALF])
        rows.append(row)
    return rows, gg


def corner_type(turn_deg):
    return pd.cut(turn_deg, CORNER_BINS, labels=CORNER_LABELS, right=False)
