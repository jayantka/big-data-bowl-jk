"""Turn route-level telemetry into findings, figures and the dashboard data.

Usage: python analyze.py [data_dir] [out_dir]     (run build.py first)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import route_telemetry as rt

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset/data")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "output")
FIG = Path("figures")
DOCS = Path("docs")
MIN_ROUTES = 100          # players shown in the tool
NB = 16                   # angular bins of the g-g envelope
KEEP = slice(0, 2 * rt.TRACE + 1 - 5)   # tracking ends soon after the throw: show -1.5 s .. +1.0 s
T = (np.arange(-rt.TRACE, rt.TRACE + 1) / rt.HZ)[KEEP]

BG, FG, GRID = "#0f1115", "#e8eaed", "#2a2e37"
C1, C2, C3, MUTED = "#00d2be", "#ff8700", "#dc0000", "#8a8f98"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
    "axes.edgecolor": GRID, "grid.color": GRID, "axes.grid": True, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


# ----------------------------------------------------------------------------- data
def load():
    routes = pd.read_parquet(OUT / "routes.parquet")
    traces = np.load(OUT / "traces.npy")[:, KEEP]
    gg = pd.read_parquet(OUT / "gg.parquet")
    return routes, traces, gg


def quality_corners(routes):
    """Corners run at real speed, after the release, against a defender in range."""
    q = routes[routes["corner_type"].notna()
               & (routes["v_entry"] >= 4.5) & (routes["apex_time"] >= 1.0)
               & routes["sep_in"].between(0.7, 8.0)
               & routes["pff_passCoverageType"].isin(["Man", "Zone"])].copy()
    # entry speed over expected: deeper breaks allow more build-up, so compare at equal depth
    q["entry_speed_oe"] = np.nan
    for _, d in q.groupby("corner_type", observed=True):
        depth = d["depth"].clip(-3, 25)
        q.loc[d.index, "entry_speed_oe"] = d["v_entry"] - np.polyval(np.polyfit(depth, d["v_entry"], 2), depth)
    return q


# ----------------------------------------------------------------------------- g-g envelope
def envelope_radii(a_lat, a_lon, nb=NB):
    """95th-percentile combined g in each direction of the g-g plane."""
    ang = np.arctan2(a_lon, a_lat)
    rad = np.hypot(a_lon, a_lat)
    b = np.floor((ang + np.pi) / (2 * np.pi) * nb).clip(0, nb - 1).astype(int)
    return np.array([np.percentile(rad[b == k], 95) if (b == k).sum() >= 15 else np.nan
                     for k in range(nb)])


def envelope_area(radii):
    if np.isnan(radii).any():
        return np.nan
    return 0.5 * np.sin(2 * np.pi / len(radii)) * np.sum(radii * np.roll(radii, 1))


def envelope_xy(radii):
    ang = -np.pi + (np.arange(len(radii)) + 0.5) * 2 * np.pi / len(radii)
    x, y = radii * np.cos(ang), radii * np.sin(ang)
    return np.append(x, x[0]), np.append(y, y[0])


# ----------------------------------------------------------------------------- stats
def zscore(v):
    return (v - v.mean()) / v.std()


DRIVERS = {"v_entry": "Entry speed", "peak_brake_g": "Braking force",
           "peak_lat_g": "Cornering (lateral) g", "exit_accel": "Exit acceleration"}
CONTROLS = ["sep_in", "turn_deg", "depth"]


def separation_model(q, outcome="sep_gain", n_boot=400, seed=7):
    """Yards of separation gained through the break per +1 SD of each driver.

    OLS with controls for entry separation, turn angle, depth and man/zone.
    Confidence intervals come from resampling players (cluster bootstrap).
    """
    cols = list(DRIVERS) + CONTROLS

    def fit(d):
        X = np.column_stack([np.ones(len(d))] + [zscore(d[c]) for c in cols]
                            + [(d["pff_passCoverageType"] == "Man").astype(float)])
        return np.linalg.lstsq(X, d[outcome].to_numpy(), rcond=None)[0][1:len(DRIVERS) + 1]

    point = fit(q)
    rng = np.random.default_rng(seed)
    by_player = {pid: d for pid, d in q.groupby("nflId")}
    ids = np.array(list(by_player))
    boots = np.array([fit(pd.concat([by_player[i] for i in rng.choice(ids, len(ids))]))
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return pd.DataFrame({"driver": list(DRIVERS.values()), "coef": point, "lo": lo, "hi": hi})


def split_half(corners, metric, min_n=8):
    """Correlation of a player's metric in weeks 1-4 with the same metric in weeks 5-8."""
    half = np.where(corners["week"] <= 4, "a", "b")
    g = corners.groupby(["nflId", half])[metric].agg(["mean", "size"]).unstack()
    g = g[(g["size"] >= min_n).all(axis=1)]
    return g["mean"].corr().iloc[0, 1], len(g)


# ----------------------------------------------------------------------------- player table
def player_table(routes, corners, gg):
    free = gg[gg["sep"] > 1.5]                      # frames without a defender in contact range
    radii = free.groupby("nflId").apply(lambda d: envelope_radii(d["a_lat_g"].to_numpy(), d["a_lon_g"].to_numpy()))
    p = routes.groupby("nflId").agg(player=("displayName", "first"), pos=("officialPosition", "first"),
                                    team=("possessionTeam", lambda s: s.mode().iat[0]),
                                    weight=("weight", "first"), routes=("playId", "size"),
                                    targets=("targeted", "sum"), sep_at_release=("sep_release", "mean"))
    c = corners.groupby("nflId").agg(corners=("playId", "size"), entry_speed=("v_entry", "mean"),
                                     entry_speed_oe=("entry_speed_oe", "mean"),
                                     apex_speed=("v_apex", "mean"), braking_g=("peak_brake_g", "mean"),
                                     cornering_g=("peak_lat_g", "mean"), exit_accel=("exit_accel", "mean"),
                                     sep_gain=("sep_gain", "mean"))
    p = p.join(c).join(radii.rename("radii"))
    p["envelope_area"] = p["radii"].map(lambda r: envelope_area(r) if isinstance(r, np.ndarray) else np.nan)
    p["top_speed"] = gg.groupby("nflId")["s"].quantile(0.99)
    p = p[p["routes"] >= MIN_ROUTES].copy()
    for col in ["entry_speed", "entry_speed_oe", "envelope_area", "braking_g", "cornering_g", "exit_accel", "top_speed"]:
        p[col + "_pct"] = p.groupby("pos")[col].rank(pct=True).mul(100).round()
    return p.sort_values("entry_speed", ascending=False)


# ----------------------------------------------------------------------------- figures
def fig_contact_sheet(routes, n=15, seed=3):
    """Raw paths of randomly drawn detected corners, apex marked: the eyeball test."""
    q = quality_corners(routes)
    game = q["gameId"].value_counts().index[0]
    picks = q[q["gameId"] == game].sample(n, random_state=seed)
    trk = rt.normalize_direction(pd.read_csv(DATA / "tracking" / f"tracking_{game}.csv"))
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    for ax, (_, r) in zip(axes.ravel(), picks.iterrows()):
        play = trk[trk["playId"] == r["playId"]]
        snap = play.loc[play["event"].isin(rt.SNAP_EVENTS), "frameId"].min()
        me = play[(play["nflId"] == r["nflId"]) & (play["frameId"] >= snap)].sort_values("frameId")
        a = int(round(r["apex_time"] * rt.HZ))
        x, y = me["x"].to_numpy(), me["y"].to_numpy()
        ax.plot(y, x, color=MUTED, lw=1.5)
        ax.plot(y[a - rt.HALF:a + rt.HALF + 1], x[a - rt.HALF:a + rt.HALF + 1], color=C1, lw=3)
        ax.scatter(y[a], x[a], color=C3, s=60, zorder=3)
        ax.scatter(y[0], x[0], color=FG, s=20, zorder=3)
        ax.set_title(f"{r['displayName']}\n{r['corner_type'].split(' (')[0]} {r['turn_deg']:.0f}°  "
                     f"{r['v_entry']:.1f}→{r['v_apex']:.1f} yd/s", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    fig.suptitle("Eyeball test: 15 randomly drawn detected corners (white = snap, teal = corner window, red = apex)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG / "01_corner_detection_check.png", dpi=130)
    plt.close(fig)


def fig_separation(model, q):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1, 1.1]})
    m = model.iloc[::-1]
    colors = [C1 if lo > 0 else C3 if hi < 0 else MUTED for lo, hi in zip(m["lo"], m["hi"])]
    ax1.barh(m["driver"], m["coef"], color=colors, height=0.55)
    ax1.errorbar(m["coef"], m["driver"], xerr=[m["coef"] - m["lo"], m["hi"] - m["coef"]],
                 fmt="none", ecolor=FG, capsize=4, lw=1.2)
    ax1.axvline(0, color=FG, lw=1)
    ax1.set_xlabel("Association with separation gained through the break\n(yards per +1 SD, 95% CI, players resampled)")
    ax1.set_title("What is associated with separation at the break?", loc="left", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", visible=False)

    q = q.copy()
    q["entry_q"] = q.groupby("corner_type", observed=True)["v_entry"].transform(
        lambda v: pd.qcut(v, 4, labels=False))
    # separation gain adjusted for where the defender started (regression to the mean)
    slope = np.polyfit(q["sep_in"], q["sep_gain"], 1)
    q["gain_adj"] = q["sep_gain"] - np.polyval(slope, q["sep_in"]) + q["sep_gain"].mean()
    for (ctype, d), color in zip(q.groupby("corner_type", observed=True), [C1, C2, C3]):
        g = d.groupby("entry_q")["gain_adj"].agg(["mean", "sem"])
        ax2.errorbar(range(4), g["mean"], yerr=1.96 * g["sem"], color=color, marker="o", lw=2,
                     capsize=3, label=f"{ctype}  (n={len(d):,})")
    ax2.set_xticks(range(4), ["Slowest 25%", "Q2", "Q3", "Fastest 25%"])
    ax2.set_xlabel("Entry speed into the break (quartile within corner type)")
    ax2.set_ylabel("Separation gained through the break (yd)\nadjusted for starting separation")
    ax2.set_title("Faster entries gain more separation", loc="left", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "02_separation_drivers.png", dpi=140)
    plt.close(fig)


def mean_trace(routes, traces, corners, pid, ctype):
    idx = corners.index[(corners["nflId"] == pid) & (corners["corner_type"] == ctype)]
    return (np.nanmean(traces[idx], axis=0), len(idx)) if len(idx) else (None, 0)


def fig_ghost_car(routes, traces, corners, players, a, b, ctype):
    ta, na = mean_trace(routes, traces, corners, a, ctype)
    tb, nb = mean_trace(routes, traces, corners, b, ctype)
    league = np.nanmean(traces[corners.index[corners["corner_type"] == ctype]], axis=0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    ax1.plot(T, league, color=MUTED, lw=1.5, ls="--", label="League average")
    ax1.plot(T, ta, color=C1, lw=3, label=f"{players.loc[a, 'player']} ({na} corners)")
    ax1.plot(T, tb, color=C2, lw=3, label=f"{players.loc[b, 'player']} ({nb} corners)")
    ax1.axvline(0, color=FG, lw=0.8, ls=":")
    ax1.text(0.02, ax1.get_ylim()[1], "APEX", fontsize=9, color=FG, va="top")
    ax1.set_ylabel("Speed (yd/s)")
    ax1.set_title(f"Ghost-car telemetry: {ctype} corners", loc="left", fontsize=13, fontweight="bold")
    ax1.legend(loc="lower center", ncol=3, fontsize=9)
    gap = np.nancumsum(ta - tb) / rt.HZ           # yards A has pulled ahead since the window opened
    ax2.fill_between(T, gap, 0, where=gap >= 0, color=C1, alpha=0.5)
    ax2.fill_between(T, gap, 0, where=gap < 0, color=C2, alpha=0.5)
    ax2.axhline(0, color=FG, lw=0.8)
    ax2.set_ylabel(f"Yards gained on\n{players.loc[b, 'player'].split()[-1]} since -1.5 s")
    ax2.set_xlabel("Seconds from apex of the cut")
    fig.tight_layout()
    fig.savefig(FIG / "03_ghost_car.png", dpi=140)
    plt.close(fig)


def fig_gg(gg, players, pids):
    fig, axes = plt.subplots(1, len(pids), figsize=(5 * len(pids), 5.4), sharex=True, sharey=True)
    for ax, pid, color in zip(axes, pids, [C1, C2, C3]):
        d = gg[(gg["nflId"] == pid) & (gg["sep"] > 1.5)]
        ax.scatter(d["a_lat_g"], d["a_lon_g"], s=3, color=color, alpha=0.25, linewidths=0)
        ex, ey = envelope_xy(players.loc[pid, "radii"])
        ax.plot(ex, ey, color=FG, lw=2)
        p = players.loc[pid]
        ax.set_title(f"{p['player']} ({p['pos']}, {p['weight']} lb)\nenvelope {p['envelope_area']:.2f} g²  ·  "
                     f"{p['envelope_area_pct']:.0f}th pct of {p['pos']}s", fontsize=11)
        ax.axhline(0, color=GRID, lw=1)
        ax.axvline(0, color=GRID, lw=1)
        ax.set_xlabel("Estimated lateral g  (cutting)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("Estimated longitudinal g  (braking  ↓   accelerating  ↑)")
    fig.suptitle("G-G diagram: each receiver's performance envelope, estimated from 10 Hz tracking (white = 95th-percentile limit)",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "04_gg_diagram.png", dpi=140)
    plt.close(fig)


def fig_payoff(q):
    """Receiver and defender paths through a slow-entry and a fast-entry corner.

    Each example is the corner closest to the median of its entry-speed quartile
    (entry speed, starting separation, separation change), so it is typical, not hand-picked.
    """
    pool = q[(q["corner_type"] == rt.CORNER_LABELS[0]) & q["def_id"].notna()].copy()
    pool["quart"] = pd.qcut(pool["v_entry"], 4, labels=False)
    keys = ["v_entry", "sep_in", "sep_gain_same"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    for ax, quart, color, label in zip(axes, (0, 3), (C2, C1), ("Slowest 25% of entries", "Fastest 25% of entries")):
        d = pool[pool["quart"] == quart]
        score = sum(((d[k] - d[k].median()) / pool[k].std()) ** 2 for k in keys)
        r = d.loc[score.idxmin()]
        trk = rt.normalize_direction(pd.read_csv(DATA / "tracking" / f"tracking_{int(r['gameId'])}.csv"))
        play = trk[trk["playId"] == r["playId"]]
        snap = play.loc[play["event"].isin(rt.SNAP_EVENTS), "frameId"].min()
        a = int(round(r["apex_time"] * rt.HZ))
        path = lambda pid: play[(play["nflId"] == pid) & play["frameId"].between(snap, snap + a + rt.HALF)] \
            .sort_values("frameId")[["y", "x"]].to_numpy()
        wr, cb = path(r["nflId"]), path(r["def_id"])
        ax.plot(*wr.T, color=color, lw=3, label=f"{r['displayName']} (receiver)")
        ax.plot(*cb.T, color=FG, lw=2, ls="--", label="Nearest defender at entry")
        for k, tag in ((a - rt.HALF, "0.5 s before"), (a + rt.HALF, "0.5 s after")):
            gap = np.hypot(*(wr[k] - cb[k]))
            ax.plot([wr[k, 0], cb[k, 0]], [wr[k, 1], cb[k, 1]], color=MUTED, lw=1.2)
            ax.scatter(*wr[k], color=color, s=50, zorder=3)
            ax.scatter(*cb[k], color=FG, s=30, zorder=3)
            mid = (wr[k] + cb[k]) / 2
            ax.annotate(f"{gap:.1f} yd apart\n{tag} apex", mid, xytext=(55, 0), textcoords="offset points",
                        fontsize=9, color=FG, ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8),
                        bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec=GRID))
        ax.scatter(*wr[a], color=C3, s=70, zorder=4, label="Apex of the cut")
        ax.set_title(f"{label}: enters at {r['v_entry']:.1f} yd/s\nseparation {r['sep_in']:.1f} → "
                     f"{r['sep_in'] + r['sep_gain_same']:.1f} yd ({r['sep_gain_same']:+.1f})",
                     loc="left", fontsize=11, fontweight="bold")
        ax.margins(x=0.6, y=0.08)
        ax.set_aspect("equal")
        ax.set_xlabel("Across the field (yd)")
        ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    axes[0].set_ylabel("Downfield (yd)")
    fig.suptitle("Separation through the break on a typical sweeper corner: receiver vs the defender nearest him at entry",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "05_separation_payoff.png", dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------- dashboard data
def dashboard_json(routes, traces, corners, players, gg, findings):
    rng = np.random.default_rng(0)
    out = {"t": T.round(1).tolist(), "types": rt.CORNER_LABELS, "findings": findings, "players": {}, "league": {}}
    for ctype in rt.CORNER_LABELS:
        out["league"][ctype] = np.nanmean(traces[corners.index[corners["corner_type"] == ctype]], axis=0).round(2).tolist()
    free = gg[gg["sep"] > 1.5]
    pts = {pid: d for pid, d in free[free["nflId"].isin(players.index)].groupby("nflId")}
    for pid, p in players.iterrows():
        tr = {}
        for ctype in rt.CORNER_LABELS:
            m, n = mean_trace(routes, traces, corners, pid, ctype)
            if n >= 5:
                tr[ctype] = {"n": n, "v": [None if np.isnan(v) else round(float(v), 2) for v in m]}
        d = pts[pid]
        d = d.iloc[rng.choice(len(d), min(len(d), 350), replace=False)]
        ex, ey = envelope_xy(p["radii"])
        stat = lambda k: None if pd.isna(p[k]) else round(float(p[k]), 2)
        out["players"][str(int(pid))] = {
            "name": p["player"], "pos": p["pos"], "team": p["team"], "weight": int(p["weight"]),
            "routes": int(p["routes"]), "corners": int(p["corners"]) if pd.notna(p["corners"]) else 0,
            "stats": {k: stat(k) for k in ["entry_speed", "entry_speed_oe", "entry_speed_oe_pct", "apex_speed", "braking_g", "cornering_g", "exit_accel",
                                           "envelope_area", "top_speed", "sep_at_release",
                                           "entry_speed_pct", "braking_g_pct", "cornering_g_pct",
                                           "exit_accel_pct", "envelope_area_pct", "top_speed_pct"]},
            "traces": tr,
            "gg": [d["a_lat_g"].round(2).tolist(), d["a_lon_g"].round(2).tolist()],
            "env": [np.round(ex, 3).tolist(), np.round(ey, 3).tolist()],
        }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "data.js").write_text("const DATA = " + json.dumps(out) + ";")


# ----------------------------------------------------------------------------- main
def main():
    FIG.mkdir(exist_ok=True)
    routes, traces, gg = load()
    q = quality_corners(routes)
    players = player_table(routes, q, gg)

    model = separation_model(q)
    model_same = separation_model(q.dropna(subset=["sep_gain_same"]), outcome="sep_gain_same")
    speed_kept = {t: d["speed_retained"].corr(d["sep_gain"], method="spearman")
                  for t, d in q.groupby("corner_type", observed=True)}
    reliability = {m: split_half(q, m) for m in ["v_entry", "entry_speed_oe", "peak_brake_g", "peak_lat_g", "exit_accel", "sep_gain"]}
    print(f"routes {len(routes):,} | detected corners {routes.corner_type.notna().sum():,} | quality corners {len(q):,}")
    print(model.round(3).to_string(index=False))
    print("robustness: separation from the SAME defender (nearest at corner entry)")
    print(model_same.round(3).to_string(index=False))
    print("speed retained vs separation gained (Spearman):", {k: round(v, 2) for k, v in speed_kept.items()})
    print("split-half r (weeks 1-4 vs 5-8):", {k: f"{r:.2f} (n={n})" for k, (r, n) in reliability.items()})

    cols = ["player", "pos", "team", "weight", "routes", "targets", "corners", "entry_speed", "entry_speed_oe", "apex_speed",
            "braking_g", "cornering_g", "exit_accel", "envelope_area", "top_speed", "sep_gain", "sep_at_release"]
    players[cols].round(2).to_csv(OUT / "player_telemetry.csv")
    for pos in ["WR", "TE", "RB"]:
        print(f"\nTop entry speed, {pos} (min 20 quality corners)")
        print(players[(players.pos == pos) & (players.corners >= 20)][cols[:1] + cols[4:13]].head(8).round(2).to_string(index=False))

    free = gg[gg["sep"] > 1.5]
    half = np.where(free["week"] <= 4, "weeks_1_4", "weeks_5_8")   # same split as the corner metrics
    area = free.groupby(["nflId", half]).apply(
        lambda d: envelope_area(envelope_radii(d["a_lat_g"].to_numpy(), d["a_lon_g"].to_numpy()))).unstack()
    size = free.groupby(["nflId", half]).size().unstack()
    enough = (size >= 1500).all(axis=1)                 # about 50 routes in each half
    env_r, env_n = area[enough].corr().iloc[0, 1], int(enough.sum())
    wr = players[players.pos == "WR"]
    env_vs_open = wr["envelope_area"].corr(wr["sep_at_release"], method="spearman")
    env_vs_tgt = wr["envelope_area"].corr(wr["targets"] / wr["routes"], method="spearman")
    print(f"envelope weeks 1-4 vs 5-8 r={env_r:.2f} (n={env_n}) | WR envelope vs separation at release rho={env_vs_open:.2f}, "
          f"vs target rate rho={env_vs_tgt:.2f} (n={len(wr)})")
    kupp = players[players.player == "Cooper Kupp"].iloc[0]
    print(f"Kupp: envelope pct {kupp.envelope_area_pct:.0f}, targets {kupp.targets:.0f}, routes {kupp.routes}")

    findings = {"model": model.round(3).to_dict("records"),
                "model_same_defender": model_same.round(3).to_dict("records"),
                "speed_retained_vs_sep_gain": {k: round(float(v), 2) for k, v in speed_kept.items()},
                "envelope_reliability": round(float(env_r), 2),
                "envelope_reliability_n": env_n,
                "wr_envelope_vs_separation": round(float(env_vs_open), 2),
                "wr_envelope_vs_target_rate": round(float(env_vs_tgt), 2),
                "reliability": {k: round(float(r), 2) for k, (r, n) in reliability.items()},
                "n_routes": int(len(routes)), "n_corners": int(len(q))}
    (OUT / "findings.json").write_text(json.dumps(findings, indent=2))

    fig_contact_sheet(routes)
    fig_separation(model, q)
    fig_payoff(q)
    by_name = players.reset_index().set_index("player")["nflId"]
    ghost = [by_name[n] for n in GHOST_PAIR]
    fig_ghost_car(routes, traces, q, players, ghost[0], ghost[1], GHOST_TYPE)
    fig_gg(gg, players, [by_name[n] for n in GG_TRIO])
    dashboard_json(routes, traces, q, players, gg, findings)


GHOST_PAIR = ("Tyreek Hill", "Jakobi Meyers")
GHOST_TYPE = rt.CORNER_LABELS[0]
GG_TRIO = ("Cooper Kupp", "Travis Kelce", "Derrick Henry")

if __name__ == "__main__":
    main()
