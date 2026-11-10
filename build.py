"""Run route telemetry over every game and save route-level results.

Usage: python build.py [data_dir] [out_dir]
"""
import glob
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import route_telemetry as rt

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else "dataset/data")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "output")
SKILL = {"WR", "TE", "RB"}
COLS = ["gameId", "playId", "nflId", "frameId", "playDirection", "x", "y", "s", "dir", "event"]


def process_game(path):
    pff = pd.read_csv(DATA / "pffScoutingData.csv", usecols=["gameId", "playId", "nflId", "pff_role"])
    trk = rt.normalize_direction(pd.read_csv(path, usecols=COLS).dropna(subset=["nflId"]))
    game_id = int(trk["gameId"].iloc[0])
    roles = pff[pff["gameId"] == game_id]
    routes = roles[roles["pff_role"] == "Pass Route"].groupby("playId")["nflId"].apply(set)
    cover = roles[roles["pff_role"] == "Coverage"].groupby("playId")["nflId"].apply(set)

    rows, gg = [], []
    for play_id, play in trk.groupby("playId", sort=False):
        if play_id not in routes.index:
            continue
        r, g = rt.analyze_play(play, routes[play_id], cover.get(play_id, set()))
        for row in r:
            row.update(gameId=game_id, playId=play_id)
        rows += r
        gg += g
    return pd.DataFrame(rows), (pd.concat(gg) if gg else pd.DataFrame())


def find_target(desc, names):
    """nflId of the targeted receiver, parsed from the play description."""
    m = re.search(r"(?:to|intended for) ([A-Z][A-Za-z]*\.\s?[A-Za-z'\-\. ]+?)(?= to | for | pushed| ran | \(|\.$|\. |,| \[)", desc or "")
    if not m:
        return np.nan
    initial, last = m.group(1).replace(" ", "").split(".", 1)
    hits = [pid for pid, n in names.items()
            if n.split()[0][:len(initial)].lower() == initial.lower()
            and last.lower() in n.replace(" ", "").lower()]
    return hits[0] if len(hits) == 1 else np.nan


def main():
    OUT.mkdir(exist_ok=True)
    files = sorted(glob.glob(str(DATA / "tracking" / "*.csv")))
    with Pool() as pool:
        results = pool.map(process_game, files)
    routes = pd.concat([r for r, _ in results], ignore_index=True)
    gg = pd.concat([g for _, g in results], ignore_index=True)

    players = pd.read_csv(DATA / "players.csv")
    plays = pd.read_csv(DATA / "plays.csv")
    games = pd.read_csv(DATA / "games.csv")[["gameId", "week"]]
    routes = (routes.merge(players[["nflId", "displayName", "officialPosition", "weight"]], on="nflId")
                    .merge(plays[["gameId", "playId", "playDescription", "passResult", "possessionTeam",
                                  "pff_passCoverageType", "pff_passCoverage"]], on=["gameId", "playId"])
                    .merge(games, on="gameId"))
    routes = routes[routes["officialPosition"].isin(SKILL)].reset_index(drop=True)

    # targeted receiver: match the name in the play description to a route runner on the play
    name_of = routes.groupby(["gameId", "playId"]).apply(lambda d: dict(zip(d["nflId"], d["displayName"])))
    desc = routes.drop_duplicates(["gameId", "playId"]).set_index(["gameId", "playId"])["playDescription"]
    target = pd.Series({k: find_target(desc[k], name_of[k]) for k in name_of.index})
    routes["targetId"] = pd.MultiIndex.from_frame(routes[["gameId", "playId"]]).map(target)
    routes["targeted"] = routes["nflId"] == routes["targetId"]
    routes["corner_type"] = rt.corner_type(routes["turn_deg"])

    traces = np.vstack([t if isinstance(t, np.ndarray) else np.full(2 * rt.TRACE + 1, np.nan)
                        for t in routes["trace"]])
    np.save(OUT / "traces.npy", traces)                      # row i <-> routes row i
    routes.drop(columns=["trace", "playDescription"]).to_parquet(OUT / "routes.parquet")
    gg = gg[gg["nflId"].isin(routes["nflId"].unique())]
    gg.astype({"nflId": "int32", "s": "float32", "sep": "float32",
               "a_lon_g": "float32", "a_lat_g": "float32"}).to_parquet(OUT / "gg.parquet")

    print(f"routes: {len(routes):,}  with corner: {routes.has_corner.mean():.1%}  "
          f"target found on {routes.groupby(['gameId', 'playId']).targeted.any().mean():.1%} of plays")
    print(routes["corner_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
