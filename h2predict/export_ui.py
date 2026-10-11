"""Export a match (+ optional half-2 rollout) to the JSON the formation UI in ui/ reads."""
import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np

from .config import Config
from .data import BALL, PITCH
from .train import load_games

COLOURS = ["#ef5b3c", "#3d9bff"]


def load_match(spec: str):
    if spec.endswith(".pkl"):
        with open(spec, "rb") as f:
            return pickle.load(f)
    return load_games(Config(), [spec])[0]


def team_names(name: str):
    """Match names are "<source>_<id> <home>-<away>" (PFF, SkillCorner, DFL), "Sample_Game_N" (Metrica) or a video stem."""
    if name.startswith("Sample_Game"):
        return ["Home", "Away"]
    if " " in name and "-" in name.split(" ", 1)[1]:
        return name.split(" ", 1)[1].split("-", 1)
    return ["Team A", "Team B"]


def frames(pos: np.ndarray) -> list:
    flat = np.round(pos.astype(np.float64), 1).reshape(len(pos), -1)
    return [[None if np.isnan(v) else v for v in row] for row in flat.tolist()]


def half(m, p: int, dt: float, sign: float):
    rows = np.flatnonzero(m.period == p)
    if len(rows) == 0:
        return None
    rows = rows[:: max(1, round(dt * m.fps))]
    out = {"t0": 45.0 * (p - 1) * 60, "dt": dt, "pos": frames(sign * m.pos[rows]), "poss": m.possessor[rows].tolist()}
    if getattr(m, "ball_seen", None) is not None:  # video: 0 seen, 1 in play but not detected, 2 out of play
        out["ball"] = np.where(m.ball_seen[rows], 0, np.where(m.state[rows] == 3, 2, 1)).tolist()
    return out


def build_doc(m, rollout=None, samples=8, dt=1.0, meta=None, out_dir=Path("ui/data")) -> dict:
    """meta (written by h2predict.vision next to match.pkl): show the camera's orientation instead of the
    model's 'team 0 attacks +x' convention, real kit colours, keeper slots, and the source video."""
    meta = meta or {}
    flipped = bool(meta.get("flip"))
    sign = -1.0 if flipped else 1.0  # the pipeline rotated by 180 deg; undo it for display
    names = team_names(m.name)
    colours = meta.get("colors") or COLOURS
    doc = {
        "name": m.name, "pitch": list(PITCH), "ball": BALL,
        "teams": [{"name": names[i], "color": colours[i]} for i in range(2)],
        "attackDir": [-1, 1] if flipped else [1, -1],
        "half1": half(m, 1, dt, sign), "half2": half(m, 2, dt, sign), "pred": None,
    }
    if "gk" in meta:
        doc["gk"] = meta["gk"]
    if meta.get("video"):
        src = Path(meta["video"]).resolve()
        link = out_dir / src.name
        out_dir.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            os.symlink(src, link)  # full-match videos are GBs; the static server follows symlinks
        doc["video"] = {"file": src.name, "offset": float(meta.get("video_offset", 0.0))}
    if rollout is not None:
        k = min(samples, len(rollout["pred_xy"]))
        doc["pred"] = {
            "t0": 45.0 * 60, "dt": float(rollout["step_s"]),
            "samples": [{"pos": frames(sign * rollout["pred_xy"][i]), "poss": rollout["pred_poss"][i].tolist()}
                        for i in range(k)],
        }
    return doc


def save_doc(doc: dict, out_dir: Path) -> Path:
    """Write one match JSON and register it in out_dir/index.json (the UI's match list)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = "".join(c if c.isalnum() or c in "-_" else "_" for c in doc["name"]) + ".json"
    path = out_dir / fname
    path.write_text(json.dumps(doc, separators=(",", ":")))
    index_path = out_dir / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else []
    index = [e for e in index if e["file"] != fname] + [{
        "file": fname, "name": doc["name"], "hasPred": doc.get("pred") is not None,
        "hasHalf2": doc.get("half2") is not None, "hasVideo": bool(doc.get("video")),
    }]
    index_path.write_text(json.dumps(index, indent=1))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--match", required=True, help="metrica:<game> | pff:<id> | path/to/match.pkl")
    ap.add_argument("--rollout", default=None, help="rollout.npz produced for the same match")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--dt", type=float, default=None, help="seconds between exported real frames "
                    "(default 1.0; every frame for video-derived matches)")
    ap.add_argument("--out-dir", default="ui/data")
    a = ap.parse_args(argv)
    for opt, path in (("--match", a.match), ("--rollout", getattr(a, "rollout", None)), ("--ckpt", getattr(a, "ckpt", None))):
        if path and path.endswith((".pkl", ".npz", ".pt")) and not Path(path).is_file():
            ap.error(f"không tìm thấy {opt} {path}: file chưa được tạo (chạy h2predict.vision trên video thật trước)")

    m = load_match(a.match)
    meta_path = Path(a.match).with_name("ui_meta.json") if a.match.endswith(".pkl") else None
    meta = json.loads(meta_path.read_text()) if meta_path and meta_path.exists() else None
    dt = a.dt or (1.0 / m.fps if meta else 1.0)
    rollout = np.load(a.rollout) if a.rollout else None
    out_dir = Path(a.out_dir)
    save_doc(build_doc(m, rollout, a.samples, dt, meta, out_dir), out_dir)


if __name__ == "__main__":
    main()
