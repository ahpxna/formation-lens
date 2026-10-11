"""Layer A: broadcast video -> metric coordinates of 22 players + ball per frame -> Match for the forecaster.

detect (YOLOv8, Roboflow football weights) -> team (jersey colour KMeans)
-> pitch calibration (homography from 32 pitch keypoints, solved independently per frame, never accumulated)
-> feet point projected to the pitch -> association in pitch metres into 11 slots per team
-> Match (same format as Metrica tracking).
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sports.configs.soccer import SoccerPitchConfiguration
from ultralytics import YOLO

from .data import BALL, N_AGENTS, N_TEAM, PITCH, build_match, normalized_to_meters

PITCH_CFG = SoccerPitchConfiguration()
VERTS_CM = np.array(PITCH_CFG.vertices, np.float32)
BALL_CLS, GK_CLS, PLAYER_CLS, REF_CLS = 0, 1, 2, 3
MARGIN_M = 5.0


def template_to_m(p_cm: np.ndarray) -> np.ndarray:
    return normalized_to_meters(p_cm / np.array([PITCH_CFG.length, PITCH_CFG.width], np.float32))


VERTS_M = template_to_m(VERTS_CM)


class PitchCalibrator:
    """Homography from detected pitch keypoints, re-solved on every frame (no drift from accumulating camera motion),
    rejected when RANSAC reprojection error is high; the last good one is held for at most `hold` frames."""

    def __init__(self, model: YOLO, device: str, kp_conf=0.5, max_err_cm=150.0, hold=5):
        self.model, self.device, self.kp_conf, self.max_err_cm, self.hold = model, device, kp_conf, max_err_cm, hold
        self.H, self.age = None, 0

    def __call__(self, frame):
        r = self.model.predict(frame, device=self.device, verbose=False)[0]
        if r.keypoints is not None and r.keypoints.conf is not None and len(r.keypoints.xy):
            xy = r.keypoints.xy[0].cpu().numpy()
            ok = r.keypoints.conf[0].cpu().numpy() > self.kp_conf
            if ok.sum() >= 4:
                H, inl = cv2.findHomography(xy[ok], VERTS_CM[ok], cv2.RANSAC, 100.0)
                if H is not None:
                    inl = inl.ravel().astype(bool)
                    proj = cv2.perspectiveTransform(xy[ok][inl][None], H)[0]
                    err = float(np.linalg.norm(proj - VERTS_CM[ok][inl], axis=1).mean())
                    if inl.sum() >= 4 and err <= self.max_err_cm:
                        self.H, self.age = H, 0
                        return H, err, int(inl.sum())
        self.age += 1
        return (self.H if self.H is not None and self.age <= self.hold else None), np.nan, 0


def torso_colour(frame, xyxy):
    """Median Lab colour of the shirt area with grass pixels removed."""
    x1, y1, x2, y2 = xyxy.astype(int)
    h, w = y2 - y1, x2 - x1
    crop = frame[max(y1 + int(0.15 * h), 0): y1 + int(0.5 * h), max(x1 + int(0.2 * w), 0): x2 - int(0.2 * w)]
    if crop.size < 30:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    keep = ~((hsv[:, 0] > 30) & (hsv[:, 0] < 90) & (hsv[:, 1] > 50))
    return np.median(lab[keep], 0) if keep.sum() >= 10 else None


def project(H, pts):
    return template_to_m(cv2.perspectiveTransform(pts[None].astype(np.float32), H)[0])


def on_pitch(p):
    return (np.abs(p[:, 0]) <= PITCH[0] / 2 + MARGIN_M) & (np.abs(p[:, 1]) <= PITCH[1] / 2 + MARGIN_M)


def detect(video, player_model, pitch_model, ball_model, stride, device, det_conf, max_frames):
    """Pass 1 (the expensive one): everything that needs pixels."""
    info = sv.VideoInfo.from_video_path(video)
    calib = PitchCalibrator(pitch_model, device)
    frames = []
    for fi, frame in enumerate(sv.get_video_frames_generator(video, stride=stride)):
        if max_frames and fi >= max_frames:
            break
        res = player_model.predict(frame, imgsz=1280, conf=det_conf, device=device, verbose=False)[0]
        det = sv.Detections.from_ultralytics(res).with_nms(threshold=0.5, class_agnostic=True)
        ball = det[det.class_id == BALL_CLS]
        if len(ball) == 0 and ball_model is not None:
            ball = sv.Detections.from_ultralytics(
                ball_model.predict(frame, imgsz=1280, conf=det_conf, device=device, verbose=False)[0])
        people = det[np.isin(det.class_id, [GK_CLS, PLAYER_CLS])]
        H, err, n_inl = calib(frame)
        rec = {"frame": fi * stride, "H": H, "err": err, "inliers": n_inl, "dets": [], "ball": None}
        if H is not None:
            if len(ball):
                b = ball.xyxy[np.argmax(ball.confidence)]
                pb = project(H, np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]]))
                rec["ball"] = pb[0] if on_pitch(pb)[0] else None
            if len(people):
                pm = project(H, people.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
                for j in np.flatnonzero(on_pitch(pm)):
                    rec["dets"].append({"cls": int(people.class_id[j]), "xy": pm[j], "box": people.xyxy[j],
                                        "colour": torso_colour(frame, people.xyxy[j]), "team": None, "agent": None})
        frames.append(rec)
        if fi % 25 == 0:
            print(f"  frame {fi * stride}/{info.total_frames} calibrated={H is not None} "
                  f"err={err:.0f}cm people={len(rec['dets'])} ball={rec['ball'] is not None}")
    return frames, info.fps / stride


def assign_teams(frames, seed=0, ambiguity=0.8):
    """Jersey colour KMeans(2), decided per detection; colour-ambiguous crops (occlusions) stay unassigned.
    Goalkeepers wear their own kit: each goal's keeper takes the majority team of the outfield players nearest
    to them across the whole clip (a keeper stays at one end for a whole half)."""
    cols = [d["colour"] for f in frames for d in f["dets"] if d["cls"] == PLAYER_CLS and d["colour"] is not None]
    if len(cols) < 20:
        raise RuntimeError("too few player crops to learn team colours")
    km = KMeans(2, n_init=10, random_state=seed).fit(np.array(cols))
    for f in frames:
        for d in f["dets"]:
            if d["cls"] == PLAYER_CLS and d["colour"] is not None:
                dist = np.linalg.norm(km.cluster_centers_ - d["colour"], axis=1)
                if dist.min() < ambiguity * dist.max():
                    d["team"] = int(dist.argmin())
    side = lambda d: 1 if d["xy"][0] >= 0 else -1
    votes = {-1: Counter(), 1: Counter()}
    for f in frames:
        outfield = [d for d in f["dets"] if d["team"] is not None]
        for d in f["dets"]:
            if d["cls"] == GK_CLS and outfield:
                near = sorted(outfield, key=lambda o: np.linalg.norm(o["xy"] - d["xy"]))[:3]
                votes[side(d)].update(o["team"] for o in near)
    for f in frames:
        for d in f["dets"]:
            if d["cls"] == GK_CLS and votes[side(d)]:
                d["team"] = votes[side(d)].most_common(1)[0][0]
    bgr = cv2.cvtColor(km.cluster_centers_.astype(np.uint8)[None], cv2.COLOR_LAB2BGR)[0]
    return [tuple(int(c) for c in b) for b in bgr]


class SlotAssigner:
    """Associates detections with 11 persistent slots per team directly in pitch metres (slot 0 = goalkeeper).
    Hungarian on distance to each slot's last position, gated by how far a player can run since it was last
    seen; unseen slots keep their last position, so a player leaving the camera view gets the same slot back.
    Pitch space instead of pixel IoU: at 5 Hz players move ~2 m per frame and image boxes stop overlapping."""

    MAX_SPEED = 9.0
    SLACK = 2.0
    NEW_COST = 30.0

    def __init__(self, fps):
        self.fps = fps
        self.last = np.full((N_TEAM, 2), np.nan)
        self.seen = np.full(N_TEAM, -np.inf)

    def step(self, fi, dets):
        """dets: list of (xy, is_goalkeeper) -> slot index or None for each."""
        out = [None] * len(dets)
        if not dets:
            return out
        cost = np.full((len(dets), N_TEAM), 1e6)
        for a, (xy, gk) in enumerate(dets):
            for s in range(N_TEAM):
                if (s == 0) != gk:
                    continue
                if np.isnan(self.last[s, 0]):
                    cost[a, s] = self.NEW_COST
                else:
                    d = np.linalg.norm(self.last[s] - xy)
                    if d <= self.MAX_SPEED * (fi - self.seen[s]) / self.fps + self.SLACK:
                        cost[a, s] = d
        for a, s in zip(*linear_sum_assignment(cost)):
            if cost[a, s] < 1e6:
                out[a] = int(s)
                self.last[s], self.seen[s] = dets[a][0], fi
        return out


def smooth(pos, fps, ball_gap_s=1.0):
    """Light temporal smoothing of homography jitter; fill short ball gaps."""
    out = pos.copy()
    for a in range(N_AGENTS):
        for d in range(2):
            s = pd.Series(pos[:, a, d])
            if a == BALL:
                out[:, a, d] = s.interpolate(limit=int(ball_gap_s * fps), limit_area="inside").to_numpy()
            else:
                out[:, a, d] = s.rolling(3, center=True, min_periods=1).mean().where(s.notna()).to_numpy()
    return out


def build_tracking(frames, fps):
    T = len(frames)
    pos = np.full((T, N_AGENTS, 2), np.nan, np.float32)
    slots = [SlotAssigner(fps), SlotAssigner(fps)]
    for fi, f in enumerate(frames):
        for k in (0, 1):
            ds = [d for d in f["dets"] if d["team"] == k]
            for d, s in zip(ds, slots[k].step(fi, [(d["xy"], d["cls"] == GK_CLS) for d in ds])):
                if s is not None:
                    d["agent"] = k * N_TEAM + s
                    pos[fi, d["agent"]] = d["xy"]
        if f["ball"] is not None:
            pos[fi, BALL] = f["ball"]
    last_ball = None
    for fi in range(T):  # drop physically impossible ball jumps (> 50 m/s), usually false detections
        b = pos[fi, BALL]
        if np.isnan(b[0]):
            continue
        if last_ball is not None and np.linalg.norm(b - last_ball[1]) > 50.0 * max(fi - last_ball[0], 1) / fps:
            pos[fi, BALL] = np.nan
            continue
        last_ball = (fi, b)
    return pos


def attack_flip(pos, team0_attacks):
    """True if team 0 defends the right-hand goal in this footage (so build_match must rotate it)."""
    if team0_attacks != "auto":
        return team0_attacks == "left"
    for gk, sign in ((0, 1), (N_TEAM, -1)):
        x = pos[:, gk, 0]
        if np.isfinite(x).sum() > 10:
            return bool(sign * np.nanmean(x) > 0)
    return None


def render(video, frames, raw_pos, colours, stride, out_path):
    info = sv.VideoInfo.from_video_path(video)
    W, H = info.width, info.height
    rw = W // 3
    scale = rw / (PITCH[0] + 2 * MARGIN_M)
    rh = int((PITCH[1] + 2 * MARGIN_M) * scale)
    to_px = lambda p: (int((p[0] + PITCH[0] / 2 + MARGIN_M) * scale), int((PITCH[1] / 2 + MARGIN_M - p[1]) * scale))
    base = np.full((rh, rw, 3), (60, 120, 60), np.uint8)
    for a, b in PITCH_CFG.edges:
        cv2.line(base, to_px(VERTS_M[a - 1]), to_px(VERTS_M[b - 1]), (235, 235, 235), 1, cv2.LINE_AA)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps / stride, (W, H))
    for fi, frame in enumerate(sv.get_video_frames_generator(video, stride=stride)):
        if fi >= len(frames):
            break
        for d in frames[fi]["dets"]:
            if d["agent"] is None:
                continue
            x1, y1, x2, y2 = d["box"].astype(int)
            c = colours[d["agent"] // N_TEAM]
            cv2.ellipse(frame, ((x1 + x2) // 2, y2), (max((x2 - x1) // 2, 8), 6), 0, -45, 235, c, 2, cv2.LINE_AA)
            cv2.putText(frame, str(d["agent"] % N_TEAM), (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2,
                        cv2.LINE_AA)
        radar = base.copy()
        for a in range(N_AGENTS):
            p = raw_pos[fi, a]
            if np.isnan(p[0]):
                continue
            if a == BALL:
                cv2.circle(radar, to_px(p), 4, (255, 255, 255), -1, cv2.LINE_AA)
                continue
            c = colours[a // N_TEAM]
            cv2.circle(radar, to_px(p), 7, c, -1, cv2.LINE_AA)
            cv2.circle(radar, to_px(p), 7, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(radar, str(a % N_TEAM), (to_px(p)[0] - 4, to_px(p)[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
        y0, x0 = H - rh - 10, (W - rw) // 2
        frame[y0:y0 + rh, x0:x0 + rw] = cv2.addWeighted(frame[y0:y0 + rh, x0:x0 + rw], 0.2, radar, 0.8, 0)
        writer.write(frame)
    writer.release()


def main(argv=None):
    p = argparse.ArgumentParser(description="Broadcast video -> player/ball pitch coordinates (layer A)")
    p.add_argument("--video", required=True)
    p.add_argument("--out", default=None, help="output dir (default runs/<video name>)")
    p.add_argument("--models", default="data/vision")
    p.add_argument("--stride", type=int, default=5, help="process every Nth frame (5 on 25 fps video = 5 Hz)")
    p.add_argument("--period", type=int, default=1)
    p.add_argument("--team0-attacks", choices=["auto", "right", "left"], default="auto")
    p.add_argument("--swap-teams", action="store_true")
    p.add_argument("--det-conf", type=float, default=0.3)
    p.add_argument("--ball-model", action="store_true", help="fall back to the dedicated ball detector (slower)")
    p.add_argument("--max-frames", type=int, default=0, help="debug: stop after N processed frames")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--ui-dir", default="ui/data", help="also export for the formation UI here")
    p.add_argument("--no-ui", action="store_true")
    p.add_argument("--device", default="auto", help="auto picks cuda, then mps (Apple GPU), then cpu")
    a = p.parse_args(argv)

    if a.device == "auto":
        from .train import pick_device
        a.device = pick_device("auto").type
    video = Path(a.video)
    if not video.is_file():
        p.error(f"không tìm thấy video: {video} (hãy truyền đường dẫn tới file .mp4 thật)")
    out = Path(a.out or f"runs/{video.stem}")
    out.mkdir(parents=True, exist_ok=True)
    md = Path(a.models)
    player = YOLO(md / "football-player-detection.pt")
    pitch = YOLO(md / "football-pitch-detection.pt")
    ball = YOLO(md / "football-ball-detection.pt") if a.ball_model else None

    print("pass 1: detect + calibrate")
    frames, fps_out = detect(str(video), player, pitch, ball, a.stride, a.device, a.det_conf, a.max_frames)
    print("pass 2: teams + slots")
    colours = assign_teams(frames)
    if a.swap_teams:
        for f in frames:
            for d in f["dets"]:
                d["team"] = None if d["team"] is None else 1 - d["team"]
        colours = colours[::-1]
    raw = smooth(build_tracking(frames, fps_out), fps_out)
    flip = attack_flip(raw, a.team0_attacks)
    period = np.full(len(frames), a.period)
    match = build_match(video.stem, fps_out, raw, period, events=None,
                        flip=None if flip is None else {a.period: flip}, missing_ball="unknown")
    missing = ~match.ball_seen
    print(f"ball missing in {100 * missing.mean():.0f}% of frames after gap filling: "
          f"{100 * np.mean(match.state[missing] == 3) if missing.any() else 0:.0f}% of those read as out of play "
          f"(last seen at a line, or the broadcast cut away), the rest as in play but not detected")

    dets = [d for f in frames for d in f["dets"]]
    calibrated = np.mean([f["H"] is not None for f in frames])
    errs = [f["err"] for f in frames if np.isfinite(f["err"])]
    print(f"frames={len(frames)} @ {fps_out:g} Hz | calibrated {100 * calibrated:.0f}% "
          f"(median reproj {np.median(errs) if errs else np.nan:.0f} cm) | detections/frame "
          f"{len(dets) / len(frames):.1f}, placed in a slot {100 * np.mean([d['agent'] is not None for d in dets]):.0f}% "
          f"(team-ambiguous {100 * np.mean([d['team'] is None for d in dets]):.0f}%) | "
          f"players located/frame {np.median(np.isfinite(raw[:, :BALL, 0]).sum(1)):.0f} | "
          f"ball found {100 * np.isfinite(raw[:, BALL, 0]).mean():.0f}%")
    print(f"team colours BGR: team0={colours[0]} team1={colours[1]} | team0 attack flip={flip}")

    np.savez_compressed(out / "tracking.npz", pos_raw=raw, pos=match.pos, frame=[f["frame"] for f in frames],
                        fps=fps_out, calib_err=[f["err"] for f in frames])
    with open(out / "match.pkl", "wb") as f:
        pickle.dump(match, f)
    meta = {"flip": flip, "colors": ["#{2:02x}{1:02x}{0:02x}".format(*c) for c in colours], "gk": [0, N_TEAM],
            "video": str(video.resolve()), "video_offset": 0.0}
    (out / "ui_meta.json").write_text(json.dumps(meta))
    print(f"saved {out / 'tracking.npz'}, {out / 'match.pkl'}, {out / 'ui_meta.json'}")
    if not a.no_ui:
        from .export_ui import main as export_ui
        export_ui(["--match", str(out / "match.pkl"), "--out-dir", a.ui_dir])
    if not a.no_render:
        render(str(video), frames, raw, colours, a.stride, out / "radar.mp4")
        print(f"saved {out / 'radar.mp4'}")


if __name__ == "__main__":
    main()
