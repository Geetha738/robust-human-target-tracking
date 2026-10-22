"""LIVE side-by-side demo: EXISTING approach (left) vs OURS (right) on the same video / camera.

  python compare_live.py --source video.mp4
  python compare_live.py --source 0                 # webcam
Click a person (in either panel): both systems start following that person.
Then watch what each does when people block the target.  Keys: space=pause/reselect  r=reset  q=quit
"""
import argparse
import json

import cv2
import numpy as np

from baselines import PlainFollower
from config import Config
from pipeline import AutoSelector, cached_tracked_frames, live_tracked_frames
from target_manager import State, TargetManager
from visualize import draw


def run_compare(frames, cfg, display=True, select_point=None, select_frame=0,
                max_width=1500, save_video=None):
    mgrs, W = None, None
    clicked, disp = {"p": None}, {"s": 1.0}
    stats = {"frames": 0, "existing_locked": 0, "ours_locked": 0}
    writer = None
    paused_once = False
    auto = AutoSelector(select_point, select_frame) if select_point else None
    win = "EXISTING (left)  vs  OURS (right)"

    def on_mouse(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN and W:
            cx = x / disp["s"]
            clicked["p"] = (cx % W, y / disp["s"])       # either panel maps to the same person

    if display:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, on_mouse)

    def render(frame, tracks, infos, msg=None):
        n = max(1, stats["frames"])
        left = draw(frame, tracks, infos[0],
                    hint=msg or f"EXISTING YOLO+ByteTrack | locked {100 * stats['existing_locked'] / n:.0f}% of frames")
        right = draw(frame, tracks, infos[1],
                     hint=msg or f"OURS | locked {100 * stats['ours_locked'] / n:.0f}% of frames")
        both = np.hstack([left, right])
        disp["s"] = min(1.0, max_width / both.shape[1])
        return both, (cv2.resize(both, None, fx=disp["s"], fy=disp["s"]) if disp["s"] < 1 else both)

    def select_all(p, frame, tracks, idx):
        ok = [m.select(tracks, p, frame, idx) for m in mgrs]
        return all(ok)

    def choose_loop(idx, frame, tracks, text):
        while True:
            infos = [m.update(frame, tracks, idx) for m in mgrs]
            cv2.imshow(win, render(frame, tracks, infos, text)[1])
            k = cv2.waitKey(30) & 0xFF
            p, clicked["p"] = clicked["p"], None
            if p is not None and select_all(p, frame, tracks, idx):
                text = "Selected in both. Press SPACE to play (click another person to change)"
                continue
            if k == ord(" "):
                return True
            if k == ord("q"):
                return False

    for idx, frame, tracks in frames:
        if mgrs is None:
            W = frame.shape[1]
            mgrs = [PlainFollower(cfg, frame.shape), TargetManager(cfg, frame.shape, True)]
        if auto is not None:
            p = auto(idx)
            if p is not None and select_all(p, frame, tracks, idx):
                auto.done()
        p, clicked["p"] = clicked["p"], None
        if p is not None:
            select_all(p, frame, tracks, idx)

        infos = [m.update(frame, tracks, idx) for m in mgrs]
        both_sel = all(m.state != State.IDLE for m in mgrs)
        if both_sel:
            stats["frames"] += 1
            stats["existing_locked"] += infos[0].state == State.ACQUIRED
            stats["ours_locked"] += infos[1].state == State.ACQUIRED and not infos[1].uncertain

        if display:
            if not both_sel and tracks and not paused_once and auto is None:
                paused_once = True
                if not choose_loop(idx, frame, tracks, "Click a person to select, then SPACE to play"):
                    break
                infos = [m.update(frame, tracks, idx) for m in mgrs]
            full, shown = render(frame, tracks, infos)
            if save_video:
                if writer is None:
                    writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*"mp4v"), 30,
                                             (full.shape[1], full.shape[0]))
                writer.write(full)
            cv2.imshow(win, shown)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("r"):
                for m in mgrs:
                    m.reset()
                stats.update(frames=0, existing_locked=0, ours_locked=0)
                paused_once = False
            if k == ord(" "):
                if not choose_loop(idx, frame, tracks, "PAUSED - click a person to (re)select, SPACE to play"):
                    break
    if writer:
        writer.release()
    if display:
        cv2.destroyAllWindows()

    def ev(m):
        e = m.events
        return {"reacquired": sum(x["type"] == "reacquired" for x in e),
                "id_switches": sum(x["type"] == "reacquired" and x["old_id"] != x["new_id"] for x in e),
                "occlusions": sum(x["type"] == "occluded" for x in e)}
    n = max(1, stats["frames"])
    out = {"frames_after_selection": stats["frames"],
           "existing": {"locked_pct": round(100 * stats["existing_locked"] / n, 1), **(ev(mgrs[0]) if mgrs else {})},
           "ours": {"locked_pct": round(100 * stats["ours_locked"] / n, 1), **(ev(mgrs[1]) if mgrs else {})}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--no-identity-check", action="store_true")
    ap.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml | botsort.yaml | botsort_reid.yaml | ...")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-width", type=int, default=1500, help="width of the combined window")
    ap.add_argument("--save-video", default=None)
    ap.add_argument("--select-point", default=None)
    ap.add_argument("--select-frame", type=int, default=0)
    ap.add_argument("--no-display", action="store_true")
    a = ap.parse_args()
    from detector import PersonTracker
    cfg = Config(model=a.model, imgsz=a.imgsz, tracker=a.tracker, verify_identity=not a.no_identity_check)
    pt = tuple(float(v) for v in a.select_point.split(",")) if a.select_point else None
    res = run_compare(live_tracked_frames(a.source, PersonTracker(cfg)), cfg, not a.no_display,
                      pt, a.select_frame, a.max_width, a.save_video)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
