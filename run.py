"""Live/offline tracking with click-to-select.

Examples:
  python run.py --source video.mp4
  python run.py --source video.mp4 --baseline                 # without the new feature
  python run.py --source video.mp4 --no-display --select-point 320,240 --select-frame 10
Keys: video pauses at first people so you can click one | space=pause/play | s=pause+reselect | r=reset | q=quit
"""
import argparse
import json
import os

import cv2

from config import Config
from detector import PersonTracker
from metrics import MetricsRecorder, load_gt
from pipeline import AutoSelector, live_tracked_frames, run_session, save_cache
from target_manager import TargetManager
from visualize import draw


def parse_point(s):
    return tuple(float(v) for v in s.split(",")) if s else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="video path or webcam index")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--no-identity-check", action="store_true", help="never auto-correct the target ID (most conservative)")
    ap.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml | botsort.yaml | botsort_reid.yaml | ...")
    ap.add_argument("--imgsz", type=int, default=640, help="detector input size; smaller = faster (e.g. 480, 320)")
    ap.add_argument("--baseline", action="store_true", help="disable occlusion-type reasoning")
    ap.add_argument("--no-display", action="store_true")
    ap.add_argument("--max-width", type=int, default=1100, help="shrink the display window to this width")
    ap.add_argument("--select-point", default=None, help="x,y to auto-select (needs --select-frame)")
    ap.add_argument("--select-frame", type=int, default=0)
    ap.add_argument("--gt", default=None, help="ground-truth CSV: frame,x1,y1,x2,y2")
    ap.add_argument("--save-video", default=None)
    ap.add_argument("--save-tracks", default=None, help="cache detections+IDs for evaluate.py")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    cfg = Config(model=args.model, imgsz=args.imgsz, tracker=args.tracker, verify_identity=not args.no_identity_check)
    os.makedirs(args.out, exist_ok=True)
    tracker = PersonTracker(cfg)
    recorder = MetricsRecorder(load_gt(args.gt) if args.gt else None)
    cache = {} if args.save_tracks else None
    det_times = []

    clicked = {"p": None}
    selector = AutoSelector(parse_point(args.select_point), args.select_frame) if args.select_point else None
    writer = {"w": None}
    disp = {"s": 1.0}
    asked = {"paused_once": False}
    win = "Target tracking"
    if not args.no_display:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

        def on_mouse(e, x, y, f, p):
            if e == cv2.EVENT_LBUTTONDOWN:   # map click back to full-size frame coordinates
                clicked["p"] = (x / disp["s"], y / disp["s"])
        cv2.setMouseCallback(win, on_mouse)

    def show(vis):
        h, w = vis.shape[:2]
        disp["s"] = min(1.0, args.max_width / w)
        cv2.imshow(win, cv2.resize(vis, None, fx=disp["s"], fy=disp["s"]) if disp["s"] < 1.0 else vis)

    def try_click(idx, frame, tracks, manager):
        p, clicked["p"] = clicked["p"], None
        return p is not None and manager.select(tracks, p, frame, idx)

    def choose_loop(idx, frame, tracks, manager, msg):
        """Paused: show boxes, wait for a click (select) or SPACE (resume) or q (quit)."""
        while True:
            info = manager.update(frame, tracks, idx)
            show(draw(frame, tracks, info, msg))
            k = cv2.waitKey(30) & 0xFF
            if try_click(idx, frame, tracks, manager):
                info = manager.update(frame, tracks, idx)
                show(draw(frame, tracks, info, "Selected! Press SPACE to play"))
                msg = "Selected! Press SPACE to play  (click another person to change)"
                continue
            if k == ord(" "):
                return True
            if k == ord("q"):
                return False

    def _fps():
        x = recorder.fps_samples[-30:]
        return sum(x) / len(x) if x else None

    def on_frame(idx, frame, tracks, info, manager):
        if args.no_display:
            vis = draw(frame, tracks, info)
        else:
            if try_click(idx, frame, tracks, manager):          # click while playing
                info = manager.update(frame, tracks, idx)
            # first frame that has people: pause so the user can pick one calmly
            if manager.state.value == "IDLE" and tracks and not asked["paused_once"] and not selector:
                asked["paused_once"] = True
                if not choose_loop(idx, frame, tracks, manager,
                                   "Click a person to select, then press SPACE to play"):
                    return False
                info = manager.update(frame, tracks, idx)
            vis = draw(frame, tracks, info, fps=_fps())
        if args.save_video:
            if writer["w"] is None:
                h, w = vis.shape[:2]
                writer["w"] = cv2.VideoWriter(args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
            writer["w"].write(vis)
        if not args.no_display:
            show(vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                return False
            if k == ord("r"):
                manager.reset()
                asked["paused_once"] = False
            if k == ord("s") or k == ord(" "):                  # pause and (re)select
                if not choose_loop(idx, frame, tracks, manager,
                                   "PAUSED - click a person to (re)select, SPACE to play"):
                    return False
        return True

    manager = run_session(live_tracked_frames(args.source, tracker, cache, det_times),
                          lambda shape: TargetManager(cfg, shape, not args.baseline),
                          recorder, selector, on_frame)

    if writer["w"] is not None:
        writer["w"].release()
    if args.save_tracks:
        cache["_meta"] = {"det_ms_avg": round(1000 * sum(det_times) / max(1, len(det_times)), 2)}
        save_cache(cache, args.save_tracks)
    recorder.write_csv(os.path.join(args.out, "log.csv"))
    summary = recorder.summary(manager.events if manager else [])
    if det_times:
        summary["detector_ms_avg"] = round(1000 * sum(det_times) / len(det_times), 2)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"summary": summary, "events": manager.events if manager else []}, f, indent=2, default=str)
    print(json.dumps(summary, indent=2))
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
