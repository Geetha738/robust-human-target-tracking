"""Compare the EXISTING approach with ours on the same cached detections (fair, repeatable).

  1) python run.py --source clip.mp4 --no-display --select-point 320,240 --select-frame 10 --save-tracks clip.json
  2) python annotate_gt.py --source clip.mp4 --tracks clip.json --out clip_gt.csv
  3) python evaluate.py --source clip.mp4 --tracks clip.json --select-point 320,240 --select-frame 10 --gt clip_gt.csv

Variants
  existing : plain YOLO11n + ByteTrack following the clicked ID (nothing else)
  base     : + our reacquisition (appearance, prediction, ambiguity, quality-gated memory, wide search)
  ours     : base + occlusion-type reasoning (the new feature)
"""
import argparse
import json

from baselines import PlainFollower
from config import Config
from metrics import MetricsRecorder, load_gt
from pipeline import AutoSelector, cached_tracked_frames, load_cache, run_session
from target_manager import TargetManager

VARIANTS = {
    "existing": lambda cfg, shape: PlainFollower(cfg, shape),
    "base": lambda cfg, shape: TargetManager(cfg, shape, False),
    "ours": lambda cfg, shape: TargetManager(cfg, shape, True),
}


def run_variant(name, source, cache, point, sel_frame, gt):
    cfg = Config()
    rec = MetricsRecorder(gt)
    mgr = run_session(cached_tracked_frames(source, cache),
                      lambda shape: VARIANTS[name](cfg, shape), rec, AutoSelector(point, sel_frame))
    return rec.summary(mgr.events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--select-point", required=True)
    ap.add_argument("--select-frame", type=int, default=0)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--out", default="comparison.json")
    a = ap.parse_args()
    point = tuple(float(v) for v in a.select_point.split(","))
    cache, gt = load_cache(a.tracks), (load_gt(a.gt) if a.gt else None)
    res = {n: run_variant(n, a.source, cache, point, a.select_frame, gt) for n in VARIANTS}

    det_ms = cache.get("_meta", {}).get("det_ms_avg")
    for n in res:                                   # whole-pipeline speed estimate
        m = res[n].get("manager_ms_avg", 0)
        res[n]["detector_ms_avg"] = det_ms
        if det_ms:
            res[n]["pipeline_fps_est"] = round(1000.0 / (det_ms + m), 1)

    rows = [("tracking_accuracy", "tracking accuracy (higher=better)"),
            ("correct_reacquisitions", "correct reacquisitions"),
            ("false_reacquisitions", "WRONG reacquisitions"),
            ("false_lock_when_absent", "false lock while target absent"),
            ("wrong_lock_frames", "frames locked on wrong person"),
            ("id_switches", "ID switches"),
            ("occlusion_events", "occlusion events"),
            ("lost", "times declared LOST"),
            ("avg_reacq_frames", "avg frames to reacquire"),
            ("ambiguous_frames", "frames flagged AMBIGUOUS"),
            ("detector_ms_avg", "YOLO+ByteTrack ms/frame"),
            ("manager_ms_avg", "our manager ms/frame"),
            ("pipeline_fps_est", "estimated pipeline FPS")]
    print(f"\n{'metric':<36}{'existing':>12}{'base':>12}{'ours':>12}")
    for k, label in rows:
        vals = [res[n].get(k, "-") for n in ("existing", "base", "ours")]
        vals = ["-" if v is None else v for v in vals]
        if all(v == "-" for v in vals):
            continue
        print(f"{label:<36}" + "".join(f"{str(v):>12}" for v in vals))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
