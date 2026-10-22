"""Quick ground-truth annotation (needed to MEASURE accuracy).

  python annotate_gt.py --source clip.mp4 --tracks clip.json --step 5 --out clip_gt.csv

Every STEP frames it shows the detections. Then:
   click inside the target's box -> saves that box as ground truth
   n  -> the target is NOT in the frame (or not detected): saves 'none'
   b  -> go back one annotated frame      q -> save and quit
Only frames you annotate are used when scoring."""
import argparse
import cv2
import numpy as np

from pipeline import cached_tracked_frames, load_cache
from visualize import _label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-width", type=int, default=1100)
    a = ap.parse_args()

    cache = load_cache(a.tracks)
    frames = [(i, f, t) for i, f, t in cached_tracked_frames(a.source, cache) if i % a.step == 0]
    gt, pos, click, s = {}, 0, {"p": None}, {"v": 1.0}
    win = "annotate target"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, lambda e, x, y, f, p: click.update(p=(x / s["v"], y / s["v"]))
                         if e == cv2.EVENT_LBUTTONDOWN else None)
    while 0 <= pos < len(frames):
        idx, frame, tracks = frames[pos]
        vis = frame.copy()
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 215, 255), 2)
            _label(vis, f"#{t.id}", (x1, max(18, y1 - 2)), (0, 0, 0), (0, 215, 255), 0.5, 1)
        done = f"saved: {gt[idx] if gt[idx] is None else 'box'}" if idx in gt else "not annotated"
        cv2.putText(vis, f"frame {idx} ({pos + 1}/{len(frames)}) {done} | click target, n=absent, b=back, q=quit",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        s["v"] = min(1.0, a.max_width / vis.shape[1])
        cv2.imshow(win, cv2.resize(vis, None, fx=s["v"], fy=s["v"]) if s["v"] < 1 else vis)
        k = cv2.waitKey(30) & 0xFF
        p, click["p"] = click["p"], None
        if p is not None:
            hit = [t for t in tracks if t.box[0] <= p[0] <= t.box[2] and t.box[1] <= p[1] <= t.box[3]]
            if hit:
                gt[idx] = min(hit, key=lambda t: (t.box[2] - t.box[0]) * (t.box[3] - t.box[1])).box
                pos += 1
        elif k == ord("n"):
            gt[idx] = None
            pos += 1
        elif k == ord("b"):
            pos = max(0, pos - 1)
        elif k == ord("q"):
            break
    with open(a.out, "w") as f:
        f.write("# frame,x1,y1,x2,y2   (or: frame,none)\n")
        for i in sorted(gt):
            f.write(f"{i},none\n" if gt[i] is None else f"{i},{','.join(f'{v:.1f}' for v in gt[i])}\n")
    print(f"saved {len(gt)} annotated frames to {a.out}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
