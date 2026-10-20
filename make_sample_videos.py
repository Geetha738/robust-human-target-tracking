"""Create two sample videos of the 'blocker takes the target's ID' problem, with tracker output and ground truth.

  python make_sample_videos.py
Output (in sample_videos/): <name>.avi, <name>_tracks.json (what YOLO+ByteTrack would hand over), <name>_gt.csv
Then:  python evaluate.py --source sample_videos/blocker_moving.avi --tracks sample_videos/blocker_moving_tracks.json \\
         --select-point 115,250 --select-frame 3 --gt sample_videos/blocker_moving_gt.csv
These are controlled clips: the tracker IDs are scripted to do what ByteTrack sometimes does (give the target's ID to
the person who blocks him). Use run.py on real videos for real results."""
import json
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "tests"))
import test_synthetic as T  # noqa: E402

CLIPS = {
    "blocker_moving": (T.scene_blocker("steal_new"), 130),    # moving blocker, half-similar outfit, steals ID 1
    "blocker_static": (T.scene_static_blocker, 140),          # target walks behind a standing person
}


def main():
    out = os.path.join(HERE, "sample_videos")
    os.makedirs(out, exist_ok=True)
    for name, (scene, n) in CLIPS.items():
        vw = cv2.VideoWriter(os.path.join(out, name + ".avi"), cv2.VideoWriter_fourcc(*"MJPG"), 30, (T.W, T.H))
        tracks, gt = {}, ["# frame,x1,y1,x2,y2   (or: frame,none)"]
        for f in range(n):
            objs = scene(f)
            vis = [(i, b, c) for i, b, c, v, _ in objs if v]
            vw.write(T.render([(b, c) for _, b, c in vis]))
            tracks[str(f)] = [[i, *map(float, b), 0.9] for i, b, _ in vis]
            tgt = [b for _, b, _, v, is_t in objs if is_t and v]
            gt.append(f"{f},{','.join(f'{x:.1f}' for x in tgt[0])}" if tgt else f"{f},none")
        vw.release()
        json.dump(tracks, open(os.path.join(out, name + "_tracks.json"), "w"))
        open(os.path.join(out, name + "_gt.csv"), "w").write("\n".join(gt))
        print("wrote", name)


if __name__ == "__main__":
    main()
