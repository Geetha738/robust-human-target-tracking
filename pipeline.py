"""Frame sources and the main processing loop (shared by live run and replay)."""
import json
import time
import cv2
import numpy as np
from utils import Track


def video_frames(source):
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, frame
        idx += 1
    cap.release()


def live_tracked_frames(source, tracker, cache=None, timing=None):
    for idx, frame in video_frames(source):
        t0 = time.perf_counter()
        tracks = tracker(frame)
        if timing is not None:
            timing.append(time.perf_counter() - t0)
        if cache is not None:
            cache[str(idx)] = [[t.id, *map(float, t.box), t.conf] for t in tracks]
        yield idx, frame, tracks


def cached_tracked_frames(source, cache):
    for idx, frame in video_frames(source):
        rows = cache.get(str(idx), [])
        yield idx, frame, [Track(int(r[0]), np.array(r[1:5], float), float(r[5])) for r in rows]


def save_cache(cache, path):
    with open(path, "w") as f:
        json.dump(cache, f)


def load_cache(path):
    with open(path) as f:
        return json.load(f)


class AutoSelector:
    """Selects whoever is at `point` on frame `frame_idx` (for repeatable experiments)."""
    def __init__(self, point, frame_idx):
        self.point, self.frame_idx = point, frame_idx

    def __call__(self, idx):
        return self.point if idx >= self.frame_idx and self.point is not None else None

    def done(self):
        self.point = None


def run_session(frames, make_manager, recorder, selector=None, on_frame=None):
    """frames yields (idx, frame, tracks). make_manager(frame_shape) builds the TargetManager."""
    manager = None
    last = time.time()
    for idx, frame, tracks in frames:
        if manager is None:
            manager = make_manager(frame.shape)
        if selector is not None:
            p = selector(idx)
            if p is not None and manager.select(tracks, p, frame, idx):
                selector.done()
        t0 = time.perf_counter()
        info = manager.update(frame, tracks, idx)
        mgr_dt = time.perf_counter() - t0
        now = time.time()
        recorder.record(idx, info, now - last, mgr_dt)
        last = now
        if on_frame is not None and on_frame(idx, frame, tracks, info, manager) is False:
            break
    return manager
