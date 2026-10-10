"""Per-frame logging and summary metrics (with optional ground truth)."""
import csv
import numpy as np
from utils import iou


def load_gt(path):
    gt = {}
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#") or not row[0].strip().lstrip("-").isdigit():
                continue
            gt[int(row[0])] = None if row[1].strip().lower() == "none" else \
                np.array([float(v) for v in row[1:5]])
    return gt


class MetricsRecorder:
    def __init__(self, gt=None):
        self.gt = gt or {}
        self.rows = []
        self.fps_samples = []
        self.mgr_times = []

    def record(self, idx, info, dt=None, mgr_dt=None):
        self.rows.append((idx, info))
        if dt:
            self.fps_samples.append(1.0 / dt)
        if mgr_dt is not None:
            self.mgr_times.append(mgr_dt)

    def write_csv(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "state", "occ_type", "target_id", "x1", "y1", "x2", "y2", "ambiguous"])
            for idx, i in self.rows:
                b = ["", "", "", ""] if i.box is None else [round(float(v), 1) for v in i.box]
                w.writerow([idx, i.state.value, i.occ_type, i.target_id, *b, int(i.ambiguous)])

    def summary(self, events):
        n = len(self.rows)
        s = {"frames": n}
        counts = {}
        for _, i in self.rows:
            counts[i.state.value] = counts.get(i.state.value, 0) + 1
        s["state_frames"] = counts
        occ = [e for e in events if e["type"] == "occluded"]
        rea = [e for e in events if e["type"] == "reacquired"]
        lost = [e for e in events if e["type"] == "lost"]
        s["occlusion_events"] = len(occ)
        s["reacquired"] = len(rea)
        s["lost"] = len(lost)
        s["id_switches"] = sum(1 for e in rea if e["old_id"] != e["new_id"])
        s["avg_reacq_frames"] = round(float(np.mean([e["duration"] for e in rea])), 1) if rea else None
        s["ambiguous_frames"] = sum(1 for _, i in self.rows if i.ambiguous)
        if self.mgr_times:
            s["manager_ms_avg"] = round(1000 * float(np.mean(self.mgr_times)), 2)
        if self.fps_samples:
            s["avg_fps"] = round(float(np.mean(self.fps_samples)), 1)
        if self.gt:
            by_frame = dict(self.rows)
            vis = [f for f, g in self.gt.items() if g is not None and f in by_frame]
            absent = [f for f, g in self.gt.items() if g is None and f in by_frame]
            correct = wrong = 0
            for f in vis:
                i = by_frame[f]
                if i.state.value == "ACQUIRED" and i.box is not None:
                    v = iou(i.box, self.gt[f])
                    correct += v >= 0.5
                    wrong += v < 0.3
            s["tracking_accuracy"] = round(correct / max(1, len(vis)), 3)
            s["wrong_lock_frames"] = int(wrong)
            # target not in the frame, yet the system claims to be tracking someone
            s["false_lock_when_absent"] = sum(1 for f in absent if by_frame[f].state.value == "ACQUIRED")
            # judge each reacquisition on the first annotated frame at/after it (GT is sparse)
            ann = sorted(self.gt)
            correct_re = false_re = unverified = 0
            for e in rea:
                nxt = next((f for f in ann if f >= e["frame"] and f in by_frame), None)
                if nxt is None or nxt - e["frame"] > 30:
                    unverified += 1
                    continue
                g, i = self.gt[nxt], by_frame[nxt]
                ok = g is not None and i.state.value == "ACQUIRED" and i.box is not None \
                    and iou(i.box, g) >= 0.5
                correct_re += ok
                false_re += (not ok)
            s["false_reacquisitions"] = false_re
            s["correct_reacquisitions"] = correct_re
            if unverified:
                s["unverified_reacquisitions"] = unverified
        return s
