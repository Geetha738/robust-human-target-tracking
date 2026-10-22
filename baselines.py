"""The 'existing approach': plain YOLO + ByteTrack following the clicked ID.
No appearance memory, no reacquisition. If the ID changes after an occlusion, the target is gone."""
from target_manager import State, TargetInfo


class PlainFollower:
    def __init__(self, cfg, frame_shape, use_occlusion_reasoning=False):
        self.cfg = cfg
        self.events = []
        self.reset()

    def reset(self):
        self.state, self.target_id, self.occ_start = State.IDLE, None, None

    def select(self, tracks, point, frame, idx):
        hits = [t for t in tracks
                if t.box[0] <= point[0] <= t.box[2] and t.box[1] <= point[1] <= t.box[3]]
        if not hits:
            return False
        t = min(hits, key=lambda t: (t.box[2] - t.box[0]) * (t.box[3] - t.box[1]))
        self.reset()
        self.target_id, self.state = t.id, State.ACQUIRED
        self.events.append({"type": "selected", "frame": idx, "id": t.id})
        return True

    def update(self, frame, tracks, idx):
        if self.state == State.IDLE:
            return TargetInfo(State.IDLE)
        tgt = {t.id: t for t in tracks}.get(self.target_id)
        if tgt is not None:
            if self.state == State.OCCLUDED:
                self.events.append({"type": "reacquired", "frame": idx, "old_id": self.target_id,
                                    "new_id": self.target_id, "occ_type": "UNKNOWN",
                                    "duration": idx - self.occ_start})
            self.state = State.ACQUIRED
            return TargetInfo(State.ACQUIRED, self.target_id, tgt.box.copy())
        if self.state == State.ACQUIRED:
            self.state, self.occ_start = State.OCCLUDED, idx
            self.events.append({"type": "occluded", "frame": idx, "occ_type": "UNKNOWN", "id": self.target_id})
        return TargetInfo(State.OCCLUDED, self.target_id, occ_type="UNKNOWN", frames_occluded=idx - self.occ_start)
