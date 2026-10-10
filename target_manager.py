"""Target state machine with occlusion-type reasoning.

States:  IDLE -> ACQUIRED -> OCCLUDED -> REACQUIRING -> (ACQUIRED | LOST)

NEW FEATURE (use_occlusion_reasoning=True):
  When the target disappears we classify WHY:
     EXITED   - was leaving through the frame edge
     PERSON   - was covered by another person
     OBSTACLE - vanished in the open (pillar, wall, object)
  and use a different search region, gate and timeout for each.
With use_occlusion_reasoning=False the system behaves as the baseline:
  one generic search for every case.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from appearance import AppearanceBank, extract_feature
from config import Config
from utils import Track, center, ioa, wh


class State(Enum):
    IDLE = "IDLE"
    ACQUIRED = "ACQUIRED"
    OCCLUDED = "OCCLUDED"
    REACQUIRING = "REACQUIRING"
    LOST = "LOST"


@dataclass
class TargetInfo:
    state: State
    target_id: Optional[int] = None
    box: Optional[np.ndarray] = None
    occ_type: str = "NONE"
    ambiguous: bool = False
    best_score: float = 0.0
    frames_occluded: int = 0
    search_center: Optional[Tuple[float, float]] = None
    search_radius: float = 0.0
    edge_zone: bool = False
    wide: bool = False


class TargetManager:
    def __init__(self, cfg: Config, frame_shape, use_occlusion_reasoning=True):
        self.cfg = cfg
        self.H, self.W = frame_shape[:2]
        self.use_reasoning = use_occlusion_reasoning
        self.events: List[dict] = []
        self.reset()

    # ------------------------------------------------------------------ public
    def reset(self):
        self.state = State.IDLE
        self.target_id = None
        self.bank = AppearanceBank(self.cfg.bank_size, self.cfg.new_template_sim)
        self.last_box = None
        self.last_center = None
        self.size = None
        self.vel = np.zeros(2)
        self.prev_tracks: List[Track] = []
        self.last_update = -10**9
        self._clear_occlusion()

    def _clear_occlusion(self):
        self.occ_type = "NONE"
        self.occ_start = None
        self.occluder_id = None
        self.occluder_box = None
        self.exit_point = None
        self.known_others = set()
        self.cand_id = None
        self.cand_count = 0
        self.ambiguous = False
        self.best_score = 0.0

    def select(self, tracks: List[Track], point, frame, frame_idx) -> bool:
        """Click-to-select: choose the smallest track box containing the point."""
        hits = [t for t in tracks
                if t.box[0] <= point[0] <= t.box[2] and t.box[1] <= point[1] <= t.box[3]]
        if not hits:
            return False
        t = min(hits, key=lambda t: (t.box[2] - t.box[0]) * (t.box[3] - t.box[1]))
        self.reset()
        self.target_id = t.id
        self.state = State.ACQUIRED
        self.bank.add_initial(extract_feature(frame, t.box))
        self._observe(t, tracks, frame, frame_idx, force_update=False)
        self.prev_tracks = list(tracks)
        self.events.append({"type": "selected", "frame": frame_idx, "id": t.id})
        return True

    def update(self, frame, tracks: List[Track], frame_idx) -> TargetInfo:
        by_id = {t.id: t for t in tracks}
        if self.state == State.IDLE:
            return TargetInfo(self.state, self.target_id, occ_type=self.occ_type)
        if self.state == State.LOST:
            if not self.cfg.lost_recover:
                return TargetInfo(self.state, self.target_id, occ_type=self.occ_type)
            return self._reacquire(frame, tracks, by_id, frame_idx, frame_idx - self.occ_start)

        if self.state == State.ACQUIRED:
            tgt = by_id.get(self.target_id)
            if tgt is not None:
                self._observe(tgt, tracks, frame, frame_idx)
                self.prev_tracks = list(tracks)
                return TargetInfo(State.ACQUIRED, self.target_id, tgt.box.copy())
            self._begin_occlusion(tracks, frame_idx)

        t_occ = frame_idx - self.occ_start
        # keep the occluder's position fresh while it is still visible
        if self.occluder_id is not None and self.occluder_id in by_id:
            self.occluder_box = by_id[self.occluder_id].box.copy()

        if self.state == State.OCCLUDED:
            tgt = by_id.get(self.target_id)
            if tgt is not None:                       # ByteTrack kept the same id
                self._finish_reacq(frame_idx, self.target_id, tgt, tracks, frame)
                return TargetInfo(State.ACQUIRED, self.target_id, tgt.box.copy())
            if t_occ >= self.cfg.occluded_wait:
                self.state = State.REACQUIRING
            else:
                return self._info(t_occ)

        if self.state == State.REACQUIRING:
            return self._reacquire(frame, tracks, by_id, frame_idx, t_occ)
        return self._info(t_occ)

    # --------------------------------------------------------------- internals
    def _observe(self, t: Track, tracks, frame, idx, force_update=True):
        c = center(t.box)
        if self.last_center is not None:
            self.vel = 0.7 * self.vel + 0.3 * (c - self.last_center)
        self.last_center = c
        s = wh(t.box)
        self.size = s if self.size is None else 0.8 * self.size + 0.2 * s
        self.last_box = t.box.copy()
        if force_update and self._good_quality(t, tracks) and \
                idx - self.last_update >= self.cfg.update_every:
            self.bank.update(extract_feature(frame, t.box))
            self.last_update = idx

    def _good_quality(self, t: Track, tracks):
        """Quality gate: only learn the target's look when it is clearly visible."""
        if t.conf < self.cfg.update_min_conf:
            return False
        for o in tracks:
            if o.id != t.id and ioa(t.box, o.box) > self.cfg.update_max_overlap:
                return False
        b = t.box
        if b[0] <= 2 or b[1] <= 2 or b[2] >= self.W - 2 or b[3] >= self.H - 2:
            return False
        return True

    def _begin_occlusion(self, tracks, idx):
        self._clear_occlusion()
        self.occ_start = idx
        self.state = State.OCCLUDED
        self.cand_id, self.cand_count = None, 0
        lb = self.last_box
        occluder = None
        best = 0.0
        for src in (self.prev_tracks, tracks):
            for o in src:
                if o.id == self.target_id:
                    continue
                ov = ioa(lb, o.box)
                if ov > best:
                    best, occluder = ov, o
        if not self.use_reasoning:
            self.occ_type = "UNKNOWN"
        else:
            mx, my = self.cfg.edge_margin * self.W, self.cfg.edge_margin * self.H
            b, v = lb, self.vel
            outward = (b[0] <= mx and v[0] < -self.cfg.exit_speed) or \
                      (b[2] >= self.W - mx and v[0] > self.cfg.exit_speed) or \
                      (b[1] <= my and v[1] < -self.cfg.exit_speed) or \
                      (b[3] >= self.H - my and v[1] > self.cfg.exit_speed)
            clipped = b[0] <= 2 or b[1] <= 2 or b[2] >= self.W - 2 or b[3] >= self.H - 2
            if occluder is not None and best >= self.cfg.person_overlap:
                self.occ_type = "PERSON"
                self.occluder_id, self.occluder_box = occluder.id, occluder.box.copy()
            elif outward or clipped:
                self.occ_type = "EXITED"
                self.exit_point = center(lb)
            else:
                self.occ_type = "OBSTACLE"
        self.known_others = {o.id for o in self.prev_tracks
                             if o.id != self.target_id and o.id != self.occluder_id}
        self.events.append({"type": "occluded", "frame": idx,
                            "occ_type": self.occ_type, "id": self.target_id})

    # ---- reacquisition
    def _search_region(self, t):
        """Returns (center, radius, edge_zone_flag, gate_enabled)."""
        diag = float(np.hypot(self.W, self.H))
        h = float(self.size[1]) if self.size is not None else 100.0
        if self.occ_type == "PERSON":
            ob = self.occluder_box
            r = max(wh(ob)) * (1.5 + 0.02 * t)
            return center(ob), r, False, True
        if self.occ_type == "EXITED":
            return self.exit_point, 0.5 * diag, True, True
        pred = self.last_center + self.vel * 0.9 * min(t, self.cfg.pred_cap_frames)
        sigma = min(h * (1.0 + 0.04 * t), 0.6 * diag)
        gate = (self.occ_type == "OBSTACLE")
        return pred, 3.0 * sigma, False, gate

    def _in_edge_zone(self, c):
        z = self.cfg.reentry_zone
        return c[0] <= z * self.W or c[0] >= (1 - z) * self.W or \
               c[1] <= z * self.H or c[1] >= (1 - z) * self.H

    def _weights(self, t):
        a = min(1.0, t / self.cfg.adapt_frames)
        if self.occ_type == "EXITED":
            a = max(a, 0.8)           # position tells little after leaving the frame
        w_pos = 0.45 * (1 - a) + 0.15 * a
        w_app = 0.35 * (1 - a) + 0.55 * a
        w = np.array([w_pos, w_app, 0.12, 0.08])
        return w / w.sum()

    def _score_candidates(self, frame, tracks, t, wide=False):
        if wide:     # type-specific gates failed: look everywhere, trust appearance most
            diag = float(np.hypot(self.W, self.H))
            c0, radius, edge_zone, gate = self.last_center, 0.5 * diag * 3, False, False
            w = np.array([0.10, 0.70, 0.12, 0.08])
        else:
            c0, radius, edge_zone, gate = self._search_region(t)
            w = self._weights(t)
        sigma = max(radius / 3.0, 1.0)
        h_last = float(self.size[1]) if self.size is not None else 100.0
        out = []
        for tr in tracks:
            if tr.conf < self.cfg.min_cand_conf:
                continue
            if tr.id in self.known_others and not wide:
                continue
            c = center(tr.box)
            d = float(np.linalg.norm(c - c0))
            if not wide:
                if self.occ_type == "EXITED" and not self._in_edge_zone(c):
                    continue
                if self.occ_type == "PERSON" and d > radius:
                    continue
                if self.occ_type == "OBSTACLE" and gate and d > radius:
                    continue
            pos = float(np.exp(-0.5 * (d / sigma) ** 2)) if (wide or self.occ_type != "EXITED") else 1.0
            app = self.bank.score(extract_feature(frame, tr.box))
            min_app = self.cfg.strict_min_app if wide else self.cfg.min_appearance
            if app < min_app and tr.id != self.target_id:
                continue          # wrong clothes: never a valid candidate (e.g. the occluder)
            hh = max(tr.box[3] - tr.box[1], 1.0)
            size = float(np.exp(-abs(np.log(hh / h_last)) / 0.6))
            score = float(w @ np.array([pos, app, size, tr.conf]))
            if tr.id == self.target_id:
                score = min(1.0, score + self.cfg.same_id_bonus)
            out.append((score, tr))
        out.sort(key=lambda x: -x[0])
        return out, c0, radius, edge_zone

    def _reacquire(self, frame, tracks, by_id, idx, t):
        cfg = self.cfg
        wide = t > cfg.timeouts[self.occ_type]          # stage 2: search everywhere, stricter
        if self.state == State.LOST:
            wide = True
        elif wide and t > cfg.wide_timeout:
            self.state = State.LOST                     # give up showing REACQUIRING, but keep watching
            self.events.append({"type": "lost", "frame": idx, "occ_type": self.occ_type})

        cands, c0, radius, edge_zone = self._score_candidates(frame, tracks, t, wide)
        self.ambiguous = False
        self.best_score = cands[0][0] if cands else 0.0
        accept = cfg.strict_accept if wide else cfg.accept_score
        confirm = cfg.strict_confirm if wide else cfg.confirm_frames
        if cands and cands[0][0] >= accept:
            gap = cands[0][0] - (cands[1][0] if len(cands) > 1 else 0.0)
            if gap >= cfg.margin:
                tr = cands[0][1]
                if tr.id == self.cand_id:
                    self.cand_count += 1
                else:
                    self.cand_id, self.cand_count = tr.id, 1
                if self.cand_count >= confirm:
                    self._finish_reacq(idx, tr.id, tr, tracks, frame)
                    return TargetInfo(State.ACQUIRED, self.target_id, tr.box.copy())
            else:
                self.ambiguous = True      # several similar candidates: refuse to guess
                self.cand_id, self.cand_count = None, 0
        else:
            self.cand_id, self.cand_count = None, 0
        info = self._info(t, None if wide else (c0, radius, edge_zone))
        info.wide = wide
        if wide:
            info.search_center = None
        return info

    def _finish_reacq(self, idx, new_id, tr, tracks, frame):
        old = self.target_id
        self.events.append({"type": "reacquired", "frame": idx, "old_id": old,
                            "new_id": new_id, "occ_type": self.occ_type,
                            "duration": idx - self.occ_start})
        self.target_id = new_id
        self.state = State.ACQUIRED
        self.last_center = None          # avoid a bogus velocity jump
        self.vel = np.zeros(2)
        self._observe(tr, tracks, frame, idx, force_update=False)
        self.prev_tracks = list(tracks)
        self._clear_occlusion()

    def _info(self, t, region=None):
        info = TargetInfo(self.state, self.target_id, None, self.occ_type,
                          self.ambiguous, self.best_score, t)
        if region is None and self.state in (State.OCCLUDED, State.REACQUIRING):
            c0, r, ez, _ = self._search_region(t)
            region = (c0, r, ez)
        if region is not None:
            c0, r, ez = region
            info.search_center = (float(c0[0]), float(c0[1]))
            info.search_radius = float(r)
            info.edge_zone = ez
        return info
