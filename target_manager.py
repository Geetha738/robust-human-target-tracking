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

from appearance import AppearanceBank, extract_feature, similarity
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
    confirm: Optional[Tuple[int, int]] = None   # (frames confirmed, frames needed)
    uncertain: bool = False                     # ACQUIRED but identity not yet confirmed after a crossing


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
        self.sus, self.sus_id, self.mismatch = 0, None, 0
        self.blocked = set()
        self.neg = {}                   # id -> (frame, feature) of nearby NON-target people
        self.cross = None               # crossing check in progress (see _cross_step)
        self._last_idx = 0
        self.pre = None                 # snapshot of the target just before any overlap
        self.uncertain = False
        self._clear_occlusion()

    def _clear_occlusion(self):
        self.occ_type = "NONE"
        self.occ_start = None
        self.occluder_id = None
        self.occluder_box = None
        self.exit_point = None
        self.known_others = set()
        self.blocked = set()
        self.cand_id = None
        self.cand_count = 0
        self.cand_miss = 0
        self.confirm = None
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
        self._last_idx = frame_idx
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
                backup = (self.last_box, self.last_center, self.vel.copy(), self.size)
                cross_verdict, other = self._cross_step(tgt, tracks, frame, frame_idx)
                if cross_verdict == "switch":      # after the crossing the OTHER person is the real target
                    self._switch_to(tgt, other, tracks, frame, frame_idx)
                    self.cross = self.pre = None
                    self.uncertain = False
                    self.prev_tracks = list(tracks)
                    return TargetInfo(State.ACQUIRED, self.target_id, other.box.copy())
                self._observe(tgt, tracks, frame, frame_idx)
                verdict, vtrack = ("distrust", tgt) if cross_verdict == "distrust" else ("ok", tgt)
                if verdict == "ok" and self.cfg.verify_identity and self.cross is None:
                    verdict, vtrack = self._verify(tgt, tracks, frame, frame_idx)   # slow, absolute check
                if verdict == "corrected":
                    self._switch_to(tgt, vtrack, tracks, frame, frame_idx)
                    self.prev_tracks = list(tracks)
                    return TargetInfo(State.ACQUIRED, self.target_id, vtrack.box.copy())
                if verdict in ("lost", "distrust"):   # the tracked box is not our person
                    self.last_box, self.last_center, self.vel, self.size = backup
                    self.cross = self.pre = None
                    self.uncertain = False
                    self._begin_occlusion(tracks, frame_idx)
                    self.blocked = {tgt.id}
                    if self.use_reasoning:
                        self.occ_type = "PERSON"
                        self.occluder_id, self.occluder_box = tgt.id, tgt.box.copy()
                    self.events.append({"type": "identity_lost", "frame": frame_idx, "id": tgt.id})
                else:
                    self.prev_tracks = list(tracks)
                    return TargetInfo(State.ACQUIRED, self.target_id, tgt.box.copy(), uncertain=self.uncertain)
            else:
                self._begin_occlusion(tracks, frame_idx)

        t_occ = frame_idx - self.occ_start
        # keep the occluder's position fresh while it is still visible
        if self.occluder_id is not None and self.occluder_id in by_id:
            self.occluder_box = by_id[self.occluder_id].box.copy()

        if self.state == State.OCCLUDED:
            tgt = by_id.get(self.target_id)
            if tgt is not None and self.target_id not in self.blocked and \
                    self.bank.score(extract_feature(frame, tgt.box)) >= self.cfg.same_id_min_app:
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
        if force_update and self.cross is None and self._good_quality(t, tracks) and \
                idx - self.last_update >= self.cfg.update_every:
            f = extract_feature(frame, t.box)
            if self.bank.score(f) >= self.cfg.update_min_app:     # looks like our person: safe to learn
                self.bank.update(f)
                self.last_update = idx
                self._remember_bystanders(t, tracks, frame, idx)

    def _remember_bystanders(self, target, tracks, frame, idx):
        """Store how the people around the target look, so a stranger who resembles
        a bystander more than the target is never accepted as the target."""
        for o in tracks:
            if o.id == target.id or o.conf < 0.5:
                continue
            if any(p.id != o.id and ioa(o.box, p.box) > 0.1 for p in tracks):
                continue
            f = extract_feature(frame, o.box)
            if f is not None:
                self.neg[o.id] = (idx, f)
        if len(self.neg) > self.cfg.neg_bank_size:
            for k, _ in sorted(self.neg.items(), key=lambda kv: kv[1][0])[:len(self.neg) - self.cfg.neg_bank_size]:
                del self.neg[k]

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

    # ------------------------------------------------------------------ crossing check
    def _snapshot(self):
        """Where the target was, how fast it moved and who was around, as of the previous frame."""
        if self.last_box is None or self.size is None:
            return None
        c = self.last_center if self.last_center is not None else center(self.last_box)
        return {"center": c.copy(), "vel": self.vel.copy(), "idx": self._last_idx,
                "ids": {t.id for t in self.prev_tracks}}

    def _cross_step(self, tgt, tracks, frame, idx):
        """When people overlap (or the tracked box suddenly jumps) the tracker may hand the target's ID to
        the other person. After they separate we compare everybody involved AGAINST EACH OTHER on
        appearance, motion continuity and size, instead of asking 'does this box look like the target?'
        (two different people in similar clothes both say yes).
        Returns ('ok' | 'switch' | 'distrust', track)."""
        cfg = self.cfg
        self.uncertain = False
        self._last_idx = idx - 1
        if not cfg.verify_identity or self.size is None:
            self._last_idx = idx
            return "ok", tgt
        h = float(self.size[1])
        overl = [o for o in tracks if o.id != tgt.id and
                 max(ioa(tgt.box, o.box), ioa(o.box, tgt.box)) >= cfg.cross_overlap]
        if self.cross is None:
            jumped = self.last_center is not None and \
                float(np.linalg.norm(center(tgt.box) - (self.last_center + self.vel))) > cfg.jump_factor * h
            if (overl or jumped) and self.pre is not None:
                self.cross = {"start": idx, "pre": self.pre, "partners": {o.id for o in overl},
                              "last_overlap": idx if overl else None, "ev": {}}
            else:
                self.pre = self._snapshot()
                self._last_idx = idx
                return "ok", tgt
        c = self.cross
        if overl:                                   # still overlapping: judge later
            c["partners"] |= {o.id for o in overl}
            c["last_overlap"] = idx
            self._last_idx = idx
            return "ok", tgt
        if c["last_overlap"] is not None and idx - c["last_overlap"] < cfg.cross_sep_frames:
            self._last_idx = idx
            return "ok", tgt
        pre = c["pre"]
        dt = min(idx - pre["idx"], cfg.cross_dt_cap)
        expected = pre["center"] + pre["vel"] * dt
        sigma = 0.8 * h + 0.15 * float(np.linalg.norm(pre["vel"])) * dt
        visible = {}
        for p in tracks:
            if not (p.id == tgt.id or p.id in c["partners"] or p.id not in pre["ids"]):
                continue
            if p.conf < cfg.min_cand_conf or \
                    any(ioa(p.box, o.box) > cfg.verify_overlap for o in tracks if o.id != p.id):
                continue
            f = extract_feature(frame, p.box)
            if f is None:
                continue
            app = self.bank.score(f)
            mot = float(np.exp(-0.5 * (float(np.linalg.norm(center(p.box) - expected)) / sigma) ** 2))
            sz = float(np.exp(-abs(np.log(max(p.box[3] - p.box[1], 1.0) / h)) / 0.6))
            w = cfg.cross_w
            c["ev"].setdefault(p.id, []).append((w[0] * app + w[1] * mot + w[2] * sz, app, mot))
            visible[p.id] = p

        def mean(ev):
            ev = ev[-cfg.cross_window:]
            return [float(np.mean([e[i] for e in ev])) for i in range(3)]

        carrier = c["ev"].get(tgt.id)
        self._last_idx = idx
        if not carrier:
            return "ok", tgt
        cm = mean(carrier)
        best = None                                  # strongest OTHER person involved in the crossing
        for pid, ev in c["ev"].items():
            if pid == tgt.id or pid not in visible or len(ev) < cfg.cross_min_evidence:
                continue
            m = mean(ev)
            if best is None or m[0] > best[1][0]:
                best = (pid, m)
        looks_ok = cm[0] >= cfg.cross_trust and cm[1] >= cfg.cross_trust_app
        beats_rival = best is None or cm[0] >= best[1][0] + cfg.cross_margin
        self.uncertain = not (looks_ok and beats_rival)
        if best is not None and best[1][0] >= cm[0] + cfg.cross_margin and best[1][1] >= cfg.cross_min_app:
            return "switch", visible[best[0]]       # the other person is clearly the better match
        if len(carrier) >= cfg.cross_min_evidence:
            if not self.uncertain:                  # confirmed: same person as before
                self.cross = None
                return "ok", tgt
            if cm[1] < cfg.cross_distrust_app and cm[2] < cfg.cross_distrust_motion:
                return "distrust", tgt              # neither looks nor moves like the target
        if idx - c["start"] > cfg.cross_max_frames:   # never settled in time
            self.cross, self.uncertain = None, False
            if cm[1] < cfg.cross_trust_app:
                return "distrust", tgt              # better to search again than keep a doubtful ID
        return "ok", tgt

    def _verify(self, tgt, tracks, frame, idx):
        """Is the box carrying our ID still the same person? (ByteTrack can swap IDs when people cross.)
        Returns (verdict, track): 'ok' | 'corrected' (follow `track` instead) | 'lost'."""
        cfg = self.cfg
        others = [o for o in tracks if o.id != tgt.id]
        if any(ioa(tgt.box, o.box) > cfg.verify_overlap for o in others) or tgt.conf < cfg.min_cand_conf:
            return "ok", tgt                       # crossing in progress: wait until they separate
        b = tgt.box
        if b[0] <= 2 or b[1] <= 2 or b[2] >= self.W - 2 or b[3] >= self.H - 2:
            return "ok", tgt                       # partly out of frame: looks different, don't judge
        f_t = extract_feature(frame, tgt.box)
        if f_t is None:
            return "ok", tgt
        app_t = self.bank.score(f_t)
        if app_t >= cfg.verify_low:
            self.sus, self.sus_id, self.mismatch = max(0, self.sus - 1), None, 0
            return "ok", tgt
        h = float(self.size[1]) if self.size is not None else 100.0
        c_t = center(tgt.box)
        best, best_app = None, 0.0
        for o in others:
            if o.conf < cfg.min_cand_conf or float(np.linalg.norm(center(o.box) - c_t)) > cfg.swap_radius * h:
                continue
            if any(ioa(o.box, p.box) > cfg.verify_overlap for p in tracks if p.id != o.id):
                continue
            a = self.bank.score(extract_feature(frame, o.box))
            if a > best_app:
                best, best_app = o, a
        if best is not None and best_app >= cfg.verify_high and best_app >= app_t + cfg.swap_margin:
            self.sus = self.sus + 1 if self.sus_id == best.id else 1
            self.sus_id = best.id
            if self.sus >= cfg.verify_frames:
                return "corrected", best
            return "ok", tgt
        self.sus, self.sus_id = 0, None
        if app_t < cfg.lost_identity_low:
            self.mismatch += 1
            if self.mismatch >= cfg.lost_identity_frames:
                self.mismatch = 0
                return "lost", tgt
        return "ok", tgt

    def _switch_to(self, old_t, new_t, tracks, frame, idx):
        self.events.append({"type": "id_corrected", "frame": idx, "old_id": old_t.id, "new_id": new_t.id})
        self.target_id = new_t.id
        self.sus, self.sus_id, self.mismatch = 0, None, 0
        self.last_center, self.vel = None, np.zeros(2)
        self._observe(new_t, tracks, frame, idx, force_update=False)

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
            if (tr.id in self.known_others or tr.id in self.blocked) and not wide:
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
            feat = extract_feature(frame, tr.box)
            app = self.bank.score(feat)
            same = tr.id == self.target_id
            min_app = self.cfg.strict_min_app if wide else self.cfg.min_appearance
            if same:
                min_app = min(min_app, self.cfg.same_id_min_app)
            if app < min_app:
                continue          # wrong clothes: never a valid candidate (e.g. the occluder)
            if not same and feat is not None and self.neg:
                app_n = max(similarity(nf, feat) for _, nf in self.neg.values())
                if app_n - app > self.cfg.neg_margin:
                    continue      # looks more like a known bystander than like the target
            hh = max(tr.box[3] - tr.box[1], 1.0)
            size = float(np.exp(-abs(np.log(hh / h_last)) / 0.6))
            score = float(w @ np.array([pos, app, size, tr.conf]))
            if tr.id == self.target_id:
                score = min(1.0, score + self.cfg.same_id_bonus)
            out.append((score, tr, app))
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
        need_app = cfg.strict_min_app if wide else cfg.accept_app
        if cands and cands[0][1].id == self.target_id:
            need_app = min(need_app, cfg.same_id_min_app)
        self.confirm = None
        if cands and cands[0][0] >= accept and cands[0][2] >= need_app:
            gap = cands[0][0] - (cands[1][0] if len(cands) > 1 else 0.0)
            if gap >= cfg.margin:
                tr, app = cands[0][1], cands[0][2]
                if tr.id == self.cand_id:
                    self.cand_count += 1
                else:
                    self.cand_id, self.cand_count = tr.id, 1
                self.cand_miss = 0
                strong = app >= cfg.fast_app and gap >= cfg.fast_margin
                need = (cfg.fast_strict_confirm if strong else confirm) if wide else \
                       (cfg.fast_confirm_frames if strong else confirm)
                if tr.id == self.target_id and app >= cfg.same_id_fast_app:
                    need = min(need, cfg.same_id_confirm)     # ByteTrack kept his id: no reason to wait
                self.confirm = (self.cand_count, need)
                if self.cand_count >= need:
                    self._finish_reacq(idx, tr.id, tr, tracks, frame)
                    return TargetInfo(State.ACQUIRED, self.target_id, tr.box.copy())
            else:
                self.ambiguous = True      # several similar candidates: refuse to guess
                self.cand_id, self.cand_count, self.cand_miss = None, 0, 0
        else:
            self._candidate_missed()
        info = self._info(t, None if wide else (c0, radius, edge_zone))
        info.wide = wide
        info.confirm = self.confirm
        if wide:
            info.search_center = None
        return info

    def _candidate_missed(self):
        """One weak frame (e.g. person half hidden) must not erase the progress made so far."""
        self.cand_miss += 1
        if self.cand_miss > self.cfg.confirm_tolerance:
            self.cand_id, self.cand_count, self.cand_miss = None, 0, 0

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
        self.cross = self.pre = None
        self.uncertain = False
        self._last_idx = idx
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
