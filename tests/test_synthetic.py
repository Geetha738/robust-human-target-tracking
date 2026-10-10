"""Offline tests with synthetic 'people' (coloured rectangles) and fake ByteTrack IDs.
Run:  python tests/test_synthetic.py     (no YOLO / GPU / internet needed)"""
import os, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from metrics import MetricsRecorder
from pipeline import AutoSelector, run_session
from target_manager import TargetManager
from utils import Track, iou

W, H = 640, 480
RED_BLUE = ((30, 30, 200), (200, 60, 30))      # torso BGR, legs BGR  (the target's clothes)
GREEN_BLACK = ((40, 180, 40), (30, 30, 30))
YELLOW_WHITE = ((30, 220, 230), (230, 230, 230))


def box_at(cx, cy, w=44, h=110):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], float)


def render(people):
    rng = np.random.default_rng(0)
    img = (np.full((H, W, 3), 120, np.uint8) + rng.integers(0, 8, (H, W, 3), dtype=np.uint8))
    for box, (torso, legs) in people:
        x1, y1, x2, y2 = [int(v) for v in box]
        h = y2 - y1
        cv2.rectangle(img, (x1, y1), (x2, y1 + int(0.15 * h)), (150, 170, 220), -1)
        cv2.rectangle(img, (x1, y1 + int(0.15 * h)), (x2, y1 + int(0.5 * h)), torso, -1)
        cv2.rectangle(img, (x1, y1 + int(0.5 * h)), (x2, y2), legs, -1)
    return img


def simulate(scene_fn, n_frames, reasoning, select_point, select_frame=3):
    """scene_fn(f) -> list of (track_id, box, clothes, visible, is_target)"""
    cfg = Config()
    rec = MetricsRecorder()
    truth = {}

    def frames():
        for f in range(n_frames):
            objs = scene_fn(f)
            vis = [(i, b, c) for i, b, c, v, _ in objs if v]
            for i, b, c, v, is_t in objs:
                if is_t and v:
                    truth[f] = b
            yield f, render([(b, c) for _, b, c in vis]), [Track(i, b, 0.9) for i, b, _ in vis]

    mgr = run_session(frames(), lambda s: TargetManager(cfg, s, reasoning), rec,
                      AutoSelector(select_point, select_frame))
    return mgr, rec, truth


def final_ok(rec, truth, last_frames=15):
    """target locked on the REAL person at the end of the clip?"""
    ok = 0
    for f in range(len(rec.rows) - last_frames, len(rec.rows)):
        info = rec.rows[f][1]
        if info.state.value == "ACQUIRED" and f in truth and iou(info.box, truth[f]) > 0.5:
            ok += 1
    return ok >= last_frames - 2


# ---------------------------------------------------------------- scenes
def scene_obstacle(f):
    cx = 100 + 4 * f
    hidden = 60 <= f < 85                       # behind a pillar around x=340..440
    tid = 1 if f < 60 else 10                   # ByteTrack gives a new id after the gap
    ppl = [(tid, box_at(cx, 260), RED_BLUE, not hidden, True),
           (2, box_at(520 - 2 * f % 300, 380), GREEN_BLACK, True, False)]
    return ppl


def scene_person(f):
    cx = 200 + 3 * f
    occ_cx = 345 + 6 * (f - 49)                  # another person steps in front of the target
    hidden = 50 <= f < 72
    tid = 1 if f < 50 else 11
    return [(tid, box_at(cx, 250), RED_BLUE, not hidden, True),
            (2, box_at(occ_cx, 250), GREEN_BLACK, 44 <= f < 100, False)]


def scene_exit_lookalike(f):
    """Target leaves right edge at ~f=40, a bystander in IDENTICAL clothes appears mid-frame
    at f=80, the real target re-enters at the right edge at f=140."""
    ppl = []
    if f < 42:
        ppl.append((1, box_at(300 + 8 * f, 250), RED_BLUE, True, True))
    if f >= 80:
        ppl.append((7, box_at(260, 300), RED_BLUE, True, False))   # lookalike, mid frame
    if f >= 140:
        ppl.append((9, box_at(W - 40 - 2 * (f - 140), 250), RED_BLUE, True, True))
    return ppl


def scene_two_lookalikes(f):
    """After an obstacle occlusion TWO identical-looking people appear near the target."""
    ppl = []
    if f < 60:
        ppl.append((1, box_at(250 + 2 * f, 250), RED_BLUE, True, True))
    else:
        ppl.append((20, box_at(375, 250), RED_BLUE, f >= 75, False))
        ppl.append((21, box_at(445, 250), RED_BLUE, f >= 75, False))
    return ppl


def scene_long_block(f):
    """Crowd blocks the target for 150 frames; he re-appears FAR from the blocker under a new id."""
    ppl = []
    if f < 40:
        ppl.append((1, box_at(300, 250), RED_BLUE, True, True))
    if f >= 190:
        ppl.append((12, box_at(150, 300), RED_BLUE, True, True))
    if 35 <= f < 80:
        ppl.append((2, box_at(310 + 8 * max(0, f - 60), 250), GREEN_BLACK, True, False))
    ppl.append((3, box_at(500, 380), YELLOW_WHITE, True, False))
    return ppl


# ---------------------------------------------------------------- tests
def test_obstacle():
    for reasoning in (True, False):
        mgr, rec, truth = simulate(scene_obstacle, 160, reasoning, (100 + 4 * 3, 260))
        assert final_ok(rec, truth), f"obstacle failed (reasoning={reasoning})"
        types = [e["occ_type"] for e in mgr.events if e["type"] == "occluded"]
        assert types == (["OBSTACLE"] if reasoning else ["UNKNOWN"]), types
    print("PASS obstacle occlusion (new id after gap is re-linked)")


def test_person():
    mgr, rec, truth = simulate(scene_person, 110, True, (200 + 9, 250))
    types = [e["occ_type"] for e in mgr.events if e["type"] == "occluded"]
    assert types == ["PERSON"], types
    assert final_ok(rec, truth), "person occlusion failed"
    print("PASS person occlusion classified as PERSON and re-linked")


def test_exit_lookalike():
    sp = (300 + 8 * 3, 250)
    mgr, rec, truth = simulate(scene_exit_lookalike, 200, True, sp)
    types = [e["occ_type"] for e in mgr.events if e["type"] == "occluded"]
    assert types == ["EXITED"], types
    assert final_ok(rec, truth), "with reasoning: should re-lock the real target at the edge"
    bm, brec, btruth = simulate(scene_exit_lookalike, 200, False, sp)
    base_ok = final_ok(brec, btruth)
    bad = [e for e in bm.events if e["type"] == "reacquired"]
    print("PASS exit + lookalike: reasoning=correct;",
          "baseline locked onto the lookalike (wrong)" if not base_ok else "baseline also correct",
          f"| baseline events: {[(e['frame'], e['new_id']) for e in bad]}")
    return base_ok


def test_ambiguous():
    mgr, rec, _ = simulate(scene_two_lookalikes, 140, True, (250 + 2 * 3, 250))
    amb = sum(1 for _, i in rec.rows if i.ambiguous)
    locked = [e for e in mgr.events if e["type"] == "reacquired"]
    assert amb > 0, "should flag ambiguity"
    assert not locked, f"should not guess between identical candidates: {locked}"
    print(f"PASS ambiguity: {amb} frames flagged AMBIGUOUS, no wrong lock")


def test_quality_gate():
    """Template bank must NOT be updated while another person overlaps the target."""
    cfg = Config()
    tgt = Track(1, box_at(300, 250), 0.9)
    near = Track(2, box_at(310, 250), 0.9)
    m = TargetManager(cfg, (H, W, 3), True)
    img = render([(tgt.box, RED_BLUE), (near.box, GREEN_BLACK)])
    m.select([tgt, near], (300, 250), img, 0)  # note: picks smaller box containing click
    assert not m._good_quality(Track(m.target_id, tgt.box, 0.9), [tgt, near])
    assert m._good_quality(Track(m.target_id, tgt.box, 0.9), [tgt])
    print("PASS quality gate blocks template updates under overlap")


def test_long_block():
    for reasoning in (True, False):
        mgr, rec, truth = simulate(scene_long_block, 260, reasoning, (300, 250))
        states = {i.state.value for _, i in rec.rows}
        assert final_ok(rec, truth), f"should re-find target after long block (reasoning={reasoning})"
    print("PASS long block: after the normal search expired, wide search found the target again")


if __name__ == "__main__":
    test_long_block()
    test_quality_gate()
    test_obstacle()
    test_person()
    test_exit_lookalike()
    test_ambiguous()
    print("\nAll synthetic tests passed.")
