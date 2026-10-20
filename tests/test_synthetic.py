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


def simulate(scene_fn, n_frames, reasoning, select_point, select_frame=3, cfg=None):
    """scene_fn(f) -> list of (track_id, box, clothes, visible, is_target)"""
    cfg = cfg or Config()
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
        if info.state.value == "ACQUIRED" and not info.uncertain and f in truth and iou(info.box, truth[f]) > 0.5:
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


def scene_cross(swap):
    """A (red/blue, the target) walks right, B (green/black) walks left; they cross at f=50.
    With swap=True the tracker hands A's ID to B and B's ID to A, as ByteTrack sometimes does."""
    def fn(f):
        a, b = box_at(100 + 5 * f, 250), box_at(600 - 5 * f, 250)
        ida, idb = (1, 2) if not (swap and f >= 50) else (2, 1)
        return [(ida, a, RED_BLUE, True, True), (idb, b, GREEN_BLACK, True, False)]
    return fn



HALF_SIMILAR = ((30, 30, 200), (30, 30, 30))   # same red top, black trousers: scores ~0.5 against the target


def scene_blocker(mode, blocker=HALF_SIMILAR):
    """Target A (red/blue) walks right, blocker B walks left in front of him; they cross at f=50.
    mode='keep'      : tracker IDs stay correct
    mode='swap'      : A and B exchange IDs at the crossing
    mode='steal_new' : B keeps A's old ID 1, A is given a NEW id (9) - the blocker 'steals' the target"""
    def fn(f):
        a, b = box_at(100 + 5 * f, 250), box_at(600 - 5 * f, 250)
        ida, idb = 1, 2
        if f >= 50 and mode == "swap":
            ida, idb = 2, 1
        if f >= 50 and mode == "steal_new":
            ida, idb = 9, 1
        return [(ida, a, RED_BLUE, True, True), (idb, b, blocker, True, False)]
    return fn


def scene_static_blocker(f):
    """B stands still. A walks right and passes BEHIND him (hidden f=48..59). While A is hidden the
    tracker moves ID 1 onto B; when A comes out he gets a new ID (9)."""
    a = box_at(100 + 5 * f, 250)
    b = box_at(300, 250)
    hidden = 48 <= f < 60
    ppl = []
    if f < 48:
        ppl.append((1, a, RED_BLUE, True, True))
        ppl.append((2, b, HALF_SIMILAR, True, False))
    else:
        ppl.append((1, b, HALF_SIMILAR, True, False))                   # ID 1 now sits on the blocker
        if not hidden:
            ppl.append((9, a, RED_BLUE, True, True))
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


def test_cross():
    from baselines import PlainFollower
    for swap in (True, False):
        mgr, rec, truth = simulate(scene_cross(swap), 100, True, (115, 250))
        assert final_ok(rec, truth), f"should still follow the same person after crossing (swap={swap})"
        fixed = [e for e in mgr.events if e["type"] == "id_corrected"]
        assert bool(fixed) == swap, f"correction events: {fixed}"
    # existing method follows the swapped ID onto the wrong person
    cfg = Config(); rec = MetricsRecorder(); truth = {}
    def frames():
        for f in range(100):
            objs = scene_cross(True)(f)
            for i, b, c, v, is_t in objs:
                if is_t: truth[f] = b
            yield f, render([(b, c) for _, b, c, _, _ in objs]), [Track(i, b, 0.9) for i, b, _, _, _ in objs]
    run_session(frames(), lambda s: PlainFollower(cfg, s), rec, AutoSelector((115, 250), 3))
    assert not final_ok(rec, truth), "plain ID-following should end up on the wrong person"
    print("PASS crossing: ID swap detected and corrected; existing method follows the wrong person")


STRANGER = ((30, 30, 200), (30, 30, 30))      # same red top, but black trousers: similarity ~0.5


def scene_stranger(f, target_returns):
    """Target vanishes behind a pillar at f=60. A stranger (half-similar clothes, new id) stands
    exactly where he disappeared. Optionally the real target returns later somewhere else."""
    ppl = []
    if f < 60:
        ppl.append((1, box_at(200 + 3 * f, 250), RED_BLUE, True, True))
    if f >= 70:
        ppl.append((30, box_at(200 + 3 * 60 + 40, 250), STRANGER, True, False))
    if target_returns and f >= 120:
        ppl.append((31, box_at(500, 330), RED_BLUE, True, True))
    return ppl


def test_stranger_not_adopted():
    sp = (200 + 3 * 3, 250)
    # 1) target never returns: system must stay OCCLUDED/REACQUIRING, never adopt the stranger
    for reasoning in (True, False):
        mgr, rec, _ = simulate(lambda f: scene_stranger(f, False), 200, reasoning, sp)
        assert not [e for e in mgr.events if e["type"] == "reacquired"], "adopted a stranger!"
        assert rec.rows[-1][1].state.value != "ACQUIRED"
    # 2) target returns elsewhere: must lock the REAL target, not the stranger
    mgr, rec, truth = simulate(lambda f: scene_stranger(f, True), 200, True, sp)
    assert final_ok(rec, truth), "should lock the real target"
    # 3) proof that the OLD loose settings did adopt the stranger
    loose = Config(min_appearance=0.30, accept_app=0.0, accept_score=0.55, confirm_frames=3,
                   strict_min_app=0.55, strict_accept=0.62, strict_confirm=5, neg_margin=9.0)
    old, _, _ = simulate(lambda f: scene_stranger(f, False), 200, True, sp, cfg=loose)
    adopted = [e for e in old.events if e["type"] == "reacquired"]
    assert adopted, "sanity check: the old loose settings should have adopted the stranger"
    print("PASS stranger not adopted (the old loose settings DID adopt him; new settings do not)")


def scene_flicker(f):
    """Target hidden f=40..70; when he returns the detector drops him every 4th frame."""
    ppl = []
    if f < 40:
        ppl.append((1, box_at(300, 250), RED_BLUE, True, True))
    if f >= 70:
        ppl.append((15, box_at(330, 250), RED_BLUE, f % 4 != 0, True))
    return ppl


def test_flicker_tolerance():
    slow = dict(fast_app=2.0)                       # disable fast path: needs 8 confirmed frames
    ok, _, _ = simulate(scene_flicker, 140, True, (300, 250), cfg=Config(**slow))
    bad, _, _ = simulate(scene_flicker, 140, True, (300, 250), cfg=Config(confirm_tolerance=0, **slow))
    assert [e for e in ok.events if e["type"] == "reacquired"], "should survive flicker"
    assert not [e for e in bad.events if e["type"] == "reacquired"], "sanity: without tolerance it never confirms"
    print("PASS flicker tolerance: a flickering detection no longer resets confirmation")


def test_blocker_takes_id():
    """The reported bug: a blocker (partly similar outfit) takes over the target's ID."""
    cases = [("swap", HALF_SIMILAR), ("steal_new", HALF_SIMILAR), ("swap", RED_BLUE), ("steal_new", RED_BLUE)]
    for mode, outfit in cases:
        mgr, rec, truth = simulate(scene_blocker(mode, outfit), 130, True, (115, 250))
        assert final_ok(rec, truth), f"should end on the real target ({mode}, outfit={outfit})"
        assert [e for e in mgr.events if e["type"] == "id_corrected"], f"should correct the ID ({mode})"
    mgr, rec, truth = simulate(scene_static_blocker, 140, True, (115, 250))
    assert final_ok(rec, truth), "static blocker: should end on the real target"
    # harmless crossings must NOT trigger corrections
    for outfit in (HALF_SIMILAR, GREEN_BLACK, RED_BLUE):
        mgr, rec, truth = simulate(scene_blocker("keep", outfit), 130, True, (115, 250))
        assert final_ok(rec, truth) and not [e for e in mgr.events if e["type"] == "id_corrected"], "false correction"
        assert sum(1 for _, i in rec.rows if i.uncertain) <= 8, "too many 'verifying' frames on a harmless crossing"
    print("PASS blocker takes the ID: corrected in 5 scenes, no false corrections on harmless crossings")


if __name__ == "__main__":
    test_blocker_takes_id()
    test_flicker_tolerance()
    test_stranger_not_adopted()
    test_cross()
    test_long_block()
    test_quality_gate()
    test_obstacle()
    test_person()
    test_exit_lookalike()
    test_ambiguous()
    print("\nAll synthetic tests passed.")
