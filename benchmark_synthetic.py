"""Existing method vs ours on controlled scenarios (no YOLO, no video, no internet needed).

  python benchmark_synthetic.py

Each scenario is a short synthetic clip (coloured 'people' with ByteTrack-like IDs) where we KNOW where the
target really is, so accuracy and frames-to-relock are exact. Produces synthetic_results.png / .md.
These are controlled tests of the tracking logic; use evaluate.py for results on your real videos."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))

import test_synthetic as T                      # noqa: E402
from baselines import PlainFollower             # noqa: E402
from config import Config                       # noqa: E402
from metrics import MetricsRecorder             # noqa: E402
from pipeline import AutoSelector, run_session  # noqa: E402
from target_manager import TargetManager        # noqa: E402
from utils import Track, iou                    # noqa: E402


def scene_same_id(f):
    hidden = 60 <= f < 85
    return [(1, T.box_at(100 + 4 * f, 260), T.RED_BLUE, not hidden, True)]


# name, scene function, frames, click point, frame where target is visible again (None = no hiding)
SCENARIOS = [
    ("ByteTrack keeps\nsame ID (easy)", scene_same_id, 140, (112, 260), 85),
    ("New ID after hiding\nbehind obstacle", T.scene_obstacle, 160, (112, 260), 85),
    ("Hidden behind\nanother person", T.scene_person, 110, (209, 250), 72),
    ("Leaves frame, look-alike\nappears first", T.scene_exit_lookalike, 200, (324, 250), 140),
    ("Stranger stands where\nhe vanished", lambda f: T.scene_stranger(f, True), 200, (209, 250), 120),
    ("People cross,\nIDs swap", T.scene_cross(True), 100, (115, 250), None),
    ("Hidden for a long\ntime (6 s)", T.scene_long_block, 260, (300, 250), 190),
    ("Blocker takes the ID\n(similar outfit)", T.scene_blocker("steal_new"), 130, (115, 250), 50),
    ("Blocker takes the ID\n(identical outfit)", T.scene_blocker("steal_new", T.RED_BLUE), 130, (115, 250), 50),
    ("Target passes behind\nstanding person", T.scene_static_blocker, 140, (115, 250), 60),
]
VARIANTS = [("existing", "Existing (YOLO+ByteTrack)", "#9aa0a6", lambda c, s: PlainFollower(c, s)),
            ("base", "+ our reacquisition", "#4e8fd6", lambda c, s: TargetManager(c, s, False)),
            ("ours", "+ occlusion reasoning (ours)", "#2e9e4f", lambda c, s: TargetManager(c, s, True))]


def run(scene_fn, n, point, back, make):
    cfg, rec, truth = Config(), MetricsRecorder(), {}

    def frames():
        for f in range(n):
            objs = scene_fn(f)
            vis = [(i, b, c) for i, b, c, v, _ in objs if v]
            for i, b, c, v, is_t in objs:
                if is_t and v:
                    truth[f] = b
            yield f, T.render([(b, c) for _, b, c in vis]), [Track(i, b, 0.9) for i, b, _ in vis]

    run_session(frames(), lambda s: make(cfg, s), rec, AutoSelector(point, 3))
    correct = wrong = 0
    oks = []
    for f, info in rec.rows:
        locked = info.state.value == "ACQUIRED" and info.box is not None and not info.uncertain
        ok = locked and f in truth and iou(info.box, truth[f]) >= 0.5
        correct += ok
        wrong += locked and not ok
        oks.append((f, ok))
    relock = None            # first frame from which the lock stays correct for 10 frames in a row
    if back is not None:
        tail = [(f, ok) for f, ok in oks if f >= back]
        for k in range(len(tail)):
            win = tail[k:k + 10]
            if len(win) >= 5 and all(ok for _, ok in win):
                relock = tail[k][0] - back
                break
    return correct / max(1, len(truth)), wrong, relock


def main():
    res = {n: {} for n, *_ in SCENARIOS}
    for name, fn, n, pt, back in SCENARIOS:
        for key, _, _, make in VARIANTS:
            res[name][key] = run(fn, n, pt, back, make)

    # ---------------- table
    lines = ["| Scenario | Method | Accuracy | Frames locked on WRONG person | Frames to re-lock after he is visible |", "|---|---|---|---|---|"]
    for name, *_ in SCENARIOS:
        for key, label, _, _ in VARIANTS:
            a, w, r = res[name][key]
            back = [s[4] for s in SCENARIOS if s[0] == name][0]
            rr = "n/a" if back is None else ("NEVER" if r is None else str(r))
            lines.append(f"| {name.replace(chr(10), ' ')} | {label} | {a:.0%} | {w} | {rr} |")
    avg = {k: np.mean([res[n][k][0] for n in res]) for k, *_ in VARIANTS}
    lines.append("")
    lines.append("**Average accuracy:** " + ", ".join(f"{l} {avg[k]:.0%}" for k, l, _, _ in VARIANTS))
    open(os.path.join(HERE, "synthetic_results.md"), "w").write("\n".join(lines))
    print("\n".join(lines))

    # ---------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [s[0] for s in SCENARIOS]
    x = np.arange(len(names))
    w = 0.26
    fig, ax = plt.subplots(3, 1, figsize=(17, 13))
    for j, (key, label, col, _) in enumerate(VARIANTS):
        ax[0].bar(x + (j - 1) * w, [res[n][key][0] * 100 for n in names], w, label=label, color=col)
        ax[1].bar(x + (j - 1) * w, [res[n][key][1] for n in names], w, label=label, color=col)
        vals, hatch = [], []
        for n, s in zip(names, SCENARIOS):
            r = res[n][key][2]
            vals.append(0 if s[4] is None else (60 if r is None else r))
            hatch.append(s[4] is not None and r is None)
        bars = ax[2].bar(x + (j - 1) * w, vals, w, label=label, color=col)
        for b, h, s, n in zip(bars, hatch, SCENARIOS, names):
            if s[4] is None:
                ax[2].text(b.get_x() + b.get_width() / 2, 1, "n/a", ha="center", fontsize=7, color="#555")
            elif h:
                b.set_hatch("//")
                ax[2].text(b.get_x() + b.get_width() / 2, 61, "never", ha="center", fontsize=8, color="#b00020")
    titles = ["Tracking accuracy (% of frames locked on the REAL person)  - higher is better",
              "Frames locked on the WRONG person  - lower is better",
              "Frames needed to re-lock after the person is visible again  - lower is better"]
    for a_, t_ in zip(ax, titles):
        a_.set_xticks(x)
        a_.set_xticklabels(names, fontsize=7)
        a_.set_title(t_, fontsize=11, loc="left")
    ax[0].set_ylim(0, 118)
    ax[0].legend(ncol=3, loc="upper center", fontsize=9)
    ax[1].set_ylabel("frames")
    ax[2].set_ylabel("frames")
    ax[2].set_ylim(0, 70)
    fig.suptitle("Existing approach vs ours - controlled scenarios (synthetic, ground truth known)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(HERE, "synthetic_results.png"), dpi=140)
    print("\nwrote synthetic_results.png and synthetic_results.md")


if __name__ == "__main__":
    main()
