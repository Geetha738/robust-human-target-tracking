"""Turn evaluate.py results into a table + chart for your report/presentation.

  python make_report.py --inputs clipA.json clipB.json clipC.json --names "Behind person" "Exit" "Crowd"
Creates report.md and report.png (needs matplotlib)."""
import argparse
import json

NAMES = {"existing": "Existing (YOLO+ByteTrack)", "base": "+ Our reacquisition", "ours": "+ Occlusion reasoning (ours)"}
ROWS = [("tracking_accuracy", "Tracking accuracy (higher = better)"),
        ("false_reacquisitions", "Wrong reacquisitions (lower = better)"),
        ("wrong_lock_frames", "Frames locked on wrong person (lower)"),
        ("correct_reacquisitions", "Correct reacquisitions (higher)"),
        ("avg_reacq_frames", "Avg frames from vanishing to re-lock (lower)"),
        ("id_switches", "ID switches"),
        ("lost", "Times declared LOST")]

NEW_TEXT = """## What is new in this project
Existing trackers (ByteTrack and similar) follow an ID. When the person is hidden and the ID changes,
the target is gone. Our contribution, built on top of YOLO11n + ByteTrack:

1. **Occlusion-type reasoning (new feature).** When the target disappears we decide *why*
   (left the frame / hidden behind a person / hidden behind an obstacle) and search accordingly:
   edge zone, area around the blocker, or along the motion direction.
2. **Ambiguity-aware reacquisition.** If two candidates look equally likely, the system waits
   (AMBIGUOUS) instead of guessing, which prevents locking onto the wrong person.
3. **Quality-gated appearance memory.** The target's appearance is learned only when the target is
   clearly visible, so a person standing in front cannot corrupt the memory.
4. **Long-occlusion recovery.** After the targeted search fails, a stricter whole-frame search continues;
   LOST is not final.

Related ideas exist individually in the literature (occlusion handling, re-identification, template-update
gating). Our contribution is the combination for a user-selected single target in a lightweight,
real-time YOLO11n + ByteTrack pipeline, measured against the plain approach on the same data.
"""


def avg(vals):
    v = [x for x in vals if isinstance(x, (int, float))]
    return round(sum(v) / len(v), 3) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--names", nargs="*", default=None)
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    names = a.names or [f"clip {i + 1}" for i in range(len(a.inputs))]
    data = [json.load(open(p)) for p in a.inputs]

    md = ["# Results: existing approach vs ours\n", NEW_TEXT, "## Per-clip results\n"]
    for n, d in zip(names, data):
        md.append(f"### {n}\n")
        md.append("| Metric | " + " | ".join(NAMES[v] for v in NAMES) + " |")
        md.append("|---|" + "---|" * len(NAMES))
        for k, label in ROWS:
            md.append(f"| {label} | " + " | ".join(str(d[v].get(k, "-")) for v in NAMES) + " |")
        md.append("")
    md.append("## Average over all clips\n")
    md.append("| Metric | " + " | ".join(NAMES[v] for v in NAMES) + " |")
    md.append("|---|" + "---|" * len(NAMES))
    for k, label in ROWS:
        md.append(f"| {label} | " + " | ".join(str(avg([d[v].get(k) for d in data])) for v in NAMES) + " |")
    det = avg([d["ours"].get("detector_ms_avg") for d in data])
    mg = avg([d["ours"].get("manager_ms_avg") for d in data])
    if det and mg is not None:
        md.append(f"\n## Speed\nYOLO11n + ByteTrack: **{det} ms/frame**. Our manager adds **{mg} ms/frame** "
                  f"({round(100 * mg / det, 2)}% of the detector time), so accuracy improves at almost no speed cost.\n")
    open(a.out + ".md", "w").write("\n".join(md))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [3, 1.4]})
        x = np.arange(len(names))
        wd = 0.26
        colors = ["#9aa0a6", "#4e8fd6", "#2e9e4f"]
        for j, v in enumerate(NAMES):
            vals = [d[v].get("tracking_accuracy") or 0 for d in data]
            ax[0].bar(x + (j - 1) * wd, vals, wd, label=NAMES[v], color=colors[j])
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(names)
        ax[0].set_ylim(0, 1.3)
        ax[0].set_ylabel("Tracking accuracy")
        ax[0].set_title("Accuracy: existing vs ours")
        ax[0].legend(loc="upper center", fontsize=8, ncol=3)
        if det and mg is not None:
            ax[1].bar(["YOLO+ByteTrack", "Our manager"], [det, mg], color=["#9aa0a6", "#2e9e4f"])
            ax[1].set_ylabel("ms per frame")
            ax[1].set_title("Speed cost")
        plt.tight_layout()
        plt.savefig(a.out + ".png", dpi=150)
        print(f"wrote {a.out}.md and {a.out}.png")
    except ImportError:
        print(f"wrote {a.out}.md (install matplotlib for the chart: pip install matplotlib)")


if __name__ == "__main__":
    main()
