"""Create botsort_reid.yaml (BoT-SORT with appearance ReID turned on) from YOUR installed Ultralytics version.

  python make_reid_config.py
  python run.py --source clip.mp4 --no-display --tracker botsort_reid.yaml --select-point 320,240 --select-frame 10 --save-tracks clip_botsort_reid.json
  python evaluate.py ... --tracks clip.json --existing-tracks clip_botsort_reid.json --gt clip_gt.csv

Needs a recent Ultralytics (pip install -U ultralytics). See docs.ultralytics.com/modes/track for your version."""
import os
import re
import shutil
import sys


def make(src=None, dst="botsort_reid.yaml"):
    if src is None:
        import ultralytics
        src = os.path.join(os.path.dirname(ultralytics.__file__), "cfg", "trackers", "botsort.yaml")
    if not os.path.exists(src):
        sys.exit(f"Could not find {src}. Copy botsort.yaml from your Ultralytics install manually.")
    shutil.copy(src, dst)
    text = open(dst).read()
    new, n = re.subn(r"with_reid:\s*False", "with_reid: True", text)
    if n == 0 and "with_reid" not in text:
        sys.exit("This Ultralytics version has no with_reid option - run: pip install -U ultralytics")
    open(dst, "w").write(new)
    print(f"created {dst} (with_reid: True) from {src}")


if __name__ == "__main__":
    make()
