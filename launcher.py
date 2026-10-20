"""Start here:  python launcher.py   - pick video file or live camera, tracking or side-by-side comparison."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

MENU = """
==============================================================
  Robust Human Target Detection and Acquisition
  YOLO11n + ByteTrack + occlusion-type reasoning
==============================================================
  TRACKING (ours)
    1) Video file
    2) Live webcam
    3) IP camera / RTSP / phone-camera URL
  COMPARISON  (existing approach LEFT  vs  ours RIGHT)
    4) Video file
    5) Live webcam
    q) Quit
"""


def pick_video():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askopenfilename(title="Choose a video",
                                          filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All", "*.*")])
        root.destroy()
        if path:
            return path
    except Exception:
        pass
    return input("Type the video path (e.g. test.mp4): ").strip().strip('"')


def pick_camera():
    v = input("Camera number [0 = built-in webcam, 1 = external]: ").strip()
    return v if v.isdigit() else "0"


def main():
    while True:
        print(MENU)
        c = input("Choose: ").strip().lower()
        if c == "q":
            return
        if c not in {"1", "2", "3", "4", "5"}:
            continue
        script = "compare_live.py" if c in ("4", "5") else "run.py"
        if c in ("1", "4"):
            src = pick_video()
        elif c in ("2", "5"):
            src = pick_camera()
        else:
            src = input("Camera URL (e.g. rtsp://user:pass@192.168.1.5/stream): ").strip()
        if not src:
            continue
        extra = []
        if c in ("2", "3", "5"):
            slow = input("Slow computer? Use smaller detector size for more FPS (y/N): ").strip().lower()
            if slow == "y":
                extra = ["--imgsz", "480"]
        cmd = [sys.executable, os.path.join(HERE, script), "--source", src, *extra]
        print("\nRunning:", " ".join(cmd), "\n(click a person to select, SPACE to play, q to quit)\n")
        subprocess.call(cmd, cwd=HERE)


if __name__ == "__main__":
    main()
