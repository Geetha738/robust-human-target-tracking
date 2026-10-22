"""Simple launcher: choose video file or live camera, normal or side-by-side comparison.
Run:  python start.py"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def launch(script, source, faster):
    args = [sys.executable, os.path.join(HERE, script), "--source", str(source)]
    if faster:
        args += ["--imgsz", "480"]
    subprocess.Popen(args, cwd=HERE)


def gui():
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.title("Human Target Tracking")
    root.geometry("440x420")
    tk.Label(root, text="Robust Human Target Tracking", font=("Segoe UI", 15, "bold")).pack(pady=10)
    faster = tk.BooleanVar(value=False)
    cam = tk.StringVar(value="0")

    def pick_video(script):
        path = filedialog.askopenfilename(title="Choose a video",
                                          filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("All", "*.*")])
        if path:
            launch(script, path, faster.get())

    def use_cam(script):
        src = cam.get().strip() or "0"
        launch(script, src, faster.get())

    f = tk.LabelFrame(root, text=" Track ", padx=10, pady=8); f.pack(fill="x", padx=14, pady=6)
    tk.Button(f, text="Video file...", width=34, command=lambda: pick_video("run.py")).pack(pady=3)
    tk.Button(f, text="Live camera (real time)", width=34, command=lambda: use_cam("run.py")).pack(pady=3)
    f2 = tk.LabelFrame(root, text=" Show teachers: Existing vs Ours side by side ", padx=10, pady=8)
    f2.pack(fill="x", padx=14, pady=6)
    tk.Button(f2, text="Video file...", width=34, command=lambda: pick_video("compare_live.py")).pack(pady=3)
    tk.Button(f2, text="Live camera (real time)", width=34, command=lambda: use_cam("compare_live.py")).pack(pady=3)
    o = tk.Frame(root); o.pack(pady=6)
    tk.Label(o, text="Camera number or stream URL:").pack(side="left")
    tk.Entry(o, textvariable=cam, width=18).pack(side="left", padx=6)
    tk.Checkbutton(root, text="Faster mode (smaller detector size, for slow laptops)", variable=faster).pack()
    tk.Label(root, text="In the video window: click a person, SPACE = play/pause, q = quit",
             fg="#555").pack(pady=8)
    root.mainloop()


def terminal():
    print("1) Track a video file\n2) Track live camera\n3) Compare existing vs ours (video)\n4) Compare existing vs ours (camera)")
    c = input("Choose 1-4: ").strip()
    script = "run.py" if c in "12" else "compare_live.py"
    src = input("Video path: ").strip().strip('"') if c in ("1", "3") else (input("Camera number [0]: ").strip() or "0")
    launch(script, src, input("Faster mode? (y/n): ").lower().startswith("y"))


if __name__ == "__main__":
    try:
        gui()
    except Exception:
        terminal()
