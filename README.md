# Robust Human Target Detection and Acquisition (YOLO11n + ByteTrack)

Click a person in a video; the system keeps following that person through occlusions and crowds.

## Pipeline
YOLO11n (people only) -> ByteTrack IDs -> **TargetManager** (state machine) -> overlay + metrics

States: `IDLE -> ACQUIRED -> OCCLUDED -> REACQUIRING -> ACQUIRED | LOST`

## Base system
- Click-to-select (smallest box under the click)
- Target remembered with upper/lower-body HSV histograms (centre of box only, less background)
- Reacquisition candidates scored on: predicted position, appearance, size, detector confidence
- Weights shift from "position" to "appearance" the longer the person stays hidden
- Quality-gated template bank: appearance memory is updated only when the target is clearly visible
  (high confidence, not overlapped by anyone, not at the frame edge)
- Accept a candidate only if score >= threshold, it beats the runner-up by a margin, and it wins
  3 frames in a row. Otherwise the HUD shows **AMBIGUOUS - waiting** instead of guessing.
- Candidates must look at least roughly like the target (stops locking onto the person who occluded it)

## NEW FEATURE: occlusion-type reasoning
When the target disappears, the system decides **why**, then searches accordingly:

| Type | Detected when | Search strategy | Timeout |
|---|---|---|---|
| EXITED | last box at frame edge, moving outward (or clipped) | only accept new people appearing in the edge band | long (300 f) |
| PERSON | another person covered >= 25% of the target box | only search around that person (follows them while visible) | short (90 f) |
| OBSTACLE | vanished in the open | search along motion direction, radius grows with time | 150 f |

`--baseline` switches this off (one generic search everywhere) so you can measure its benefit.

## Easiest way: the launcher menu
```bash
python launcher.py
```
Choose: video file / live webcam / IP camera, for tracking, or the **side-by-side comparison** (existing approach left,
ours right, same video or camera, live). Best thing to show your teachers.

## Install and run
```bash
pip install -r requirements.txt
python run.py --source video.mp4            # click a person; q=quit, r=reset, space=pause
python run.py --source 0                    # webcam
python run.py --source video.mp4 --baseline # without the new feature
```
Outputs go to `results/`: `log.csv` (per frame), `summary.json` (metrics + events), optional `--save-video`.

## Easiest way to start
```bash
python start.py      # window with buttons: video file / live camera / side-by-side comparison
```

## Identity verification (people crossing)
ByteTrack sometimes swaps IDs when two people cross, and a plain ID-follower then follows the wrong person.
Every frame we check that the tracked box still looks like the target (skipped while someone overlaps it or it
is cut by the frame edge). If a neighbour matches the target clearly better for 3 frames, the target ID is
corrected (`id_corrected` event). The appearance memory is never updated from a box that does not look like the target.

## Show it to your teachers
1. **Live side-by-side demo** - same detections, same click, existing method on the left, ours on the right,
   with a running "tracked %" for each:
   `python compare_live.py --source video.mp4`   or   `python compare_live.py --source 0` (live camera)
   Add `--save-video demo.mp4` to record it.
2. **Numbers + report**: after the ablation steps below, run
   `python make_report.py --inputs clipA.json clipB.json --names "Behind person" "Crossing" --out report.html`
   and open `report.html` in a browser (print to PDF). `sample/sample_report.html` shows the layout
   (built from synthetic clips; replace with your own).

## Avoiding wrong switches (conservative by design)
A wrong person is worse than "not found yet", so the system only locks on a candidate when ALL of these hold:
- it looks like the target (appearance >= `accept_app`), not merely stands in the right place
- it does NOT look more like a known bystander than like the target (bystander memory)
- it clearly beats the runner-up (else AMBIGUOUS - waiting)
- it wins `confirm_frames` (8) frames in a row (15 in the whole-frame search)
The automatic ID-swap correction needs 12 consistent frames and strong evidence. Turn it off completely with
`python run.py --source clip.mp4 --no-identity-check`.
Too strict (stays OCCLUDED too long)? Lower `accept_app`, `min_appearance` or `confirm_frames` in `config.py`.
Still switching on your video? Raise them, or upgrade the appearance cue to a ReID embedding (e.g. OSNet).

## Blocker takes the target's ID (crossing check)
When someone overlaps the target (or the tracked box suddenly jumps) the tracker may hand the target's ID to the other
person. After they separate, `_cross_step` compares everybody involved against each other (appearance, motion continuity,
size). The orange "VERIFYING IDENTITY" box means the system is not sure yet and does NOT count as locked. It then either
confirms, switches to the right person, or searches again. `python make_sample_videos.py` creates two example videos of
this problem (moving blocker, standing blocker) with tracker output and ground truth in `sample_videos/`.

## Real-time / webcam / live streams
```bash
python run.py --source 0                      # webcam
python run.py --source rtsp://user:pass@ip/stream   # IP camera
python run.py --source 0 --imgsz 480          # faster on a slow laptop (try 320 if still slow)
```
The FPS is shown in the top-right corner. The video pauses on the first frame with people so you can click one;
press SPACE to start. Press `s` any time to pause and re-select.

## What happens when the person is hidden for a long time
1. **Stage 1 - type-specific search** (EXITED / PERSON / OBSTACLE): gated and quick (see table above).
2. **Stage 2 - wide search**: if stage 1 times out, the HUD shows "searching whole frame". It looks at everybody,
   trusts appearance most, and needs a stronger match (`strict_*` values in `config.py`).
3. **LOST** only after `wide_timeout` frames (default 900 = 30 s at 30 FPS) - and even then it keeps watching
   ("LOST - still watching") and re-locks if the person clearly comes back.

## Proving the accuracy improved (compare with the existing approach)
Three systems are run on the SAME detections, so the comparison is fair:
- **existing**: plain YOLO11n + ByteTrack following the clicked ID (what everybody already has)
- **base**: + our reacquisition (appearance memory, prediction, ambiguity handling, wide search)
- **ours**: base + occlusion-type reasoning (the new feature)

```bash
# 1. run YOLO+ByteTrack once; this also measures detector speed
python run.py --source clip.mp4 --no-display --select-point 320,240 --select-frame 10 --save-tracks clip.json
# 2. make ground truth: click the target every 5th frame (n = target not visible)
python annotate_gt.py --source clip.mp4 --tracks clip.json --step 5 --out clip_gt.csv
# 3. compare
python evaluate.py --source clip.mp4 --tracks clip.json --select-point 320,240 --select-frame 10 --gt clip_gt.csv
```
You get one table: tracking accuracy, correct/wrong reacquisitions, frames locked on the wrong person,
ID switches, time to reacquire, and **timing** (YOLO ms/frame, our manager ms/frame, estimated pipeline FPS).
Use `--select-point` with a pixel inside the target on the `--select-frame` (read it off the paused window).
Record 4-6 clips: exit and re-entry, hidden behind a person, behind an obstacle, crowd with similar clothes,
long block. Report the table per clip and the average. Analysis to write: our manager costs well under 1 ms/frame
next to YOLO's tens of ms, so accuracy rises at essentially no speed cost.

## Comparing with stronger existing trackers (BoT-SORT, ReID)
`python run.py --source clip.mp4 --tracker botsort.yaml ...` uses any Ultralytics tracker. `python make_reid_config.py`
creates `botsort_reid.yaml` (appearance ReID on). `evaluate.py --existing-tracks <that cache>` then compares it with ours.
Details in `IMPROVEMENTS.md`.

## Report for teachers (table + chart)
```bash
python make_report.py --inputs clipA.json clipB.json clipC.json --names "Behind person" "Exit" "Crowd"
```
(the .json files are the `comparison.json` outputs of `evaluate.py`; use `--out` there to give each clip its own name).
Creates `report.md` (what is new, per-clip and average tables, speed cost) and `report.png` (accuracy chart).

## Quick numbers + chart without any video
`python benchmark_synthetic.py` runs 7 controlled scenarios (ID changes, stranger, crossing, long block ...) for the
existing method vs ours and writes `synthetic_results.png` / `synthetic_results.md`: accuracy, frames locked on the wrong
person, and frames to re-lock. These are controlled logic tests; use `evaluate.py` for results on your real videos.

## Tests (no GPU/internet needed)
`python tests/test_synthetic.py` - synthetic people with fake ByteTrack IDs. Covers all three occlusion
types, the ambiguity state, the template quality gate, and a case where the baseline locks onto a lookalike
and the new feature does not.

## Files
`start.py` launcher | `compare_live.py` side-by-side demo | `make_report.py` report | `annotate_gt.py` ground truth |
`config.py` all parameters | `target_manager.py` state machine + reasoning | `appearance.py` features/bank |
`detector.py` YOLO+ByteTrack | `pipeline.py` loop | `metrics.py` | `run.py` | `evaluate.py` | `visualize.py`

## Known limitations (write these in your report)
- Colour histograms fail for identical clothing; a ReID embedding (e.g. OSNet) is the natural upgrade.
- Rules and thresholds are hand-tuned; tune `config.py` on your own clips.
- Only one target; a moving camera would need camera-motion compensation.
- Reasoning can be slower to reacquire than baseline, because it refuses implausible candidates and waits for the real one.
