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

## Install and run
```bash
pip install -r requirements.txt
python run.py --source video.mp4            # click a person; q=quit, r=reset, space=pause
python run.py --source 0                    # webcam
python run.py --source video.mp4 --baseline # without the new feature
```
Outputs go to `results/`: `log.csv` (per frame), `summary.json` (metrics + events), optional `--save-video`.

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

## Tests (no GPU/internet needed)
`python tests/test_synthetic.py` - synthetic people with fake ByteTrack IDs. Covers all three occlusion
types, the ambiguity state, the template quality gate, and a case where the baseline locks onto a lookalike
and the new feature does not.

## Files
`config.py` all parameters | `target_manager.py` state machine + reasoning | `appearance.py` features/bank |
`detector.py` YOLO+ByteTrack | `pipeline.py` loop | `metrics.py` | `run.py` | `evaluate.py` | `visualize.py`

## Known limitations (write these in your report)
- Colour histograms fail for identical clothing; a ReID embedding (e.g. OSNet) is the natural upgrade.
- Rules and thresholds are hand-tuned; tune `config.py` on your own clips.
- Only one target; a moving camera would need camera-motion compensation.
- Reasoning can be slower to reacquire than baseline, because it refuses implausible candidates and waits for the real one.
