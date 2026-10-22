# Explain-it-yourself guide (read this before the review)

## 1. The 30-second version
"We built a system where you click one person in a video or live camera and it keeps following that person.
YOLO11n finds people and ByteTrack gives each an ID. Plain ByteTrack loses the person when he is hidden and gets a
new ID. We added a manager on top that works out why he vanished, remembers how he looks, and re-finds him, and that
refuses to guess when it is not sure. We measured it against the plain approach on the same videos."

## 2. Where each piece came from
| Piece | Where from |
|---|---|
| YOLO11n, ByteTrack (and BoT-SORT if we compare with it) | Ultralytics library, installed with `pip install ultralytics`, runs on our own computer |
| "Existing method" = follow the clicked ID | `baselines.py`, about 30 lines written for this project. It is how a tracker is normally used, not a published algorithm |
| Our manager and appearance code | `target_manager.py`, `appearance.py`: written for this project, not copied from another repository |
| Comparison tools | `annotate_gt.py`, `metrics.py`, `evaluate.py`, `compare_live.py`, `make_report.py`: written for this project |
Be ready to say: "The detector and trackers are the library's. The state machine, the appearance logic, and the comparison
harness are our own code. Related ideas exist in published work (BoT-SORT, DeepSORT, StrongSORT) and we cite them."

## 3. How the comparison works (use this analogy)
Think of one exam paper, one answer key, three students.
1. **Same exam paper:** YOLO + ByteTrack runs ONCE on the video and the result is saved (`run.py --save-tracks`).
   Every method replays exactly these detections, so differences come only from our logic, not from luck.
2. **Answer key:** we click the real person every 5th frame (`annotate_gt.py`). That is the ground truth.
3. **Three students** (`evaluate.py`): (a) existing: follow the clicked ID; (b) ours without the new feature;
   (c) ours with occlusion-type reasoning. Optionally also BoT-SORT with ReID as the existing method.
4. **Marking** (`metrics.py`): on each marked frame, is the box we are locked on the right person?
5. **Report** (`make_report.py`) and the live side-by-side (`compare_live.py`).

## 4. What the numbers mean (say these in plain words)
- **Tracking accuracy:** of the frames where the person is really visible, how many did we have a box on him
  (overlap with the true box of at least 50%).
- **Frames locked on the wrong person:** how long we confidently followed someone else. Lower is better.
- **ID switches / wrong reacquisitions:** how often we re-locked on the wrong person.
- **Frames to re-lock:** how long from vanishing until we follow him again.
- **ms per frame:** the speed cost of our manager next to YOLO.
The live "locked %" is NOT accuracy: it counts frames where the system CLAIMS a lock. Do not call it accuracy.

## 5. What counts as proof (and what does not)
Proof = real clips + hand-marked ground truth + the same detections for every method + a table with all methods.
- Use 4-6 clips, at least some hard ones (people crossing, leaving the frame, long block, similar clothes).
- Report every method, including cases where the existing method is as good or better. On an easy clip where ByteTrack keeps
  the same ID, the existing method is already right. Say so; it makes the other results more believable.
- Do NOT call the scripted benchmark (`benchmark_synthetic.py`) real results. It shows what the logic does when the ID changes.
- Do NOT claim "80%" or "high accuracy" before you have measured it. Say the measured number: "X% on our N clips".
- Say how many clips and frames, and how the ground truth was made.

## 6. Read the code in this order (about 30 minutes)
1. `run.py`: opens the video, runs the detector, handles clicks, draws the window.
2. `pipeline.py` (`run_session`): for each frame, send the detections to the manager and record the result.
3. `target_manager.py`:
   - `select`: you clicked a person: remember who and how he looks.
   - `update`: called every frame; moves between ACQUIRED, OCCLUDED, REACQUIRING, LOST.
   - `_observe` and `_good_quality`: learn his appearance only when he is clearly visible.
   - `_begin_occlusion`: he vanished: was it the frame edge, another person, or an obstacle? (the new feature)
   - `_search_region`: where to look, depending on that cause.
   - `_score_candidates`: score each new person on place, looks, size, confidence; veto strangers.
   - `_reacquire`: accept only a clear winner that stays the winner for a few frames; otherwise wait.
4. `appearance.py`: colour histograms of torso and legs, and a small memory of how he looks.
5. `baselines.py`: the existing method, so you can show it is simple.
6. `metrics.py` and `evaluate.py`: how the marking and comparison work.
Tip: run `python tests/test_synthetic.py`, then read one scene in that file; each test is a story you can explain.

## 7. Questions to expect
- **Where did you get the code?** Section 2.
- **Isn't this existing?** Parts are: appearance matching, ReID, occlusion handling. Ours is the combination for a
  user-selected target with occlusion-cause reasoning and refusal to guess, in a lightweight real-time pipeline, measured
  against the plain approach. We cite BoT-SORT, DeepSORT, StrongSORT.
- **Why is your baseline fair?** Same detections, same click, same marking; we also compare with BoT-SORT + ReID.
- **How do you know it is accurate?** Section 5.
- **Where does it fail?** Identical clothing, strong lighting changes, thresholds tuned by hand, one target at a time.
- **Does it slow things down?** `evaluate.py` prints YOLO ms and our manager ms for your machine.

## 8. 2-minute demo script
1. `python compare_live.py --source clip.mp4`; click the target; press SPACE.
2. "Left is YOLO plus ByteTrack alone, right is ours."
3. When someone blocks him: "Left loses him, right shows what it is doing, and re-finds him."
4. Show the results table and chart from your real clips, then the limits and future work.
