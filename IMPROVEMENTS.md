# What we improved, where it is in the code, and how we know it helped

**Starting point (existing approach):** YOLO11n detects people, ByteTrack gives each an ID, and we follow the clicked ID.
When the person is hidden and ByteTrack gives him a new ID, the target is gone for good.
That "existing" behaviour is implemented in `baselines.py` (`PlainFollower`) so we can measure against it fairly.

## The improvements

| # | Improvement | Problem it solves | Where in the code |
|---|---|---|---|
| 1 | **Occlusion-type reasoning (the new feature)**: decide WHY the target vanished (left frame / behind a person / behind an obstacle) and search accordingly | A search that ignores the cause looks everywhere and picks wrong people | `target_manager.py`: `_begin_occlusion` (classifier), `_search_region` + `_score_candidates` (per-type gates), `Config.timeouts` |
| 2 | **Re-linking after the ID changes**: score candidates on predicted position, appearance, size, confidence | ByteTrack gives a new ID after occlusion, so the old method loses him | `_score_candidates`, `_weights` (weights shift from position to appearance as time passes) |
| 3 | **Appearance memory**: upper/lower-body colour histograms from the centre of the box, small template bank | Recognise the same person by clothes | `appearance.py`: `extract_feature`, `similarity`, `AppearanceBank` |
| 4 | **Quality-gated learning**: update memory only when he is clearly visible | A person standing in front would corrupt the memory | `_good_quality`, `_observe` |
| 5 | **Ambiguity handling**: if two candidates are equally good, wait instead of guessing | Wrong-person lock-ons | `_reacquire` (margin test, AMBIGUOUS state) |
| 6 | **Bystander memory + appearance minimum**: reject anyone who looks more like a known bystander than the target | Strangers standing at the right place were adopted | `_remember_bystanders`, `_score_candidates` (`accept_app`, `neg_margin`) |
| 7 | **Long-occlusion recovery**: after the targeted search fails, search the whole frame (stricter); LOST is not final | He came back after a long block and was not recognised | `_reacquire` (wide stage), `Config.wide_timeout` |
| 8 | **ID-swap correction**: detect when ByteTrack moved our ID to another person while they crossed | Following the wrong person without noticing | `_verify`, `_switch_to` |
| 10 | **Crossing check (identity handover)**: when people overlap, compare everyone involved AGAINST EACH OTHER on appearance, motion continuity and size; show VERIFYING until sure | A blocker took over the target's ID and the old check ("does this box look like him?") said yes | `target_manager.py`: `_cross_step`, `_snapshot`; `Config.cross_*` |
| 9 | **Fast lock + flicker tolerance**: strong evidence needs only 3 frames; one weak frame does not reset progress | Re-locking was too slow | `_reacquire`, `_candidate_missed`, `Config.fast_*` |

Improvements 5, 6 and 8 make it safe (fewer wrong people). Improvements 1, 2, 7 and 9 make it find him more often and sooner.

## Why the existing method can look equal (or slightly better) on an easy video
The live "locked %" counts how many frames the system CLAIMS to be locked, not whether it is locked on the right person.
If ByteTrack happens to give the same ID back after an occlusion and nobody else interferes, the plain method is already
correct, and ours only adds a few frames of safety checking. Our advantage shows up when (a) the ID changes after an
occlusion, (b) a stranger stands where the target vanished, (c) IDs swap when people cross, or (d) he is hidden for a long
time. To compare fairly, use ground-truth accuracy (`annotate_gt.py` + `evaluate.py`), not the live percentage, and include
at least one hard clip. When ByteTrack hands back the ORIGINAL id and he looks right, ours now re-locks within 2 frames.

## How we make sure the accuracy really improved
1. Same input for every method: YOLO+ByteTrack runs once and the detections are saved (`run.py --save-tracks`).
2. Ground truth: we click the real person every 5th frame (`annotate_gt.py`).
3. Three systems replay the identical detections (`evaluate.py`): existing, ours without the new feature, ours with it.
4. Metrics (`metrics.py`): tracking accuracy (frames where the locked box overlaps the true person, IoU >= 0.5),
   wrong reacquisitions, frames locked on a wrong person, ID switches, time to reacquire, ms per frame.
5. `make_report.py` turns this into a table and chart; `compare_live.py` shows it live side by side.
6. Automated tests (`tests/test_synthetic.py`) cover each failure case, including "old loose settings fail, new ones pass".

## Where does each part come from? (answer this honestly)

| Part | Source | Runs where |
|---|---|---|
| YOLO11n (person detector) | Ultralytics open-source library (`pip install ultralytics`) | Locally on your PC after the weights download |
| ByteTrack (IDs) | The implementation inside the same Ultralytics library (`tracker=bytetrack.yaml`) | Locally on your PC |
| "Existing" baseline = follow the clicked ID (`baselines.py`, `PlainFollower`) | Written for this project, about 30 lines: it represents how a tracker is normally used (select an ID, follow it). It is NOT a published algorithm | Locally |
| Our manager (`target_manager.py`, `appearance.py`, ...) | Written for this project. Not copied from another repository. Built on general ideas from the literature (appearance matching, template-update gating, occlusion handling) | Locally |

So in the live demo the left panel really is YOLO11n + ByteTrack from Ultralytics, and the only thing we added to make it a
"selected person" system is the 30-line ID-following rule. When you run `run.py` or `compare_live.py`, everything runs on your
own computer, no internet needed after the first model download.

## Stronger existing trackers DO exist - compare with them too
Ultralytics itself offers more than ByteTrack: BoT-SORT (default tracker; camera-motion compensation and optional appearance
ReID with `with_reid: True`) and, in newer versions, Deep OC-SORT, OC-SORT and others (see docs.ultralytics.com/modes/track).
Outside Ultralytics: DeepSORT and StrongSORT (appearance features), and single-object trackers. A good teacher may ask
"why not just turn ReID on?" - so measure it:
```bash
python make_reid_config.py                     # makes botsort_reid.yaml from YOUR installed version
python run.py --source clip.mp4 --no-display --tracker botsort_reid.yaml --select-point X,Y --select-frame N --save-tracks clip_botsort_reid.json
python evaluate.py --source clip.mp4 --tracks clip.json --existing-tracks clip_botsort_reid.json --select-point X,Y --select-frame N --gt clip_gt.csv
```
The "existing" column then shows BoT-SORT + ReID following the clicked ID, against ours. Report whatever you find; if
BoT-SORT + ReID does well on a clip, say so and show where ours still adds something (strangers, long occlusions, exits,
refusing to guess), or where it does not.

## Honest limits of our evidence
- `benchmark_synthetic.py` uses SCRIPTED IDs (it assumes the tracker hands out a new ID after a gap). That shows what our logic
  does WHEN the ID changes; real ByteTrack sometimes keeps the same ID (as in your own video). Use real clips for real claims.
- The baseline is the simple one. Say which baselines you compared with (ByteTrack-follow, BoT-SORT, BoT-SORT + ReID).
- Related methods exist (BoT-SORT, DeepSORT, StrongSORT, Deep OC-SORT). Cite them and state the difference in your setting:
  a user-selected single target, occlusion-CAUSE reasoning, refusing ambiguous matches, lightweight and real time.
- Suggested citations (check details yourself): ByteTrack - Zhang et al., ECCV 2022; BoT-SORT - Aharon et al., 2022;
  DeepSORT - Wojke et al., ICIP 2017; StrongSORT - Du et al., IEEE TMM 2023.

## Questions your teachers may ask
- **"Isn't this existing?"** Parts exist separately (occlusion handling, re-identification, template-update gating).
  Our contribution is the combination for a user-selected single target, with occlusion-cause reasoning, in a lightweight
  real-time YOLO11n + ByteTrack pipeline, and a measured comparison against the plain approach on the same data.
- **"How do you know accuracy improved?"** Steps 1-5 above: same detections, hand-marked ground truth, numbers per clip.
- **"Does it slow the system down?"** The manager costs about a tenth of a millisecond per frame; YOLO takes tens of ms
  (`evaluate.py` prints both for your machine).
- **"What are the limits?"** Colour histograms are weak for identical clothing (a ReID model is the next step),
  thresholds are hand-tuned, one target at a time, fixed camera assumed. Being strict means it can take a moment
  longer to re-lock, in exchange for far fewer wrong switches.
