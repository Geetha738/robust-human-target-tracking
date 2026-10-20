# Changes in this version (files to replace in your project)

Problem fixed: when one person blocks another, the tracker could move the target's ID onto the blocker and the system
kept following the blocker. The old identity check asked "does this box look like the target?" with a fixed threshold, but a
person with a similar shirt scores about 0.5, which the old check accepted. The new crossing check compares the people
involved with each other.

| File | What changed |
|---|---|
| `config.py` | New `cross_*` settings (overlap, evidence frames, weights, margins), `jump_factor`; `tracker`; `same_id_fast_app/confirm` |
| `target_manager.py` | New `_cross_step` and `_snapshot`; the ACQUIRED branch of `update` calls it before `_observe`; no appearance learning while a crossing is unresolved; `TargetInfo.uncertain`; crossing state reset in `_finish_reacq` |
| `metrics.py` | "VERIFYING" frames count as not locked; new `verifying_frames` |
| `visualize.py` | Orange "VERIFYING IDENTITY" box |
| `compare_live.py` | "locked %" excludes verifying frames; `--tracker` option |
| `detector.py`, `run.py`, `evaluate.py`, `make_report.py` | `--tracker`, `--existing-tracks`, extra report row |
| `tests/test_synthetic.py` | New scenes and `test_blocker_takes_id` |
| `benchmark_synthetic.py` | 10 scenarios (3 new blocker scenes), stricter "re-lock" definition |
| `make_sample_videos.py` (new) | Makes two example videos with tracker output and ground truth |

After copying the files run `python tests/test_synthetic.py` (it should end with "All synthetic tests passed.").
