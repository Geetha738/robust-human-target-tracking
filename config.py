"""All tunable parameters in one place."""
from dataclasses import dataclass, field


@dataclass
class Config:
    # ---------- detection / tracking ----------
    model: str = "yolo11n.pt"
    tracker: str = "bytetrack.yaml"   # any Ultralytics tracker yaml: botsort.yaml, deepocsort.yaml, my_config.yaml ...
    det_conf: float = 0.30
    imgsz: int = 640

    # ---------- state machine ----------
    occluded_wait: int = 3          # frames to wait for ByteTrack to bring back the SAME id
    # max frames to keep searching before LOST, per occlusion type
    timeouts: dict = field(default_factory=lambda: {
        "EXITED": 300,      # people who leave the frame may take long to return
        "PERSON": 90,       # hidden behind a person: should reappear soon
        "OBSTACLE": 150,
        "UNKNOWN": 150,     # baseline (no occlusion reasoning)
    })

    # ---------- long-occlusion recovery (wide search) ----------
    wide_timeout: int = 900         # after the type-specific search fails, search EVERYWHERE this long
    lost_recover: bool = True       # even in LOST, keep watching for the person to come back
    strict_accept: float = 0.72     # wide search is riskier, so it needs a stronger match
    strict_min_app: float = 0.70
    strict_confirm: int = 15

    # ---------- identity verification while tracking (fixes ID swaps when people cross) ----------
    verify_identity: bool = True
    verify_overlap: float = 0.10    # skip checks while someone overlaps the target (features unreliable)
    verify_low: float = 0.40        # tracked box looks less like the target than this = suspicious
    verify_high: float = 0.75       # a neighbour must look at least this much like the target to take over
    swap_margin: float = 0.30       # ...and beat the tracked box by this much
    swap_radius: float = 2.5        # only consider neighbours within this many target-heights
    verify_frames: int = 12         # consecutive suspicious frames before correcting
    lost_identity_low: float = 0.15 # tracked box looks NOTHING like the target
    lost_identity_frames: int = 20
    # ---------- crossing check: "who is the target after people overlap?" ----------
    cross_overlap: float = 0.20     # another person covers this much of the target box = a crossing starts
    jump_factor: float = 1.2        # box jumps further than this many heights in one frame = suspicious
    cross_sep_frames: int = 2       # wait this many frames after they separate before judging
    cross_min_evidence: int = 3     # frames of evidence needed before a decision
    cross_max_frames: int = 45      # give up checking after this long
    cross_dt_cap: int = 45
    cross_window: int = 6           # judge on the most recent frames, not the early overlap
    cross_w: tuple = (0.55, 0.25, 0.20)   # weights: appearance, motion continuity, size
    cross_margin: float = 0.08      # another person must beat the current one by this much to take over
    cross_min_app: float = 0.50     # ...and must look at least this much like the target
    cross_trust: float = 0.62       # current box scores at least this = identity confirmed...
    cross_trust_app: float = 0.65   # ...AND looks at least this much like the target (motion alone never confirms)
    cross_distrust_app: float = 0.65
    cross_distrust_motion: float = 0.25
    update_min_app: float = 0.45    # never learn appearance from a box that doesn't look like the target

    # ---------- occlusion-type reasoning ----------
    edge_margin: float = 0.06       # fraction of frame size counted as "at the edge"
    exit_speed: float = 0.5         # px/frame outward speed to count as leaving
    person_overlap: float = 0.25    # fraction of target box covered by another person
    reentry_zone: float = 0.20      # EXITED: candidates must appear within this edge band

    # ---------- reacquisition scoring ----------
    accept_score: float = 0.65
    accept_app: float = 0.60        # the winner must LOOK like the target, whatever its position
    margin: float = 0.08            # best must beat 2nd best by this much (else AMBIGUOUS)
    confirm_frames: int = 8         # candidate must win this many frames in a row
    min_cand_conf: float = 0.35
    min_appearance: float = 0.50    # candidate must at least roughly look like the target
    pred_cap_frames: int = 20       # stop extrapolating velocity after this many frames
    adapt_frames: int = 90          # position->appearance weight shift over this time
    same_id_bonus: float = 0.10
    # fast lock: strong evidence needs fewer confirmation frames
    fast_confirm_frames: int = 3    # (normal search) when appearance AND margin are both strong
    fast_strict_confirm: int = 6    # (whole-frame search)
    same_id_fast_app: float = 0.50  # ByteTrack handed back the ORIGINAL id and he looks right: trust it quickly
    same_id_confirm: int = 2
    fast_app: float = 0.85
    fast_margin: float = 0.12
    confirm_tolerance: int = 2      # a candidate may flicker this many frames without losing its count
    same_id_min_app: float = 0.35   # even ByteTrack's own ID must look roughly right
    neg_margin: float = 0.03        # reject a candidate that resembles a known bystander more than the target
    neg_bank_size: int = 15

    # ---------- appearance memory ----------
    bank_size: int = 4
    update_every: int = 5           # frames between template updates
    update_min_conf: float = 0.60
    update_max_overlap: float = 0.10
    new_template_sim: float = 0.85  # add new template if less similar than this
