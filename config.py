"""All tunable parameters in one place."""
from dataclasses import dataclass, field


@dataclass
class Config:
    # ---------- detection / tracking ----------
    model: str = "yolo11n.pt"
    det_conf: float = 0.30
    imgsz: int = 640

    # ---------- state machine ----------
    occluded_wait: int = 5          # frames to wait for ByteTrack to bring back the SAME id
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
    strict_accept: float = 0.62     # wide search is riskier, so it needs a stronger match
    strict_min_app: float = 0.55
    strict_confirm: int = 5

    # ---------- occlusion-type reasoning ----------
    edge_margin: float = 0.06       # fraction of frame size counted as "at the edge"
    exit_speed: float = 0.5         # px/frame outward speed to count as leaving
    person_overlap: float = 0.25    # fraction of target box covered by another person
    reentry_zone: float = 0.20      # EXITED: candidates must appear within this edge band

    # ---------- reacquisition scoring ----------
    accept_score: float = 0.55
    margin: float = 0.08            # best must beat 2nd best by this much (else AMBIGUOUS)
    confirm_frames: int = 3         # candidate must win this many frames in a row
    min_cand_conf: float = 0.35
    min_appearance: float = 0.30    # candidate must at least roughly look like the target
    pred_cap_frames: int = 20       # stop extrapolating velocity after this many frames
    adapt_frames: int = 90          # position->appearance weight shift over this time
    same_id_bonus: float = 0.10

    # ---------- appearance memory ----------
    bank_size: int = 4
    update_every: int = 5           # frames between template updates
    update_min_conf: float = 0.60
    update_max_overlap: float = 0.10
    new_template_sim: float = 0.85  # add new template if less similar than this
