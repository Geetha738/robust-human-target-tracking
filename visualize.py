import cv2
import numpy as np
from target_manager import State

COLORS = {
    State.IDLE: (200, 200, 200), State.ACQUIRED: (0, 220, 0), State.OCCLUDED: (0, 200, 255),
    State.REACQUIRING: (0, 140, 255), State.LOST: (0, 0, 255),
}


def _label(img, text, org, fg, bg, scale=0.6, th=2):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = org
    cv2.rectangle(img, (x, y - h - 6), (x + w + 6, y + 2), bg, -1)
    cv2.putText(img, text, (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, th)


def draw(frame, tracks, info, hint=None, fps=None):
    out = frame.copy()
    H, W = out.shape[:2]
    col = COLORS[info.state]
    thick = max(2, W // 400)                       # scales with video resolution
    selecting = info.state == State.IDLE
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.box)
        is_t = info.state == State.ACQUIRED and t.id == info.target_id
        if is_t:
            overlay = out.copy()                   # translucent green fill = very obvious
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), -1)
            out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), thick * 2)
            _label(out, f"TARGET #{t.id}", (x1, max(22, y1 - 4)), (0, 0, 0), (0, 255, 0), 0.7, 2)
        else:
            c = (0, 215, 255) if selecting else (150, 150, 150)   # yellow while choosing
            cv2.rectangle(out, (x1, y1), (x2, y2), c, thick)
            _label(out, f"#{t.id}", (x1, max(18, y1 - 2)), (0, 0, 0), c, 0.5, 1)
    if info.state in (State.OCCLUDED, State.REACQUIRING) and info.search_center is not None:
        if info.edge_zone:
            z = 0.2
            cv2.rectangle(out, (int(z * W), int(z * H)), (int((1 - z) * W), int((1 - z) * H)), col, 1)
        else:
            c = (int(info.search_center[0]), int(info.search_center[1]))
            cv2.circle(out, c, int(min(info.search_radius, max(W, H))), col, thick)
            cv2.circle(out, c, 5, col, -1)
    label = info.state.value
    if info.state in (State.OCCLUDED, State.REACQUIRING):
        label += f" ({info.occ_type.lower()}, {info.frames_occluded}f)"
    if info.wide and info.state == State.REACQUIRING:
        label += "  [searching whole frame]"
    if info.state == State.LOST:
        label += " - still watching for the target"
    if info.ambiguous:
        label += "  AMBIGUOUS - waiting"
    bar = max(28, H // 20)
    cv2.rectangle(out, (0, 0), (W, bar), (30, 30, 30), -1)
    cv2.putText(out, label, (8, int(bar * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, bar / 45.0, col, 2)
    if fps:
        cv2.putText(out, f"{fps:.0f} FPS", (W - 110, int(bar * 0.72)), cv2.FONT_HERSHEY_SIMPLEX,
                    bar / 50.0, (255, 255, 255), 2)
    if hint:
        cv2.rectangle(out, (0, H - bar), (W, H), (30, 30, 30), -1)
        cv2.putText(out, hint, (8, H - int(bar * 0.28)), cv2.FONT_HERSHEY_SIMPLEX, bar / 50.0, (255, 255, 255), 2)
    return out
