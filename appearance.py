"""Appearance features: upper/lower-body HSV histograms + a small template bank."""
import cv2
import numpy as np


def extract_feature(frame, box):
    """Two HSV histograms (torso band, legs band) from the central part of the box.
    Using only the central columns/bands reduces background pixels."""
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    w, h = x2 - x1, y2 - y1
    if w < 6 or h < 12:
        return None
    cx1, cx2 = x1 + int(0.2 * w), x2 - int(0.2 * w)
    bands = [(0.15, 0.50), (0.50, 0.90)]  # torso, legs
    feats = []
    for a, b in bands:
        roi = frame[y1 + int(a * h): y1 + int(b * h), cx1:cx2]
        if roi.size == 0:
            return None
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        hist = hist.astype(np.float32)
        s = hist.sum()
        feats.append(hist / s if s > 0 else hist)
    return feats


def similarity(f1, f2):
    """1 = identical colour distribution, 0 = very different. Mean over the two bands."""
    if f1 is None or f2 is None:
        return 0.0
    sims = []
    for a, b in zip(f1, f2):
        d = cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA)
        sims.append(1.0 - float(d))
    return float(np.mean(sims))


class AppearanceBank:
    """Small bank of templates (front/back/side look different).
    Updated only when the caller says the observation is high quality."""

    def __init__(self, size=4, new_sim=0.85):
        self.size = size
        self.new_sim = new_sim
        self.templates = []

    def add_initial(self, feat):
        if feat is not None:
            self.templates = [feat]

    def score(self, feat):
        if not self.templates or feat is None:
            return 0.0
        return max(similarity(t, feat) for t in self.templates)

    def update(self, feat):
        if feat is None or not self.templates:
            return
        sims = [similarity(t, feat) for t in self.templates]
        best = int(np.argmax(sims))
        if sims[best] < self.new_sim and len(self.templates) < self.size:
            self.templates.append(feat)
        else:  # slow moving average of the closest template
            self.templates[best] = [0.9 * t + 0.1 * f for t, f in zip(self.templates[best], feat)]
