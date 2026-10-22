import numpy as np
from dataclasses import dataclass


@dataclass
class Track:
    id: int
    box: np.ndarray  # x1,y1,x2,y2
    conf: float


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a, b):
    i = inter(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def ioa(a, b):
    """Fraction of box a that is covered by box b."""
    aa = area(a)
    return inter(a, b) / aa if aa > 0 else 0.0


def center(b):
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])


def wh(b):
    return np.array([b[2] - b[0], b[3] - b[1]])
