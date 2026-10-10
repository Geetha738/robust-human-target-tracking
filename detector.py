"""YOLO11n detection + ByteTrack (via Ultralytics) -> list of Track."""
import numpy as np
from utils import Track


class PersonTracker:
    def __init__(self, cfg):
        from ultralytics import YOLO          # imported lazily so tests run without it
        self.cfg = cfg
        self.model = YOLO(cfg.model)

    def __call__(self, frame):
        r = self.model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[0],
                             conf=self.cfg.det_conf, imgsz=self.cfg.imgsz, verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            return []
        xyxy = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        return [Track(int(i), b.astype(float), float(c)) for i, b, c in zip(ids, xyxy, conf)]
