"""Per-run logger: writes frame telemetry to runs/<base>/<timestamp>/log.csv and saves raw frames alongside."""
import csv
import os
from datetime import datetime

import cv2


class RunLogger:
    """Timestamped run directory holding a CSV telemetry log and saved frames for offline review and tuning."""

    def __init__(self, base: str = "runs/rover", save_frames: bool = True, frame_every: int = 1):
        self.dir = os.path.join(base, datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.frames_dir = os.path.join(self.dir, "frames")
        os.makedirs(self.frames_dir if save_frames else self.dir, exist_ok=True)
        self.save_frames = save_frames
        self.frame_every = max(1, frame_every)
        self.csv_path = os.path.join(self.dir, "log.csv")
        self._file = open(self.csv_path, "w", newline="")
        self._writer: csv.DictWriter | None = None
        self.count = 0

    def log(self, record: dict, frame=None) -> None:
        """Append one telemetry row (keys become CSV columns on the first call) and save the frame."""
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(record))
            self._writer.writeheader()
        self._writer.writerow(record)
        self._file.flush()
        if self.save_frames and frame is not None and self.count % self.frame_every == 0:
            cv2.imwrite(os.path.join(self.frames_dir, f"{self.count:05d}.jpg"), frame)
        self.count += 1

    def close(self) -> None:
        self._file.close()
        print(f"logged {self.count} frames to {self.dir}")
