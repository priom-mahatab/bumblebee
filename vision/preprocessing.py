"""Reusable image-processing building blocks for lane detection."""
import cv2
import numpy as np


def to_roi(frame: np.ndarray, roi_top_fraction: float = 0.55) -> np.ndarray:
    """Crop to the bottom portion of the frame, where lane lines appear."""
    height = frame.shape[0]
    top = int(height * roi_top_fraction)
    return frame[top:, :]


def white_mask(frame: np.ndarray, threshold: int = 200) -> np.ndarray:
    """Binary mask of near-white pixels — matches white/light tape lane
    markers on a darker floor/track."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    return mask
