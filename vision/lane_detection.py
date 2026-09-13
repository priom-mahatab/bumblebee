"""Turns a raw frame into a lane-center offset the control loop can steer on."""
import numpy as np

from vision.preprocessing import to_roi, white_mask


def compute_lane_offset(frame: np.ndarray, roi_top_fraction: float = 0.55) -> float | None:
    """
    Returns the lane center's horizontal offset from the frame's center,
    as a fraction of half-frame-width, roughly in [-1, 1]. Negative means
    the lane center is left of image center (car should turn left).
    Returns None if no lane pixels are found in the ROI.
    """
    roi = to_roi(frame, roi_top_fraction)
    mask = white_mask(roi)

    col_has_lane = np.any(mask > 0, axis=0)
    lane_columns = np.where(col_has_lane)[0]
    if lane_columns.size == 0:
        return None

    lane_center_px = lane_columns.mean()
    frame_center_px = mask.shape[1] / 2
    return float((lane_center_px - frame_center_px) / frame_center_px)


if __name__ == "__main__":
    from sim.mock_inputs import make_test_frame

    for shift in (-80, -20, 0, 20, 80):
        frame = make_test_frame(lane_offset_px=shift)
        offset = compute_lane_offset(frame)
        print(f"shift={shift:4d}px -> lane_offset={offset:+.3f}")
