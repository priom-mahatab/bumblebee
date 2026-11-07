"""Fake stand-ins for camera frames and sensor readings.

Lets vision/control code be exercised before the physical camera,
ultrasonic sensor, and track exist.
"""
import cv2
import numpy as np


def make_test_frame(width: int = 640, height: int = 480, lane_offset_px: int = 0) -> np.ndarray:
    """A synthetic frame with two vertical white 'lane line' stripes,
    offset by lane_offset_px from centered, so lane_detection.py can be
    tested without a physical track.
    """
    frame = np.full((height, width, 3), 40, dtype=np.uint8)  # dark 'road'
    lane_gap = width // 2
    line_width = 8
    left_x = width // 2 - lane_gap // 2 + lane_offset_px
    right_x = width // 2 + lane_gap // 2 + lane_offset_px

    left_x = max(left_x, 0)
    right_x = min(right_x, width - line_width)
    frame[:, left_x:left_x + line_width] = (255, 255, 255)
    frame[:, right_x:right_x + line_width] = (255, 255, 255)
    return frame


def make_lane_frame(boundary_fracs, width: int = 640, height: int = 480,
                    vanishing_frac: float = 0.5, converge: float = 0.55,
                    line_width: int = 8, y_bottom_frac: float = 0.98,
                    y_top_frac: float = 0.5) -> np.ndarray:
    """Synthetic multi-lane road: one white boundary line per fraction in
    boundary_fracs (its x at the bottom edge, as a fraction of width), each
    slanting toward a vanishing point so it looks like perspective.

    The slant matters: make_test_frame drew PERFECTLY VERTICAL stripes, which the
    detector drops (x2 == x1, and zero horizontal run), so it read as None. Real
    lane lines converge toward a vanishing point, which is both what the camera
    sees and what Hough can actually fit — so the adjacency logic can be tested.
    """
    frame = np.full((height, width, 3), 40, dtype=np.uint8)  # dark 'road'
    vx = vanishing_frac * width
    yb, yt = int(y_bottom_frac * height), int(y_top_frac * height)
    for bf in boundary_fracs:
        bx = bf * width
        tx = bx + converge * (vx - bx)          # pull the far end toward the vanishing point
        cv2.line(frame, (int(bx), yb), (int(tx), yt), (255, 255, 255), line_width)
    return frame


def mock_distance_readings() -> list[float]:
    """A short scripted sequence of ultrasonic readings (cm), simulating
    an object slowly approaching the car.
    """
    return [120, 90, 60, 45, 30, 18, 10]


if __name__ == "__main__":
    frame = make_test_frame(lane_offset_px=20)
    print("frame shape:", frame.shape)
    print("distance sequence:", mock_distance_readings())
