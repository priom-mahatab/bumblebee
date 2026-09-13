"""Fake stand-ins for camera frames and sensor readings.

Lets vision/control code be exercised before the physical camera,
ultrasonic sensor, and track exist.
"""
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


def mock_distance_readings() -> list[float]:
    """A short scripted sequence of ultrasonic readings (cm), simulating
    an object slowly approaching the car.
    """
    return [120, 90, 60, 45, 30, 18, 10]


if __name__ == "__main__":
    frame = make_test_frame(lane_offset_px=20)
    print("frame shape:", frame.shape)
    print("distance sequence:", mock_distance_readings())
