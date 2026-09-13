"""Pure decision logic for the control loop — no hardware calls.

Kept separate from motor_control.py so it can be written and tested
before any hardware arrives: feed it numbers, get back an Action.
"""
from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    FORWARD = "forward"
    SLOW = "slow"
    STOP = "stop"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixel coords


# Tunable thresholds — adjust once real sensor/camera data is available.
STOP_DISTANCE_CM = 15
SLOW_DISTANCE_CM = 40
LANE_OFFSET_DEADZONE = 0.08  # fraction of half-frame-width
IN_PATH_X_RANGE = (0.3, 0.7)  # fraction of frame width counted as "ahead"
STOP_SIGN_CLASSES = {"stop sign"}
OBSTACLE_CLASSES = {"person", "car", "truck", "bus"}


def _bbox_center_x_fraction(bbox: tuple[float, float, float, float], frame_width: int) -> float:
    x1, _, x2, _ = bbox
    return ((x1 + x2) / 2) / frame_width


def decide_action(
    lane_offset: float,
    detections: list[Detection],
    distance_cm: float | None,
    frame_width: int = 640,
) -> Action:
    """
    lane_offset: fraction of half-frame-width the lane center is offset
                 from image center, roughly in [-1, 1]. Negative means the
                 lane center is left of image center (car should turn left).
    detections: objects seen in the current frame.
    distance_cm: latest ultrasonic reading, or None if unavailable.
    """
    if distance_cm is not None and distance_cm < STOP_DISTANCE_CM:
        return Action.STOP

    low, high = IN_PATH_X_RANGE
    for det in detections:
        center_x = _bbox_center_x_fraction(det.bbox, frame_width)
        if not (low < center_x < high):
            continue
        if det.class_name in STOP_SIGN_CLASSES:
            return Action.STOP
        if det.class_name in OBSTACLE_CLASSES:
            return Action.SLOW

    if distance_cm is not None and distance_cm < SLOW_DISTANCE_CM:
        return Action.SLOW

    if lane_offset > LANE_OFFSET_DEADZONE:
        return Action.TURN_RIGHT
    if lane_offset < -LANE_OFFSET_DEADZONE:
        return Action.TURN_LEFT
    return Action.FORWARD


if __name__ == "__main__":
    cases = [
        ("clear road, centered", 0.0, [], None),
        ("lane drifted right", 0.15, [], None),
        ("lane drifted left", -0.2, [], None),
        ("car ahead, mid distance", 0.0, [Detection("car", 0.9, (280, 100, 360, 300))], 30),
        ("very close object", 0.0, [], 8),
        ("stop sign in path", 0.0, [Detection("stop sign", 0.95, (300, 50, 340, 90))], 100),
        ("stop sign off to the side", 0.0, [Detection("stop sign", 0.95, (10, 50, 50, 90))], 100),
    ]
    for label, offset, dets, dist in cases:
        action = decide_action(offset, dets, dist)
        print(f"{label:30s} -> {action.value}")
