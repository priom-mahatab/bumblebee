import time
from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    FORWARD = "forward"
    SLOW = "slow"
    STOP = "stop"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"


class Phase(Enum):
    CRUISING = "cruising"  # normal driving; a centered stop sign arms a stop
    STOPPED = "stopped"    # holding at the sign for STOP_HOLD_SECONDS
    CLEARING = "clearing"  # stop served; drive through, ignore THIS sign


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
OBSTACLE_SLOW_CLOSENESS = 0.22  # lead vehicle bbox this tall (frac of frame height) -> SLOW
OBSTACLE_STOP_CLOSENESS = 0.45  # lead vehicle very close (fills ~half the frame) -> STOP
STOP_HOLD_SECONDS = 2.0  # how long to hold at a stop sign before proceeding
STOP_SIGN_ARM_FRAMES = 3  # consecutive sign frames before a stop is triggered
STOP_SIGN_CLEAR_FRAMES = 3  # consecutive sign-free frames before the latch resets


def _bbox_center_x_fraction(bbox: tuple[float, float, float, float], frame_width: int) -> float:
    x1, _, x2, _ = bbox
    return ((x1 + x2) / 2) / frame_width


def _in_path(bbox: tuple[float, float, float, float], frame_width: int) -> bool:
    low, high = IN_PATH_X_RANGE
    return low < _bbox_center_x_fraction(bbox, frame_width) < high


def stop_sign_in_path(detections: list[Detection], frame_width: int = 640) -> bool:
    """True if a stop sign is detected roughly centered ahead of the car."""
    return any(
        det.class_name in STOP_SIGN_CLASSES and _in_path(det.bbox, frame_width)
        for det in detections
    )


def decide_action(
    lane_offset: float | None,
    detections: list[Detection],
    distance_cm: float | None,
    frame_width: int = 640,
    frame_height: int = 480,
) -> Action:
    """
    Pure, stateless per-frame decision. A stop sign here only produces STOP
    while the sign is visible and centered — pause-and-go timing lives in
    StopSignController, which wraps this function.

    lane_offset: fraction of half-frame-width the lane center is offset
                 from image center, roughly in [-1, 1]. Negative means the
                 lane center is left of image center (car should turn left).
    detections: objects seen in the current frame.
    distance_cm: latest ultrasonic reading, or None if unavailable.
    """
    
    if distance_cm is not None and distance_cm < STOP_DISTANCE_CM:
        return Action.STOP

    if stop_sign_in_path(detections, frame_width):
        return Action.STOP

    # Distance-aware obstacle response (monocular, no ultrasonic on video):
    # a closer in-path vehicle has a taller bbox. Far cars are ignored.
    closeness = lead_vehicle_closeness(detections, frame_width, frame_height)
    if closeness is not None:
        if closeness >= OBSTACLE_STOP_CLOSENESS:
            return Action.STOP
        if closeness >= OBSTACLE_SLOW_CLOSENESS:
            return Action.SLOW

    if distance_cm is not None and distance_cm < SLOW_DISTANCE_CM:
        return Action.SLOW

    if lane_offset is None:
        return Action.SLOW
    if lane_offset > LANE_OFFSET_DEADZONE:
        return Action.TURN_RIGHT
    if lane_offset < -LANE_OFFSET_DEADZONE:
        return Action.TURN_LEFT
    return Action.FORWARD

def lead_vehicle_closeness(detections, frame_width, frame_height):
    """Height fraction of the closest in-path vehicle, or None if none ahead.
    Bigger = closer. Used as a monocular distance proxy (no ultrasonic)."""
    heights = []
    for det in detections:
        if det.class_name in OBSTACLE_CLASSES and _in_path(det.bbox, frame_width):
            x1, y1, x2, y2 = det.bbox
            heights.append((y2 - y1) / frame_height)
    return max(heights) if heights else None

class StopSignController:
    """
    Adds pause-and-go stop-sign behavior on top of the pure decide_action.

    All cross-frame memory (which phase we're in, when we stopped) lives here,
    so decide_action stays stateless and unit-testable. The clock is injected
    via update(now=...) so phases can be stepped deterministically in tests
    without real waiting or hardware.

    Phase changes are debounced: the sign must be seen for STOP_SIGN_ARM_FRAMES
    consecutive frames before a stop is armed, and be absent for
    STOP_SIGN_CLEAR_FRAMES consecutive frames before the latch resets, so a
    single dropped or spurious detection can't flip the phase.
    """

    def __init__(self) -> None:
        self.phase = Phase.CRUISING
        self.stopped_at: float | None = None
        self.sign_streak = 0  # consecutive frames with a sign in path
        self.no_sign_streak = 0  # consecutive frames without one

    def update(
        self,
        lane_offset: float | None,
        detections: list[Detection],
        distance_cm: float | None,
        frame_width: int = 640,
        frame_height: int = 480,
        now: float | None = None,
    ) -> Action:
        if now is None:
            now = time.monotonic()
        sign = stop_sign_in_path(detections, frame_width)
        if sign:
            self.sign_streak += 1
            self.no_sign_streak = 0
        else:
            self.no_sign_streak += 1
            self.sign_streak = 0

        armed = self.sign_streak >= STOP_SIGN_ARM_FRAMES
        cleared = self.no_sign_streak >= STOP_SIGN_CLEAR_FRAMES

        if self.phase is Phase.CRUISING:
            if armed:
                self.phase = Phase.STOPPED
                self.stopped_at = now
        elif self.phase is Phase.STOPPED:
            if cleared:
                # Sign gone for good before the hold completed (misdetection).
                self.phase = Phase.CRUISING
                self.stopped_at = None
            elif now - self.stopped_at >= STOP_HOLD_SECONDS:
                self.phase = Phase.CLEARING
        elif self.phase is Phase.CLEARING:
            # Stay until THIS sign leaves view, so it can't re-arm the stop.
            if cleared:
                self.phase = Phase.CRUISING

        # While holding, force STOP regardless of what else is in frame.
        if self.phase is Phase.STOPPED:
            return Action.STOP

        # While clearing, hide the stop sign from the pure decision so lanes,
        # obstacles, and ultrasonic still work but the sign no longer stops us.
        if self.phase is Phase.CLEARING:
            detections = [d for d in detections if d.class_name not in STOP_SIGN_CLASSES]

        return decide_action(lane_offset, detections, distance_cm, frame_width, frame_height)


if __name__ == "__main__":
    cases = [
        ("clear road, centered", 0.0, [], None),
        ("lane drifted right", 0.15, [], None),
        ("lane drifted left", -0.2, [], None),
        ("lane lost", None, [], None),
        ("car ahead, mid distance", 0.0, [Detection("car", 0.9, (280, 100, 360, 300))], 30),
        ("very close object", 0.0, [], 8),
        ("stop sign in path", 0.0, [Detection("stop sign", 0.95, (300, 50, 340, 90))], 100),
        ("stop sign off to the side", 0.0, [Detection("stop sign", 0.95, (10, 50, 50, 90))], 100),
    ]
    for label, offset, dets, dist in cases:
        action = decide_action(offset, dets, dist)
        print(f"{label:30s} -> {action.value}")

    print("\nStopSignController pause-and-go with debounce (sign stays in view):")
    controller = StopSignController()
    sign = [Detection("stop sign", 0.95, (300, 50, 340, 90))]
    timeline = [
        ("t=0.0 sign frame 1", 0.0, sign, 100, 0.0),
        ("t=0.1 sign frame 2", 0.0, sign, 100, 0.1),
        ("t=0.2 sign frame 3 (arm)", 0.0, sign, 100, 0.2),  # 3rd frame arms the stop
        ("t=1.0 holding", 0.0, sign, 100, 1.0),
        ("t=1.1 dropped detection", 0.0, [], 100, 1.1),  # one miss: hold survives
        ("t=2.5 hold done", 0.0, sign, 100, 2.5),
        ("t=3.0 clearing", 0.0, sign, 100, 3.0),
        ("t=3.1 sign gone 1", 0.0, [], None, 3.1),
        ("t=3.2 sign gone 2", 0.0, [], None, 3.2),
        ("t=3.3 sign gone 3 (reset)", 0.0, [], None, 3.3),  # 3 misses reset the latch
    ]
    for label, offset, dets, dist, clock in timeline:
        action = controller.update(offset, dets, dist, now=clock)
        print(f"{label:30s} -> {action.value:12s} [{controller.phase.value}]")
