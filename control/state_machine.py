import time
from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    FORWARD = "forward"
    SLOW = "slow"
    STOP = "stop"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    CHANGE_LEFT = "change_left"
    CHANGE_RIGHT = "change_right"


class Phase(Enum):
    CRUISING = "cruising"
    APPROACHING = "approaching"
    STOPPED = "stopped"
    CLEARING = "clearing"


class LaneChangePhase(Enum):
    IN_LANE = "in_lane"
    CHANGING = "changing"
    SETTLING = "settling"


@dataclass
class Detection:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixel coords


STOP_DISTANCE_CM = 15
SLOW_DISTANCE_CM = 40
LANE_OFFSET_DEADZONE = 0.08  # fraction of half-frame-width
IN_PATH_X_RANGE = (0.3, 0.7)  # fraction of frame width
STOP_SIGN_CLASSES = {"stop sign"}
OBSTACLE_CLASSES = {"person", "car", "truck", "bus"}
OBSTACLE_SLOW_CLOSENESS = 0.22  # bbox height fraction
OBSTACLE_STOP_CLOSENESS = 0.45  # bbox height fraction
STOP_LINE_MARGIN_M = 1.0
STOP_HOLD_SECONDS = 2.0
STOP_SIGN_ARM_FRAMES = 3
STOP_SIGN_CLEAR_FRAMES = 3
SPEED_LIMIT_VOTE_FRAMES = 3

LANE_CHANGE_TRIGGER_CLOSENESS = 0.30  # above SLOW, below STOP
LANE_CHANGE_ARM_FRAMES = 3
LANE_CHANGE_DISTANCE_M = 1.0
LANE_SETTLE_DISTANCE_M = 0.5
TARGET_LANE_BLOCKED_CLOSENESS = 0.15  # bbox height fraction
PREFERRED_SIDE = "left"
LEFT_LANE_X_RANGE = (0.0, 0.4)
RIGHT_LANE_X_RANGE = (0.6, 1.0)


def _bbox_center_x_fraction(bbox: tuple[float, float, float, float], frame_width: int) -> float:
    x1, _, x2, _ = bbox
    return ((x1 + x2) / 2) / frame_width


def _in_band(bbox: tuple[float, float, float, float], frame_width: int,
             band: tuple[float, float]) -> bool:
    """True if the bbox center falls in the given lateral band (fractions of frame width)."""
    low, high = band
    return low < _bbox_center_x_fraction(bbox, frame_width) < high


def _in_path(bbox: tuple[float, float, float, float], frame_width: int) -> bool:
    return _in_band(bbox, frame_width, IN_PATH_X_RANGE)


def _obstacle_in_band(detections: list[Detection], frame_width: int, frame_height: int,
                      band: tuple[float, float],
                      min_closeness: float = TARGET_LANE_BLOCKED_CLOSENESS) -> bool:
    """True if a vehicle sits in the given band with bbox height fraction >= min_closeness."""
    for det in detections:
        if det.class_name in OBSTACLE_CLASSES and _in_band(det.bbox, frame_width, band):
            x1, y1, x2, y2 = det.bbox
            if (y2 - y1) / frame_height >= min_closeness:
                return True
    return False


def stop_sign_visible(detections: list[Detection]) -> bool:
    """True if a stop sign is detected this frame (any size); whether to STOP is decided elsewhere."""
    return any(det.class_name in STOP_SIGN_CLASSES for det in detections)


def decide_action(
    lane_offset: float | None,
    detections: list[Detection],
    distance_cm: float | None,
    frame_width: int = 640,
    frame_height: int = 480,
) -> Action:
    """Pure, stateless per-frame decision for lanes, obstacles, and ultrasonic (not stop signs)."""

    if distance_cm is not None and distance_cm < STOP_DISTANCE_CM:
        return Action.STOP

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
    """Height fraction of the closest in-path vehicle (monocular distance proxy), or None."""
    heights = []
    for det in detections:
        if det.class_name in OBSTACLE_CLASSES and _in_path(det.bbox, frame_width):
            x1, y1, x2, y2 = det.bbox
            heights.append((y2 - y1) / frame_height)
    return max(heights) if heights else None

class StopSignController:
    """Range-and-dead-reckon stop-sign phase machine layered over decide_action."""

    def __init__(self) -> None:
        self.phase = Phase.CRUISING
        self.stopped_at: float | None = None
        self.stop_target_m: float | None = None
        self.sign_streak = 0
        self.no_sign_streak = 0

    def advance(
        self,
        detections: list[Detection],
        odometry_m: float = 0.0,
        stop_sign_distance_m: float | None = None,
        now: float | None = None,
    ) -> Action | None:
        """Step the stop-sign phase machine one frame, returning Action.STOP while holding else None."""
        if now is None:
            now = time.monotonic()
        seen = stop_sign_visible(detections)
        if seen:
            self.sign_streak += 1
            self.no_sign_streak = 0
        else:
            self.no_sign_streak += 1
            self.sign_streak = 0

        armed = self.sign_streak >= STOP_SIGN_ARM_FRAMES
        cleared = self.no_sign_streak >= STOP_SIGN_CLEAR_FRAMES

        if self.phase is Phase.CRUISING:
            if armed and stop_sign_distance_m is not None:
                self.stop_target_m = odometry_m + stop_sign_distance_m - STOP_LINE_MARGIN_M
                self.phase = Phase.APPROACHING
        elif self.phase is Phase.APPROACHING:
            if odometry_m >= self.stop_target_m:
                self.phase = Phase.STOPPED
                self.stopped_at = now
        elif self.phase is Phase.STOPPED:
            if now - self.stopped_at >= STOP_HOLD_SECONDS:
                self.phase = Phase.CLEARING
        elif self.phase is Phase.CLEARING:
            if cleared:
                self.phase = Phase.CRUISING
                self.stop_target_m = None

        if self.phase is Phase.STOPPED:
            return Action.STOP
        return None

    def update(
        self,
        lane_offset: float | None,
        detections: list[Detection],
        distance_cm: float | None,
        frame_width: int = 640,
        frame_height: int = 480,
        odometry_m: float = 0.0,
        stop_sign_distance_m: float | None = None,
        now: float | None = None,
    ) -> Action:
        """Backward-compatible wrapper: advance the stop-sign phase, else fall through to decide_action."""
        override = self.advance(detections, odometry_m, stop_sign_distance_m, now)
        if override is not None:
            return override
        return decide_action(lane_offset, detections, distance_cm, frame_width, frame_height)


class LaneChangeController:
    """Obstacle-triggered lane change as a committed, dead-reckoned maneuver (IN_LANE/CHANGING/SETTLING)."""

    def __init__(
        self,
        *,
        trigger_closeness: float = LANE_CHANGE_TRIGGER_CLOSENESS,
        arm_frames: int = LANE_CHANGE_ARM_FRAMES,
        change_distance_m: float = LANE_CHANGE_DISTANCE_M,
        settle_distance_m: float = LANE_SETTLE_DISTANCE_M,
        blocked_closeness: float = TARGET_LANE_BLOCKED_CLOSENESS,
        preferred_side: str = PREFERRED_SIDE,
        left_lane_band: tuple[float, float] = LEFT_LANE_X_RANGE,
        right_lane_band: tuple[float, float] = RIGHT_LANE_X_RANGE,
    ) -> None:
        self.trigger_closeness = trigger_closeness
        self.arm_frames = arm_frames
        self.change_distance_m = change_distance_m
        self.settle_distance_m = settle_distance_m
        self.blocked_closeness = blocked_closeness
        self.preferred_side = preferred_side
        self.left_lane_band = left_lane_band
        self.right_lane_band = right_lane_band

        self.phase = LaneChangePhase.IN_LANE
        self.direction: str | None = None
        self.maneuver_start_m: float | None = None
        self.arm_streak = 0

    def _choose_side(
        self,
        detections: list[Detection],
        frame_width: int,
        frame_height: int,
        left_lane_exists: bool,
        right_lane_exists: bool,
    ) -> str | None:
        """Pick a side that both exists and is clear, or None if neither works."""
        left_ok = left_lane_exists and not _obstacle_in_band(
            detections, frame_width, frame_height, self.left_lane_band, self.blocked_closeness)
        right_ok = right_lane_exists and not _obstacle_in_band(
            detections, frame_width, frame_height, self.right_lane_band, self.blocked_closeness)
        if left_ok and right_ok:
            return self.preferred_side
        if left_ok:
            return "left"
        if right_ok:
            return "right"
        return None

    def update(
        self,
        detections: list[Detection],
        frame_width: int,
        frame_height: int,
        odometry_m: float,
        left_lane_exists: bool,
        right_lane_exists: bool,
    ) -> Action | None:
        if self.phase is LaneChangePhase.IN_LANE:
            closeness = lead_vehicle_closeness(detections, frame_width, frame_height)
            if closeness is not None and closeness >= self.trigger_closeness:
                self.arm_streak += 1
            else:
                self.arm_streak = 0
            if self.arm_streak >= self.arm_frames:
                side = self._choose_side(detections, frame_width, frame_height,
                                         left_lane_exists, right_lane_exists)
                if side is not None:
                    self.direction = side
                    self.maneuver_start_m = odometry_m
                    self.phase = LaneChangePhase.CHANGING
                    self.arm_streak = 0
                    return Action.CHANGE_LEFT if side == "left" else Action.CHANGE_RIGHT
            return None

        if self.phase is LaneChangePhase.CHANGING:
            if odometry_m - self.maneuver_start_m >= self.change_distance_m:
                self.phase = LaneChangePhase.SETTLING
                self.maneuver_start_m = odometry_m
                return None
            return Action.CHANGE_LEFT if self.direction == "left" else Action.CHANGE_RIGHT

        if odometry_m - self.maneuver_start_m >= self.settle_distance_m:
            self.phase = LaneChangePhase.IN_LANE
            self.direction = None
            self.maneuver_start_m = None
        return None


class DrivingController:
    """Top-level coordinator applying the priority ladder: ultrasonic STOP > stop-sign hold > lane change > decide_action."""

    def __init__(self, lane_change: "LaneChangeController | None" = None) -> None:
        self.stop_sign = StopSignController()
        self.lane_change = lane_change if lane_change is not None else LaneChangeController()

    def update(
        self,
        lane_offset: float | None,
        detections: list[Detection],
        distance_cm: float | None,
        frame_width: int = 640,
        frame_height: int = 480,
        odometry_m: float = 0.0,
        stop_sign_distance_m: float | None = None,
        left_lane_exists: bool = False,
        right_lane_exists: bool = False,
        now: float | None = None,
    ) -> Action:
        if now is None:
            now = time.monotonic()

        if distance_cm is not None and distance_cm < STOP_DISTANCE_CM:
            return Action.STOP

        stop_override = self.stop_sign.advance(detections, odometry_m, stop_sign_distance_m, now)
        if stop_override is not None:
            return stop_override

        lane_override = self.lane_change.update(
            detections, frame_width, frame_height, odometry_m,
            left_lane_exists, right_lane_exists)
        if lane_override is not None:
            return lane_override

        return decide_action(lane_offset, detections, distance_cm, frame_width, frame_height)


class SpeedLimitTracker:
    """Holds the active speed limit (mph), committing a new read only after SPEED_LIMIT_VOTE_FRAMES agreeing frames."""

    def __init__(self) -> None:
        self.current_limit: int | None = None
        self._candidate: int | None = None
        self._streak = 0

    def update(self, reading: int | None) -> int | None:
        """Feed this frame's OCR reading (or None) and return the currently-in-force limit."""
        if reading is not None and reading == self._candidate:
            self._streak += 1
        else:
            self._candidate = reading
            self._streak = 1 if reading is not None else 0
        if self._streak >= SPEED_LIMIT_VOTE_FRAMES and self._candidate is not None:
            self.current_limit = self._candidate
        return self.current_limit


if __name__ == "__main__":
    cases = [
        ("clear road, centered", 0.0, [], None),
        ("lane drifted right", 0.15, [], None),
        ("lane drifted left", -0.2, [], None),
        ("lane lost", None, [], None),
        ("car ahead, mid distance", 0.0, [Detection("car", 0.9, (280, 100, 360, 300))], 30),
        ("very close object", 0.0, [], 8),
        ("stop sign (decide_action ignores it)", 0.0, [Detection("stop sign", 0.95, (300, 50, 340, 90))], 100),
    ]
    for label, offset, dets, dist in cases:
        action = decide_action(offset, dets, dist)
        print(f"{label:38s} -> {action.value}")

    print("\nStopSignController range-and-dead-reckon (sign 10 m ahead, stop 1 m before):")
    controller = StopSignController()
    sign = [Detection("stop sign", 0.95, (300, 50, 340, 90))]
    timeline = [
        ("odom 0.0  sign seen 1",             0.0, sign, None, 0.0, 10.0, 0.0),
        ("odom 0.5  sign seen 2",             0.0, sign, None, 0.5,  9.5, 0.1),
        ("odom 1.0  sign seen 3 (arm/latch)", 0.0, sign, None, 1.0,  9.0, 0.2),
        ("odom 5.0  approaching",             0.0, sign, None, 5.0,  5.0, 0.3),
        ("odom 8.5  approaching, sign gone",  0.0, [],   None, 8.5,  1.5, 0.4),
        ("odom 9.0  reached -> stop",         0.0, [],   None, 9.0,  1.0, 0.5),
        ("odom 9.0  holding",                 0.0, [],   None, 9.0,  1.0, 1.5),
        ("odom 9.0  hold done -> clear",      0.0, [],   None, 9.0,  1.0, 3.0),
        ("odom 9.5  clearing -> reset",       0.0, [],   None, 9.5, None, 3.1),
    ]
    for label, offset, dets, dist, odom, ssd, clock in timeline:
        action = controller.update(offset, dets, dist, odometry_m=odom,
                                   stop_sign_distance_m=ssd, now=clock)
        print(f"{label:38s} -> {action.value:8s} [{controller.phase.value}]")

    def lead(h):
        return [Detection("car", 0.9, (280, 240 - h // 2, 360, 240 + h // 2))]

    print("\nDrivingController obstacle-triggered lane change (left lane open, right absent):")
    drive = DrivingController()
    lane_timeline = [
        ("odom 0.0  lead closing, arm 1",      0.0, lead(160), 0.0, 0.0),
        ("odom 0.2  arm 2",                    0.0, lead(160), 0.2, 0.1),
        ("odom 0.4  arm 3 -> commit left",     0.0, lead(160), 0.4, 0.2),
        ("odom 0.9  changing",                 0.0, lead(160), 0.9, 0.3),
        ("odom 1.5  maneuver done -> settle",  0.0, [],        1.5, 0.4),
        ("odom 1.8  settling",                 0.0, [],        1.8, 0.5),
        ("odom 2.2  settled -> in lane",       0.0, [],        2.2, 0.6),
    ]
    for label, offset, dets, odom, clock in lane_timeline:
        action = drive.update(offset, dets, None, odometry_m=odom,
                              left_lane_exists=True, right_lane_exists=False, now=clock)
        print(f"{label:38s} -> {action.value:11s} [{drive.lane_change.phase.value}]")

    print("\nDrivingController obstacle, no adjacent lane -> falls back to STOP:")
    drive2 = DrivingController()
    for label, h, odom in [("lead mid", 120, 0.0), ("lead closer", 160, 0.2), ("lead very close", 230, 0.4)]:
        action = drive2.update(0.0, lead(h), None, odometry_m=odom,
                               left_lane_exists=False, right_lane_exists=False, now=0.0)
        print(f"{label:38s} -> {action.value}")
