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
    CRUISING = "cruising"        # normal driving; a visible stop sign arms an approach
    APPROACHING = "approaching"  # sign ranged; rolling forward to the latched stop point
    STOPPED = "stopped"          # holding at the sign for STOP_HOLD_SECONDS
    CLEARING = "clearing"        # stop served; drive through, ignore THIS sign


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
STOP_LINE_MARGIN_M = 1.0  # stop this far BEFORE the stop sign's along-road position
STOP_HOLD_SECONDS = 2.0  # how long to hold at a stop sign before proceeding
STOP_SIGN_ARM_FRAMES = 3  # consecutive sign frames before an approach is armed
STOP_SIGN_CLEAR_FRAMES = 3  # consecutive sign-free frames before the latch resets


def _bbox_center_x_fraction(bbox: tuple[float, float, float, float], frame_width: int) -> float:
    x1, _, x2, _ = bbox
    return ((x1 + x2) / 2) / frame_width


def _in_path(bbox: tuple[float, float, float, float], frame_width: int) -> bool:
    low, high = IN_PATH_X_RANGE
    return low < _bbox_center_x_fraction(bbox, frame_width) < high


def stop_sign_visible(detections: list[Detection]) -> bool:
    """True if a stop sign is detected this frame (any size).

    Whether to STOP is NOT decided here. A roadside sign sits off to the side,
    so it slides to the frame edge and out of view before the car reaches it —
    we can't stop on its bbox size or position. Instead the controller RANGES
    the sign when it first becomes visible (perception supplies the along-road
    distance) and then dead-reckons forward to a latched stop point. This just
    answers "do we see one?".
    """
    return any(det.class_name in STOP_SIGN_CLASSES for det in detections)


def decide_action(
    lane_offset: float | None,
    detections: list[Detection],
    distance_cm: float | None,
    frame_width: int = 640,
    frame_height: int = 480,
) -> Action:
    """
    Pure, stateless per-frame decision for lanes, obstacles, and ultrasonic.
    Stop-sign behavior is NOT handled here — it's range-and-dead-reckon logic
    that lives entirely in StopSignController, which wraps this function.

    lane_offset: fraction of half-frame-width the lane center is offset
                 from image center, roughly in [-1, 1]. Negative means the
                 lane center is left of image center (car should turn left).
    detections: objects seen in the current frame.
    distance_cm: latest ultrasonic reading, or None if unavailable.
    """
    
    if distance_cm is not None and distance_cm < STOP_DISTANCE_CM:
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
    Adds range-and-dead-reckon stop-sign behavior on top of decide_action.

    A roadside stop sign leaves view before the car reaches it, so we can't stop
    on its bbox. Instead: CRUISING -> (sign seen + ranged) latch a stop point one
    margin before the sign and go APPROACHING -> (odometry reaches it) STOPPED and
    hold -> CLEARING -> CRUISING (re-armable for the next sign). All cross-frame
    memory (phase, when we stopped, the latched stop point) lives here, so
    decide_action stays stateless and unit-testable. The clock is injected via
    update(now=...) so phases can be stepped deterministically in tests.

    Arming is debounced: the sign must be seen for STOP_SIGN_ARM_FRAMES
    consecutive frames before an approach latches, and be absent for
    STOP_SIGN_CLEAR_FRAMES consecutive frames before the latch resets, so a
    single dropped or spurious detection can't flip the phase.
    """

    def __init__(self) -> None:
        self.phase = Phase.CRUISING
        self.stopped_at: float | None = None
        self.stop_target_m: float | None = None  # latched odometry at which to STOP
        self.sign_streak = 0  # consecutive frames a sign is visible
        self.no_sign_streak = 0  # consecutive frames without one

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
        """Range-and-dead-reckon stop-sign handling wrapped around decide_action.

        odometry_m: how far the car has driven along its path, as arc length.
        stop_sign_distance_m: curve-aware along-road distance to the next stop
            sign ahead, supplied by perception (the sim computes it from the
            course; a real car would range it off the bbox). Used ONLY at the
            instant we latch — afterward we dead-reckon on odometry, because the
            sign slides out of view before the car reaches it.
        """
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
            # First confident sighting WITH a range -> latch the stop point and
            # start rolling toward it. The sign may leave view from here on.
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
            # Hold until THIS sign has left view, so it can't re-arm; then reset
            # so the NEXT stop sign down the road arms fresh.
            if cleared:
                self.phase = Phase.CRUISING
                self.stop_target_m = None

        # Forced STOP only while serving the hold.
        if self.phase is Phase.STOPPED:
            return Action.STOP

        # Every other phase: lanes/obstacles/ultrasonic drive us. Stop signs
        # never stop us through decide_action — the phase logic above owns that.
        return decide_action(lane_offset, detections, distance_cm, frame_width, frame_height)


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
    #  label, lane_offset, detections, dist_cm, odometry_m, stop_sign_distance_m, clock
    timeline = [
        ("odom 0.0  sign seen 1",             0.0, sign, None, 0.0, 10.0, 0.0),
        ("odom 0.5  sign seen 2",             0.0, sign, None, 0.5,  9.5, 0.1),
        ("odom 1.0  sign seen 3 (arm/latch)", 0.0, sign, None, 1.0,  9.0, 0.2),  # target = 1+9-1 = 9 m
        ("odom 5.0  approaching",             0.0, sign, None, 5.0,  5.0, 0.3),
        ("odom 8.5  approaching, sign gone",  0.0, [],   None, 8.5,  1.5, 0.4),
        ("odom 9.0  reached -> stop",         0.0, [],   None, 9.0,  1.0, 0.5),
        ("odom 9.0  holding",                 0.0, [],   None, 9.0,  1.0, 1.5),
        ("odom 9.0  hold done -> clear",      0.0, [],   None, 9.0,  1.0, 3.0),  # >2 s hold
        ("odom 9.5  clearing -> reset",       0.0, [],   None, 9.5, None, 3.1),
    ]
    for label, offset, dets, dist, odom, ssd, clock in timeline:
        action = controller.update(offset, dets, dist, odometry_m=odom,
                                   stop_sign_distance_m=ssd, now=clock)
        print(f"{label:38s} -> {action.value:8s} [{controller.phase.value}]")
