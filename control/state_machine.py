import time
from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    FORWARD = "forward"
    SLOW = "slow"
    STOP = "stop"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    CHANGE_LEFT = "change_left"    # cross into the adjacent left lane (sharper, sustained than TURN_LEFT)
    CHANGE_RIGHT = "change_right"  # cross into the adjacent right lane


class Phase(Enum):
    CRUISING = "cruising"        # normal driving; a visible stop sign arms an approach
    APPROACHING = "approaching"  # sign ranged; rolling forward to the latched stop point
    STOPPED = "stopped"          # holding at the sign for STOP_HOLD_SECONDS
    CLEARING = "clearing"        # stop served; drive through, ignore THIS sign


class LaneChangePhase(Enum):
    IN_LANE = "in_lane"      # tracking the current lane; watching for an obstacle to trigger a change
    CHANGING = "changing"    # committed lateral maneuver, dead-reckoned on odometry
    SETTLING = "settling"    # re-centering in the new lane; a fresh change is suppressed here


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
SPEED_LIMIT_VOTE_FRAMES = 3  # consecutive agreeing OCR reads before a new limit sticks

# --- Lane-change (obstacle-avoidance) thresholds. Tune on the real track. ---
LANE_CHANGE_TRIGGER_CLOSENESS = 0.30  # in-path lead this close -> prefer a lane change over stopping (above SLOW, below STOP)
LANE_CHANGE_ARM_FRAMES = 3            # consecutive trigger frames before a change commits (debounce)
LANE_CHANGE_DISTANCE_M = 1.0          # odometry length of the committed lateral maneuver (dead-reckoned)
LANE_SETTLE_DISTANCE_M = 0.5          # distance after the maneuver before a new change can arm (anti ping-pong)
TARGET_LANE_BLOCKED_CLOSENESS = 0.15  # a vehicle at least this tall in the target lane's band -> lane not clear
PREFERRED_SIDE = "left"               # which way to go when both adjacent lanes are open and clear
LEFT_LANE_X_RANGE = (0.0, 0.4)        # frame-width band treated as the left lane, for the target-clear check
RIGHT_LANE_X_RANGE = (0.6, 1.0)       # frame-width band treated as the right lane


def _bbox_center_x_fraction(bbox: tuple[float, float, float, float], frame_width: int) -> float:
    x1, _, x2, _ = bbox
    return ((x1 + x2) / 2) / frame_width


def _in_band(bbox: tuple[float, float, float, float], frame_width: int,
             band: tuple[float, float]) -> bool:
    """True if the bbox center falls in the given lateral band (fractions of
    frame width). Generalizes _in_path so the same check works for the lane
    straight ahead AND for an adjacent target lane."""
    low, high = band
    return low < _bbox_center_x_fraction(bbox, frame_width) < high


def _in_path(bbox: tuple[float, float, float, float], frame_width: int) -> bool:
    return _in_band(bbox, frame_width, IN_PATH_X_RANGE)


def _obstacle_in_band(detections: list[Detection], frame_width: int, frame_height: int,
                      band: tuple[float, float],
                      min_closeness: float = TARGET_LANE_BLOCKED_CLOSENESS) -> bool:
    """True if a vehicle sits in the given lateral band and is close enough
    (bbox height fraction >= min_closeness) to make that lane unsafe to enter.
    A tiny, far car in the target lane does not block it."""
    for det in detections:
        if det.class_name in OBSTACLE_CLASSES and _in_band(det.bbox, frame_width, band):
            x1, y1, x2, y2 = det.bbox
            if (y2 - y1) / frame_height >= min_closeness:
                return True
    return False


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

    def advance(
        self,
        detections: list[Detection],
        odometry_m: float = 0.0,
        stop_sign_distance_m: float | None = None,
        now: float | None = None,
    ) -> Action | None:
        """Step the stop-sign phase machine one frame. Returns Action.STOP while
        holding at the sign, else None (no stop-sign override this frame).

        This is PURE stop-sign phase logic — no lane-keeping. The coordinator
        (DrivingController) calls this directly and supplies the base action when
        it returns None; the legacy update() below wraps it around decide_action
        so existing callers (main.py, the sim) keep their old behavior.

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

        # Forced STOP only while serving the hold; every other phase defers to
        # the caller's base logic (the stop-sign phase never stops us any other
        # way — it owns that decision here).
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
        """Backward-compatible wrapper: advance the stop-sign phase, and if it
        isn't forcing a hold, fall through to the stateless decide_action base.
        Unchanged signature/behavior so main.py and the sim keep working; new
        code should prefer DrivingController, which also layers lane changing."""
        override = self.advance(detections, odometry_m, stop_sign_distance_m, now)
        if override is not None:
            return override
        return decide_action(lane_offset, detections, distance_cm, frame_width, frame_height)


class LaneChangeController:
    """Obstacle-triggered lane change, layered on as a committed, dead-reckoned
    maneuver — the same latch-and-dead-reckon idea as StopSignController, applied
    to a lateral move instead of a stop.

      IN_LANE  : watch for a close in-path obstacle. When one is confidently seen
                 (debounced over LANE_CHANGE_ARM_FRAMES) AND an adjacent lane
                 exists and is clear, commit a change toward it.
      CHANGING : steer that way open-loop for LANE_CHANGE_DISTANCE_M of odometry.
                 The detector's left/right split is meaningless mid-cross, so we
                 dead-reckon the maneuver rather than vision-track it, and we do
                 NOT re-decide on the obstacle — we're going around it.
      SETTLING : hand steering back to lane-keeping so the car re-centers in the
                 new lane, but suppress a fresh trigger for LANE_SETTLE_DISTANCE_M
                 so it can't ping-pong. Then back to IN_LANE.

    update() returns an overriding Action (CHANGE_LEFT/RIGHT) only while
    committing or maneuvering; None means 'no lane-change override this frame'
    and the caller's base logic (lane-keeping, obstacle SLOW/STOP) applies — so
    with no valid lane to take, the car simply falls back to slowing/stopping.
    Adjacency (left/right_lane_exists) and target-lane-clear arrive as perception
    INPUTS, exactly as the stop-sign controller takes stop_sign_distance_m, so
    this stays pure, testable decision logic. All cross-frame memory lives here.
    The maneuver is odometry-based, so an emergency STOP upstream freezes odometry
    and naturally pauses it until the car moves again.
    """

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
        # Tunables default to the module constants (small-rover scale); the sim
        # passes road-scale values (longer maneuver, prefer right). See run_sim.py.
        self.trigger_closeness = trigger_closeness
        self.arm_frames = arm_frames
        self.change_distance_m = change_distance_m
        self.settle_distance_m = settle_distance_m
        self.blocked_closeness = blocked_closeness
        self.preferred_side = preferred_side
        self.left_lane_band = left_lane_band
        self.right_lane_band = right_lane_band

        self.phase = LaneChangePhase.IN_LANE
        self.direction: str | None = None        # 'left' / 'right' while a change is live
        self.maneuver_start_m: float | None = None  # odometry at which the current phase began
        self.arm_streak = 0                      # consecutive frames the trigger has held

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
            return None  # no valid change -> caller's base logic slows/stops instead

        if self.phase is LaneChangePhase.CHANGING:
            if odometry_m - self.maneuver_start_m >= self.change_distance_m:
                self.phase = LaneChangePhase.SETTLING
                self.maneuver_start_m = odometry_m
                return None  # maneuver done; let lane-keeping re-center this frame
            return Action.CHANGE_LEFT if self.direction == "left" else Action.CHANGE_RIGHT

        # SETTLING: lane-keeping steers (return None), but no new change may arm
        # until we've travelled the settle distance.
        if odometry_m - self.maneuver_start_m >= self.settle_distance_m:
            self.phase = LaneChangePhase.IN_LANE
            self.direction = None
            self.maneuver_start_m = None
        return None


class DrivingController:
    """Top-level coordinator. Composes the stop-sign and lane-change phase
    machines over the stateless decide_action base with ONE explicit priority
    ladder, so the stateful controllers can't fight over the output:

        emergency ultrasonic STOP  >  stop-sign hold  >  lane-change maneuver
        >  lane-keeping / obstacle SLOW-STOP (decide_action)

    Each stateful layer returns an overriding Action or None; the first override
    wins, otherwise decide_action drives. This replaces NESTING the controllers,
    which gets ambiguous once more than one can force an action.

    main.py and the sim still drive StopSignController directly; they move onto
    this once lane detection reports adjacency (left_lane_exists /
    right_lane_exists). Until then those inputs default False and the lane-change
    layer is inert, so adopting DrivingController never regresses current behavior.
    """

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

        # 1. Emergency ultrasonic stop — wins over everything, including an
        #    in-progress maneuver. Odometry then stops advancing, so a maneuver
        #    simply pauses and resumes once the car is moving again.
        if distance_cm is not None and distance_cm < STOP_DISTANCE_CM:
            return Action.STOP

        # 2. Stop-sign hold.
        stop_override = self.stop_sign.advance(detections, odometry_m, stop_sign_distance_m, now)
        if stop_override is not None:
            return stop_override

        # 3. Lane-change trigger / maneuver.
        lane_override = self.lane_change.update(
            detections, frame_width, frame_height, odometry_m,
            left_lane_exists, right_lane_exists)
        if lane_override is not None:
            return lane_override

        # 4. Base: lane-keeping + obstacle SLOW/STOP + ultrasonic SLOW.
        return decide_action(lane_offset, detections, distance_cm, frame_width, frame_height)


class SpeedLimitTracker:
    """Holds the active speed limit, committing a newly-read one only after it's
    been recognized on SPEED_LIMIT_VOTE_FRAMES consecutive frames.

    Per-frame OCR is noisy, so a single misread shouldn't jerk the cruise speed.
    `current_limit` is the last committed value (mph), or None until the first
    confident reading; it persists after the sign leaves view (a limit stays in
    force until the next sign changes it), exactly like real driving.
    """

    def __init__(self) -> None:
        self.current_limit: int | None = None
        self._candidate: int | None = None
        self._streak = 0

    def update(self, reading: int | None) -> int | None:
        """Feed this frame's OCR reading (or None if no/failed read). Returns the
        currently-in-force limit."""
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

    def lead(h):  # a car dead ahead, h px tall -> closeness h/480 of the frame
        return [Detection("car", 0.9, (280, 240 - h // 2, 360, 240 + h // 2))]

    print("\nDrivingController obstacle-triggered lane change (left lane open, right absent):")
    drive = DrivingController()
    #  label, lane_offset, detections, odometry_m, clock
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
