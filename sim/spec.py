"""Camera, rover, and world constants for the 3D simulator.

These are the sim's tunables, kept here as module-level constants (same
convention as control/state_machine.py) rather than scattered inline.

SCALE NOTE — the simulated world is at *real-road* scale (metres), matching
sim/worlds/track.wbt (7 m lanes, full-size signs and cars). That's deliberate:
the detector is YOLO trained to see real road scenes, so a road-scale world
produces frames that look like what it expects. The physical mini-rover is of
course much smaller/slower; the values flagged ASSUMPTION below are road-scale
stand-ins for the demo and should be reconciled with hardware/ once the bill of
materials exists.

The one spec that transfers unchanged from the real camera to the sim is the
lens field-of-view — a lens is a lens regardless of world scale.
"""

# --- Camera intrinsics ----------------------------------------------------
# Processing resolution. 640x480 matches sim/mock_inputs.make_test_frame and
# control.state_machine's default frame_width, so nothing downstream changes.
CAM_WIDTH = 640
CAM_HEIGHT = 480

# Horizontal field of view in degrees. 62.2 deg is the Raspberry Pi Camera
# Module v2 spec. ASSUMPTION until the actual camera module is confirmed in
# hardware/; if you switch to a wide module (e.g. Camera Module 3 Wide, ~102 deg)
# change this one number.
CAM_HFOV_DEG = 62.2

# --- Camera mounting (road-scale) -----------------------------------------
CAM_HEIGHT_M = 1.2          # ASSUMPTION: dash height for the road-scale scene
CAM_PITCH_DEG = 8.0         # downward tilt so the road fills the lower frame
CAM_FORWARD_OFFSET_M = 1.5  # camera sits ahead of the rover's turn centre

# --- Rover kinematics (road-scale) ----------------------------------------
# Simple bicycle-ish steering: the control loop sets a target behaviour
# (forward / slow / stop / turn) and these govern how fast that plays out.
CRUISE_SPEED_MPS = 4.0      # ASSUMPTION (scaled down from 8.0 for the road size)
SLOW_SPEED_MPS = 1.5
TURN_YAW_RATE_DPS = 17.5    # deg/sec: the MAX (clamp) heading-change rate
STEER_GAIN_DPS = 100.0      # P-controller gain: yaw_rate = GAIN * lane_offset,
                            # clamped to +/-TURN_YAW_RATE_DPS. Higher = sharper
                            # correction (risk of oscillation); lower = lazier.
                            # NOTE: steering is per-TIME, so yaw-per-metre is
                            # GAIN/speed and max curvature is CLAMP/speed. To
                            # change cruise speed WITHOUT changing the driven
                            # path, scale CRUISE, SLOW, GAIN and CLAMP together
                            # (all were halved from 8.0/3.0/200/35).
STOP_HOLD_SECONDS = 1.5     # how long to sit at a stop sign before resuming
# Map a recognized speed limit (mph) to a sim cruise speed. The sim's speeds are
# road-scale stand-ins, so we scale so the 35 sign == the base cruise and lower
# limits are proportionally slower (25 -> ~2.9 m/s). Tunable.
SPEED_LIMIT_MPS_PER_MPH = CRUISE_SPEED_MPS / 35.0

# --- Road geometry --------------------------------------------------------
LANE_WIDTH_M = 3.5          # matches track.wbt
NUM_LANES = 3               # "3/4 lane road" — set 4 for four lanes
EGO_LANE_INDEX = 1          # 0 = leftmost; ego drives in the centre lane
EDGE_LINE_WIDTH_M = 0.15    # painted line width (matches track.wbt RoadLine)
DASH_LENGTH_M = 3.0         # dashed divider: dash + gap pattern
DASH_GAP_M = 3.0

# --- Lane-change (obstacle avoidance) -------------------------------------
# Controller tunables, road-scale — they override state_machine's small-rover
# defaults when run_sim builds the LaneChangeController.
LANE_CHANGE_TRIGGER_CLOSENESS = 0.18  # lead-car bbox height frac that prefers a change over slowing.
                                      # Must sit BELOW OBSTACLE_SLOW_CLOSENESS (0.22): the lead car
                                      # moves, so if SLOW engages first it caps the ego below the
                                      # lead's speed and the gap never closes to a higher trigger.
LANE_CHANGE_DISTANCE_M = 12.0         # longitudinal distance the lateral move plays out over
LANE_SETTLE_DISTANCE_M = 6.0          # re-centering distance before a new change can arm
TARGET_LANE_BLOCKED_CLOSENESS = 0.12  # a vehicle at least this tall in the target lane = not clear
PREFERRED_SIDE = "right"              # the left lane carries oncoming traffic on this course
# Maneuver execution (pure-pursuit lateral move; see run_sim.Vehicle).
LANE_CHANGE_LOOKAHEAD_M = 7.0         # look-ahead point distance into the target lane
LANE_CHANGE_YAW_RATE_DPS = 35.0       # max yaw rate during a change (sharper than lane-keeping)

# --- Simulation loop ------------------------------------------------------
SIM_FPS = 20.0              # fixed timestep; main.py assumes ~20 fps on webcam


def vertical_fov_deg() -> float:
    """Vertical FOV implied by the horizontal FOV and the frame aspect ratio."""
    import math
    hfov = math.radians(CAM_HFOV_DEG)
    vfov = 2 * math.atan(math.tan(hfov / 2) * (CAM_HEIGHT / CAM_WIDTH))
    return math.degrees(vfov)
