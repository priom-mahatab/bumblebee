"""Camera, rover, and world constants for the 3D simulator (module-level, like control/state_machine.py); the world is real-road scale in metres, and values marked ASSUMPTION are road-scale stand-ins to reconcile with hardware later."""

CAM_WIDTH = 640
CAM_HEIGHT = 480

CAM_HFOV_DEG = 62.2         # Pi Camera Module v2; ASSUMPTION until confirmed

CAM_HEIGHT_M = 1.2          # ASSUMPTION: road-scale dash height
CAM_PITCH_DEG = 8.0
CAM_FORWARD_OFFSET_M = 1.5

CRUISE_SPEED_MPS = 4.0      # ASSUMPTION: scaled for road size
SLOW_SPEED_MPS = 1.5
TURN_YAW_RATE_DPS = 17.5    # deg/sec, max clamp
STEER_GAIN_DPS = 100.0      # scale with CRUISE/SLOW/CLAMP to change speed without changing the path
STOP_HOLD_SECONDS = 1.5
SPEED_LIMIT_MPS_PER_MPH = CRUISE_SPEED_MPS / 35.0

LANE_WIDTH_M = 3.5
NUM_LANES = 3
EGO_LANE_INDEX = 1          # 0 = leftmost
EDGE_LINE_WIDTH_M = 0.15
DASH_LENGTH_M = 3.0
DASH_GAP_M = 3.0

# Road-scale tunables that override state_machine's small-rover defaults.
LANE_CHANGE_TRIGGER_CLOSENESS = 0.18  # must stay below OBSTACLE_SLOW_CLOSENESS or SLOW caps the ego and the gap never closes
LANE_CHANGE_DISTANCE_M = 12.0
LANE_SETTLE_DISTANCE_M = 6.0
TARGET_LANE_BLOCKED_CLOSENESS = 0.12  # bbox height fraction
PREFERRED_SIDE = "right"              # left lane is oncoming here
LANE_CHANGE_LOOKAHEAD_M = 7.0
LANE_CHANGE_YAW_RATE_DPS = 35.0       # deg/sec

SIM_FPS = 20.0              # main.py assumes ~20 fps


def vertical_fov_deg() -> float:
    """Vertical FOV implied by the horizontal FOV and the frame aspect ratio."""
    import math
    hfov = math.radians(CAM_HFOV_DEG)
    vfov = 2 * math.atan(math.tan(hfov / 2) * (CAM_HEIGHT / CAM_WIDTH))
    return math.degrees(vfov)
