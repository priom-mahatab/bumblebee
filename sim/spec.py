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
CRUISE_SPEED_MPS = 8.0      # ASSUMPTION
SLOW_SPEED_MPS = 3.0
TURN_YAW_RATE_DPS = 35.0    # deg/sec of heading change on TURN_LEFT/RIGHT
STOP_HOLD_SECONDS = 1.5     # how long to sit at a stop sign before resuming

# --- Road geometry --------------------------------------------------------
LANE_WIDTH_M = 3.5          # matches track.wbt
NUM_LANES = 3               # "3/4 lane road" — set 4 for four lanes
EGO_LANE_INDEX = 1          # 0 = leftmost; ego drives in the centre lane
EDGE_LINE_WIDTH_M = 0.15    # painted line width (matches track.wbt RoadLine)
DASH_LENGTH_M = 3.0         # dashed divider: dash + gap pattern
DASH_GAP_M = 3.0

# --- Simulation loop ------------------------------------------------------
SIM_FPS = 20.0              # fixed timestep; main.py assumes ~20 fps on webcam


def vertical_fov_deg() -> float:
    """Vertical FOV implied by the horizontal FOV and the frame aspect ratio."""
    import math
    hfov = math.radians(CAM_HFOV_DEG)
    vfov = 2 * math.atan(math.tan(hfov / 2) * (CAM_HEIGHT / CAM_WIDTH))
    return math.degrees(vfov)
