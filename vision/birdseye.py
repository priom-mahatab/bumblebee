"""Rover camera to floor-plane mapping (cm, x right of the camera, z forward of the lens) from a one-time tape-rectangle calibration."""
import cv2
import numpy as np

# Calibration 2026-11-17 (calib.jpg): 24 x 39 cm tape rectangle, near edge 25 cm ahead, camera 5 cm high.
# Corners as NL, NR, FR, FL; near corners extrapolated just outside the 640x480 frame.
ROVER_CALIB_IMAGE_PTS = ((-64.1, 430.0), (656.7, 429.0), (444.1, 321.4), (156.2, 319.5))
ROVER_CALIB_FLOOR_CM = ((-12.0, 25.0), (12.0, 25.0), (12.0, 64.0), (-12.0, 64.0))

IMAGE_TO_FLOOR = cv2.getPerspectiveTransform(np.float32(ROVER_CALIB_IMAGE_PTS), np.float32(ROVER_CALIB_FLOOR_CM))
FLOOR_TO_IMAGE = np.linalg.inv(IMAGE_TO_FLOOR)

TOPDOWN_X_RANGE_CM = (-50.0, 50.0)
TOPDOWN_Z_RANGE_CM = (15.0, 130.0)
TOPDOWN_PX_PER_CM = 3.0


def image_to_floor(points: np.ndarray) -> np.ndarray:
    """Map Nx2 image pixel coordinates to Nx2 floor coordinates in cm."""
    return cv2.perspectiveTransform(np.float32(points).reshape(-1, 1, 2), IMAGE_TO_FLOOR).reshape(-1, 2)


def floor_to_image(points: np.ndarray) -> np.ndarray:
    """Map Nx2 floor coordinates in cm to Nx2 image pixel coordinates."""
    return cv2.perspectiveTransform(np.float32(points).reshape(-1, 1, 2), FLOOR_TO_IMAGE).reshape(-1, 2)


def _floor_to_topdown_matrix() -> np.ndarray:
    s = TOPDOWN_PX_PER_CM
    return np.array([[s, 0, -s * TOPDOWN_X_RANGE_CM[0]],
                     [0, -s, s * TOPDOWN_Z_RANGE_CM[1]],
                     [0, 0, 1]], dtype=np.float64)


def topdown_size() -> tuple[int, int]:
    """(width, height) in pixels of the top-down view."""
    s = TOPDOWN_PX_PER_CM
    return (int(s * (TOPDOWN_X_RANGE_CM[1] - TOPDOWN_X_RANGE_CM[0])),
            int(s * (TOPDOWN_Z_RANGE_CM[1] - TOPDOWN_Z_RANGE_CM[0])))


def floor_to_topdown(points: np.ndarray) -> np.ndarray:
    """Map Nx2 floor coordinates in cm to Nx2 pixel coordinates in the top-down view."""
    return cv2.perspectiveTransform(np.float32(points).reshape(-1, 1, 2), _floor_to_topdown_matrix()).reshape(-1, 2)


def warp_topdown(frame: np.ndarray) -> np.ndarray:
    """Warp a camera frame to the top-down floor view (far at the top, rover at the bottom centre)."""
    return cv2.warpPerspective(frame, _floor_to_topdown_matrix() @ IMAGE_TO_FLOOR, topdown_size())


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "calib.jpg"
    out = src.rsplit(".", 1)[0] + "_topdown.jpg"
    cv2.imwrite(out, warp_topdown(cv2.imread(src)))
    print(f"wrote {out}")
