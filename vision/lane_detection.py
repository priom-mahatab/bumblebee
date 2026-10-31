"""Turns a raw frame into a lane-center offset the control loop can steer on."""
import numpy as np
import cv2

def _fit_lane_line(segments, y_bottom, y_top):
    if not segments:
        return None
    xs, ys = [], []
    for x1, y1, x2, y2 in segments:
        xs += [x1, x2]
        ys += [y1, y2]
    m, b = np.polyfit(ys, xs, 1)          # x = m*y + b
    return (int(m * y_bottom + b), y_bottom), (int(m * y_top + b), y_top)


# ROI trapezoid as (x_frac, y_frac) corners. The ROI is CAMERA-SPECIFIC:
# this default suits real dashcam clips; the sim has different geometry and
# passes its own trapezoid (see SIM_ROI in sim/run_sim.py).
REAL_ROI = [(0.10, 0.82), (0.95, 0.82), (0.58, 0.60), (0.42, 0.60)]

# Half the lane width at the look-ahead sample point, as a fraction of frame
# width. Used to reconstruct the lane center from ONE line when the other leaves
# view mid-curve (measured ~0.215 in the sim; only used in the one-sided case).
LANE_HALF_WIDTH_FRAC = 0.215


def compute_lane_offset(frame: np.ndarray, roi=REAL_ROI, lookahead: float = 0.85) -> float | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    h, w = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[(int(fx * w), int(fy * h)) for fx, fy in roi]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked, 2, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
    if lines is None:
        return None

    left, right = [], []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.5:
            continue
        (left if slope < 0 else right).append((x1, y1, x2, y2))

    fys = [fy for _, fy in roi]
    y_bottom, y_top = int(max(fys) * h), int(min(fys) * h)
    left_fit = _fit_lane_line(left, y_bottom, y_top)
    right_fit = _fit_lane_line(right, y_bottom, y_top)
    if left_fit is None and right_fit is None:
        return None

    # Sample the lane center at a LOOK-AHEAD point along the fitted lines:
    # fit[0] is the bottom (right in front of the car), fit[1] is the far end
    # (top of ROI). lookahead in [0,1] interpolates between them. Looking ahead
    # lets the car steer INTO a curve instead of reacting only once the bend
    # reaches the bumper (where a curving road still looks centered).
    def sample_x(fit):
        return fit[0][0] + lookahead * (fit[1][0] - fit[0][0])

    # Lane center at the look-ahead point. Both lines -> their midpoint. Only one
    # (common mid-curve, when the inner line leaves the ROI) -> offset that line
    # by the known half-lane-width, so we still get a usable steering signal
    # instead of giving up (None -> the car would coast straight off the curve).
    half_w = LANE_HALF_WIDTH_FRAC * w
    if left_fit is not None and right_fit is not None:
        lane_center_x = (sample_x(left_fit) + sample_x(right_fit)) / 2
    elif left_fit is not None:
        lane_center_x = sample_x(left_fit) + half_w
    else:
        lane_center_x = sample_x(right_fit) - half_w

    frame_center_x = w / 2
    return (lane_center_x - frame_center_x) / frame_center_x


if __name__ == "__main__":
    from sim.mock_inputs import make_test_frame

    for shift in (-80, -20, 0, 20, 80):
        frame = make_test_frame(lane_offset_px=shift)
        offset = compute_lane_offset(frame)
        print(f"shift={shift:4d}px -> lane_offset={offset:+.3f}")
