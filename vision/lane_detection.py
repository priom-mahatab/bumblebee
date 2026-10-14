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


def compute_lane_offset(frame: np.ndarray, roi_top_fraction: float = 0.55) -> float | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    h, w = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(0.10 * w), int(0.82 * h)),
        (int(0.95 * w), int(0.82 * h)),
        (int(0.58 * w), int(0.60 * h)),
        (int(0.42 * w), int(0.60 * h)),
    ]], dtype=np.int32)
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

    y_bottom, y_top = int(0.82 * h), int(0.60 * h)
    left_fit = _fit_lane_line(left, y_bottom, y_top)
    right_fit = _fit_lane_line(right, y_bottom, y_top)
    if left_fit is None or right_fit is None:
        return None

    left_x = left_fit[0][0]                # x at the bottom
    right_x = right_fit[0][0]
    lane_center_x = (left_x + right_x) / 2
    frame_center_x = w / 2
    return (lane_center_x - frame_center_x) / frame_center_x


if __name__ == "__main__":
    from sim.mock_inputs import make_test_frame

    for shift in (-80, -20, 0, 20, 80):
        frame = make_test_frame(lane_offset_px=shift)
        offset = compute_lane_offset(frame)
        print(f"shift={shift:4d}px -> lane_offset={offset:+.3f}")
