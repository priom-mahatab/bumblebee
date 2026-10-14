"""Runs compute_lane_offset on the sample road images and saves a
visualization (fitted left/right lane lines + frame-center line) for each,
so you can eyeball how the detector does per image.

Run from the repo root:
    PYTHONPATH=. .cv/bin/python detection/lane_test.py
"""
import cv2
import numpy as np

from vision.lane_detection import compute_lane_offset, _fit_lane_line

ROADS = "datasets/test_images/roads"
OUT = f"{ROADS}/lane_output"
IMAGES = ["dashcam_lane.png", "lane_2.jpg", "lane_3.jpg"]


def visualize(frame):
    """Redraws the pipeline's fitted lane lines on the frame (debug only)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    h, w = edges.shape
    mask = np.zeros_like(edges)
    poly = np.array([[
        (int(0.10 * w), int(0.82 * h)), (int(0.95 * w), int(0.82 * h)),
        (int(0.58 * w), int(0.60 * h)), (int(0.42 * w), int(0.60 * h)),
    ]], np.int32)
    cv2.fillPoly(mask, poly, 255)
    masked = cv2.bitwise_and(edges, mask)
    lines = cv2.HoughLinesP(masked, 2, np.pi / 180, 50, minLineLength=40, maxLineGap=100)

    out = frame.copy()
    cv2.line(out, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
    if lines is not None:
        left, right = [], []
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if x2 == x1:
                continue
            s = (y2 - y1) / (x2 - x1)
            if abs(s) < 0.5:
                continue
            (left if s < 0 else right).append((x1, y1, x2, y2))
        yb, yt = int(0.82 * h), int(0.60 * h)
        for seg, color in [(left, (255, 0, 0)), (right, (0, 0, 255))]:
            fit = _fit_lane_line(seg, yb, yt)
            if fit:
                cv2.line(out, fit[0], fit[1], color, 5)
    return out


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    for name in IMAGES:
        frame = cv2.imread(f"{ROADS}/{name}")
        offset = compute_lane_offset(frame)
        print(f"{name:20s} -> offset = {offset}")
        stem = name.rsplit(".", 1)[0]
        cv2.imwrite(f"{OUT}/{stem}_result.jpg", visualize(frame))
    print(f"saved visualizations to {OUT}/")


if __name__ == "__main__":
    main()
