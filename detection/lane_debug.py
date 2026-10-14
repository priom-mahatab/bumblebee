import cv2
from pathlib import Path
import numpy as np

img_path = Path("datasets/test_images/roads/dashcam_lane.png")

img = cv2.imread(str(img_path))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(blur, 50, 150)
cv2.imwrite("edges.jpg", edges)

h, w = edges.shape
mask = np.zeros_like(edges)
polygon = np.array([[
    (int(0.10*w), int(0.82*h)),   # bottom-left   (above the hood)
    (int(0.95*w), int(0.82*h)),   # bottom-right
    (int(0.58*w), int(0.60*h)),   # top-right  (near vanishing point)
    (int(0.42*w), int(0.60*h)),   # top-left
]], dtype=np.int32)

cv2.fillPoly(mask, polygon, 255)
masked = cv2.bitwise_and(edges, mask)
cv2.imwrite("masked_edges.jpg", masked)

lines = cv2.HoughLinesP(
    masked,
    rho=2,
    theta=np.pi/180,
    threshold=50,
    minLineLength=40,
    maxLineGap=100,
)

overlay = img.copy()
if lines is not None:
    for x1, y1, x2, y2 in lines[:, 0]:
        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
cv2.imwrite("hough.jpg", overlay)

left, right = [], []
for x1, y1, x2, y2 in lines[:, 0]:
    if x2 == x1:            # avoid divide-by-zero 
        continue
    slope = (y2 - y1) / (x2 - x1)
    if abs(slope) < 0.5:   # drop near-horizontal junk
        continue
    (left if slope < 0 else right).append((x1, y1, x2, y2))

split_img = img.copy()
for x1,y1,x2,y2 in left: cv2.line(split_img, (x1,y1), (x2,y2), (255,0,0), 3)
for x1,y1,x2,y2 in right: cv2.line(split_img, (x1,y1), (x2,y2), (0,0,255), 3)

cv2.imwrite("split.jpg", split_img)

def fit_lane_line(segments, y_bottom, y_top):
    if not segments:
        return None
    xs, ys = [], []
    for x1, y1, x2, y2 in segments:
        xs += [x1, x2]
        ys += [y1, y2]
    m, b = np.polyfit(ys, xs, 1)
    x_bottom = int(m * y_bottom + b)
    x_top = int(m * y_top + b)
    return (x_bottom, y_bottom), (x_top, y_top)

y_bottom = int(0.82 * h)
y_top = int(0.62 * h)

lanes_img = img.copy()
for seg, color in [(left, (255, 0, 0)), (right, (0, 0, 255))]:
    fit = fit_lane_line(seg, y_bottom, y_top)
    if fit:
        cv2.line(lanes_img, fit[0], fit[1], color, 5)
cv2.imwrite("lanes.jpg", lanes_img)

left_fit  = fit_lane_line(left,  y_bottom, y_top)
right_fit = fit_lane_line(right, y_bottom, y_top)

left_x  = left_fit[0][0]      # x at y_bottom
right_x  = right_fit[0][0]
lane_center_x = (left_x + right_x) / 2
frame_center_x = w / 2
offset = (lane_center_x - frame_center_x) / frame_center_x
print(f"lane_offset = {offset:+.3f}")

cv2.circle(lanes_img, (int(lane_center_x), y_bottom), 10, (0,255,255), -1)
cv2.line(lanes_img, (int(frame_center_x), 0), (int(frame_center_x), h), (255,255,255), 1)
cv2.imwrite("offset.jpg", lanes_img)