import cv2
import numpy as np

frame = cv2.imread("datasets/test_videos/crosscheck/f799_sharp-right-curve_went-forward.jpg")

hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
white = cv2.inRange(hls, (0, 200, 0), (179, 255, 255))
yellow = cv2.inRange(hls, (15, 30, 115), (35, 255, 255))
color_mask = cv2.bitwise_or(white, yellow)
gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur  = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

combined = cv2.bitwise_and(edges, color_mask)

h, w = combined.shape
mask = np.zeros_like(combined)
polygon = np.array([[
        (int(0.10 * w), int(0.82 * h)),
        (int(0.95 * w), int(0.82 * h)),
        (int(0.58 * w), int(0.60 * h)),
        (int(0.42 * w), int(0.60 * h)),
    ]], dtype=np.int32)
cv2.fillPoly(mask, polygon, 255)
roi = cv2.bitwise_and(combined, mask)

cv2.imwrite("color_mask_right_curve_roi.jpg", roi)