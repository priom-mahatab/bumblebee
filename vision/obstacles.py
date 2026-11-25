"""Classical-CV obstacle detector: finds a distinctly-coloured object on the road and emits it as a Detection."""
import cv2
import numpy as np

from control.state_machine import Detection

OBSTACLE_CLASS_NAME = "obstacle"
# HSV range for the placed object's colour; default tuned for bright orange. Retune via __main__ for your object.
OBSTACLE_HSV_LOW = (5, 120, 120)
OBSTACLE_HSV_HIGH = (25, 255, 255)
# Lower-frame band to search (fractions of height), full width so side lanes are included, not the lane trapezoid.
OBSTACLE_BAND_Y = (0.50, 0.95)
MIN_OBSTACLE_AREA_FRAC = 0.004  # ignore blobs smaller than this fraction of frame area
# Rover pinhole calibration from cone frames at 40-120 cm (2026-11-25); the box width is the base, reflection included.
CAMERA_FOCAL_PX = 898.0
CAMERA_CENTER_X_PX = 300.0
CAMERA_LENS_OFFSET_CM = 10.0
CONE_WIDTH_CM = 12.7


def _band_mask(shape, band_y):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[int(band_y[0] * h):int(band_y[1] * h), :] = 255
    return mask


def detect_obstacles(frame, hsv_low=OBSTACLE_HSV_LOW, hsv_high=OBSTACLE_HSV_HIGH,
                     band_y=OBSTACLE_BAND_Y, min_area_frac=MIN_OBSTACLE_AREA_FRAC) -> list[Detection]:
    """Colour-threshold the road band and return each large enough blob as a Detection box."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_high))
    mask = cv2.bitwise_and(mask, _band_mask(frame.shape, band_y))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    h, w = frame.shape[:2]
    min_area = min_area_frac * h * w
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        detections.append(Detection(OBSTACLE_CLASS_NAME, 1.0, (x, y, x + bw, y + bh)))
    return detections


def cone_position(detection: Detection) -> tuple[float, float]:
    """Distance ahead of the lens and lateral offset (cm, + right) of a cone from its box width and centre."""
    x1, _, x2, _ = detection.bbox
    depth = CAMERA_FOCAL_PX * CONE_WIDTH_CM / (x2 - x1)
    return depth - CAMERA_LENS_OFFSET_CM, ((x1 + x2) / 2 - CAMERA_CENTER_X_PX) * depth / CAMERA_FOCAL_PX


if __name__ == "__main__":
    from vision.camera import WebcamSource

    cam = WebcamSource()
    print("Live obstacle detection on webcam. Hold up your coloured object; Ctrl+C to stop.")
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            w = frame.shape[1]
            for d in detect_obstacles(frame):
                cx = (d.bbox[0] + d.bbox[2]) / 2 / w
                band = "left" if cx < 0.4 else "right" if cx > 0.6 else "ahead"
                print(f"obstacle bbox={tuple(round(v) for v in d.bbox)} x={cx:.2f} ({band})")
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
