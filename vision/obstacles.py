"""Classical-CV obstacle detector: finds a distinctly-coloured object on the road and emits it as a Detection."""
import cv2
import numpy as np

from control.state_machine import Detection
from vision.birdseye import image_to_floor

OBSTACLE_CLASS_NAME = "obstacle"
# HSV range for the placed object's colour; default tuned for bright orange. Retune via __main__ for your object.
OBSTACLE_HSV_LOW = (5, 120, 120)
OBSTACLE_HSV_HIGH = (25, 255, 255)
# Lower-frame band to search (fractions of height), full width so side lanes are included, not the lane trapezoid.
OBSTACLE_BAND_Y = (0.50, 0.95)
MIN_OBSTACLE_AREA_FRAC = 0.004  # ignore blobs smaller than this fraction of frame area


def _band_mask(shape, band_y):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[int(band_y[0] * h):int(band_y[1] * h), :] = 255
    return mask


def _blobs(frame, hsv_low, hsv_high, band_y, min_area_frac):
    """Colour mask of the road band and its contours large enough to be an obstacle."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_high))
    mask = cv2.bitwise_and(mask, _band_mask(frame.shape, band_y))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    h, w = frame.shape[:2]
    min_area = min_area_frac * h * w
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return mask, [c for c in contours if cv2.contourArea(c) >= min_area]


def detect_obstacles(frame, hsv_low=OBSTACLE_HSV_LOW, hsv_high=OBSTACLE_HSV_HIGH,
                     band_y=OBSTACLE_BAND_Y, min_area_frac=MIN_OBSTACLE_AREA_FRAC) -> list[Detection]:
    """Colour-threshold the road band and return each large enough blob as a Detection box."""
    _, contours = _blobs(frame, hsv_low, hsv_high, band_y, min_area_frac)
    detections = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        detections.append(Detection(OBSTACLE_CLASS_NAME, 1.0, (x, y, x + bw, y + bh)))
    return detections


def nearest_obstacle_cm(frame, half_width_cm: float, hsv_low=OBSTACLE_HSV_LOW, hsv_high=OBSTACLE_HSV_HIGH,
                        band_y=OBSTACLE_BAND_Y, min_area_frac=MIN_OBSTACLE_AREA_FRAC) -> float | None:
    """Floor distance (cm) to the nearest obstacle base within half_width_cm of the rover's centreline, or None; the base is the blob's widest row, where the object meets the floor and its reflection on a shiny floor."""
    mask, contours = _blobs(frame, hsv_low, hsv_high, band_y, min_area_frac)
    nearest = None
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        blob = np.zeros_like(mask)
        cv2.drawContours(blob, [c], -1, 255, cv2.FILLED)
        blob = cv2.bitwise_and(blob, mask)[y:y + bh, x:x + bw]
        row = int(np.argmax((blob > 0).sum(axis=1)))
        cols = np.nonzero(blob[row])[0]
        floor_x, floor_z = image_to_floor(np.array([[x + cols.mean(), y + row]], dtype=np.float32))[0]
        if abs(floor_x) <= half_width_cm and (nearest is None or floor_z < nearest):
            nearest = float(floor_z)
    return nearest


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
