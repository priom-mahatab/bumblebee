"""Camera capture abstraction with a common read()/release() interface across sources."""
import cv2


class WebcamSource:
    """Laptop webcam capture, for prototyping without the Pi."""

    def __init__(self, device_index: int = 0):
        self.cap = cv2.VideoCapture(device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam at index {device_index}")

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


class VideoSource:
    """Reads frames from a video file; read() returns None at end-of-file."""

    def __init__(self, path: str):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


class PiCameraSource:
    """Raspberry Pi camera capture via picamera2, usable only on the Pi."""

    def __init__(self):
        from picamera2 import Picamera2  # Pi-only dependency, import lazily

        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration())
        self.picam2.start()

    def read(self):
        return self.picam2.capture_array()

    def release(self):
        self.picam2.stop()
