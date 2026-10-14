"""Camera capture abstraction — same read()/release() interface whether
frames come from a laptop webcam (now) or the Raspberry Pi camera (once
hardware arrives), so the rest of the pipeline doesn't care which."""
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
    """Reads frames from a video file — the open-loop sim stand-in for
    running the pipeline on recorded road footage. Same read()/release()
    interface as the camera sources, so main.py doesn't care it's a file.
    read() returns None at end-of-file."""

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
    """Raspberry Pi camera capture via picamera2. Only usable on the Pi
    itself — picamera2 isn't pip-installable on macOS, see requirements-pi.txt."""

    def __init__(self):
        from picamera2 import Picamera2  # imported lazily; Pi-only dependency

        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration())
        self.picam2.start()

    def read(self):
        return self.picam2.capture_array()

    def release(self):
        self.picam2.stop()
