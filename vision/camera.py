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
