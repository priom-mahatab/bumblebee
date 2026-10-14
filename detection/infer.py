from ultralytics import YOLO

from control.state_machine import Detection

_model = None


def load_model():
    global _model
    if _model is None:
        _model = YOLO("detection/bdd_v3_best.pt")  
    return _model


def detect(frame, confidence: float = 0.4) -> list[Detection]:
    """Runs YOLO on a single BGR frame, returns a list of Detection."""
    model = load_model()
    results = model(frame, conf=confidence, verbose=False)[0]

    detections = []
    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(Detection(class_name, conf, (x1, y1, x2, y2)))
    return detections


if __name__ == "__main__":
    from vision.camera import WebcamSource

    cam = WebcamSource()
    print("Running live detection on webcam. Press Ctrl+C to stop.")
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            for d in detect(frame):
                print(f"{d.class_name:15s} conf={d.confidence:.2f} bbox={tuple(round(v) for v in d.bbox)}")
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()
