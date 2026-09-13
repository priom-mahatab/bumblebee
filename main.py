"""Dry-run pipeline: webcam -> lane detection + object detection ->
control decision -> print. No motors involved — this exercises the full
software pipeline before hardware arrives. motor_control.py plugs in once
the Pi and motor driver are wired up.
"""
from control.state_machine import decide_action
from detection.infer import detect
from vision.camera import WebcamSource
from vision.lane_detection import compute_lane_offset


def main():
    cam = WebcamSource()
    print("Running dry-run pipeline (no motors). Press Ctrl+C to stop.")
    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            lane_offset = compute_lane_offset(frame)
            detections = detect(frame)
            action = decide_action(
                lane_offset=lane_offset or 0.0,
                detections=detections,
                distance_cm=None,  # no ultrasonic sensor yet
                frame_width=frame.shape[1],
            )
            print(f"lane_offset={lane_offset} detections={len(detections)} -> {action.value}")
    except KeyboardInterrupt:
        pass
    finally:
        cam.release()


if __name__ == "__main__":
    main()
