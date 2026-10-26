"""Full pipeline runner: frame source -> lane detection + object detection ->
control decision -> annotated output. No motors involved — this exercises the
whole software pipeline before hardware arrives. motor_control.py plugs in once
the Pi and motor driver are wired up.

Usage:
    # run on a recorded road video (open-loop sim stand-in):
    PYTHONPATH=. .cv/bin/python main.py path/to/road.mp4
    # or live laptop webcam (no argument):
    PYTHONPATH=. .cv/bin/python main.py

Writes an annotated video (lanes/detections/action drawn on each frame) to
runs/pipeline/annotated.mp4 so you can watch the pipeline's decisions.
"""
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2

from control.state_machine import StopSignController
from detection.infer import detect
from vision.camera import VideoSource, WebcamSource
from vision.lane_detection import compute_lane_offset

OUT_DIR = "runs/pipeline"


def annotate(frame, lane_offset, detections, action):
    """Draw the pipeline's view on the frame: detection boxes, the lane-center
    marker (reconstructed from the offset) vs. frame center, and a HUD line."""
    h, w = frame.shape[:2]

    # frame center (where the car is pointing) vs. lane center (where it wants to be)
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
    if lane_offset is not None:
        lane_x = int((w / 2) * (1 + lane_offset))
        cv2.circle(frame, (lane_x, int(0.9 * h)), 8, (0, 255, 255), -1)

    # object detections
    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{d.class_name} {d.confidence:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # HUD
    off_txt = f"{lane_offset:+.3f}" if lane_offset is not None else "none"
    cv2.putText(frame, f"offset: {off_txt}   action: {action.value}", (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return frame


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else None
    if video_path:
        source = VideoSource(video_path)
        fps = source.fps
        is_video = True
        print(f"Running pipeline on video: {video_path}")
    else:
        source = WebcamSource()
        fps = 20.0
        is_video = False
        print("Running pipeline on webcam. Press Ctrl+C to stop.")

    # unique output name per run: <source>_<timestamp>.mp4 (never overwrites)
    stem = Path(video_path).stem if video_path else "webcam"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUT_DIR, f"{stem}_{timestamp}.mp4")
    os.makedirs(OUT_DIR, exist_ok=True)
    writer = None
    frames = 0
    controller = StopSignController()  # holds stop-sign pause-and-go state across frames

    try:
        while True:
            frame = source.read()
            if frame is None:
                if is_video:
                    break          # end of file
                continue           # webcam dropped a frame

            lane_offset = compute_lane_offset(frame)
            detections = detect(frame)
            action = controller.update(
                lane_offset=lane_offset or 0.0,
                detections=detections,
                distance_cm=None,          # no ultrasonic sensor yet
                frame_width=frame.shape[1],
            )

            annotated = annotate(frame, lane_offset, detections, action)
            if writer is None:
                h, w = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )
            writer.write(annotated)

            frames += 1
            print(f"[{frames:5d}] offset={lane_offset} detections={len(detections)} -> {action.value}")
    except KeyboardInterrupt:
        pass
    finally:
        source.release()
        if writer is not None:
            writer.release()
            print(f"\nSaved annotated video ({frames} frames) to {out_path}")


if __name__ == "__main__":
    main()
