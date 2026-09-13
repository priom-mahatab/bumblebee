# Mini Autonomous Vehicle Using Computer Vision

COSC 4358 project: a small autonomous vehicle that combines classical lane
detection with a YOLO object detector to navigate a controlled track,
responding to lanes, obstacles, and stop signs.

## Status

Hardware (Raspberry Pi, camera, motors, ultrasonic sensor) is on order.
Until it arrives, development is hardware-independent: control logic,
lane-detection algorithm, and object detection all run and are testable
on a laptop.

## Setup

```bash
python3.12 -m venv .cv
source .cv/bin/activate
pip install -r requirements.txt
```

`requirements-pi.txt` lists the additional packages needed once code runs
on the Raspberry Pi itself (installed there, on top of `requirements.txt`).

## Running things now (no hardware needed)

Quick self-tests for the hardware-independent pieces:

```bash
python -m control.state_machine   # decision logic against scripted scenarios
python -m sim.mock_inputs         # fake camera frame + distance readings
python -m vision.lane_detection   # lane-offset calc against a synthetic frame
```

Full dry-run pipeline using your laptop's webcam (lane detection + object
detection + control decision, printed each frame, no motors):

```bash
python main.py
```

Live object detection only, printed per frame:

```bash
python -m detection.infer
```

Object detection uses pretrained YOLOv8n (COCO weights) — no training
required, since COCO already covers person/car/truck/bus/stop sign.
Fine-tuning on BDD100K is a separate, later experiment (comparing against
this pretrained baseline for the graded mAP/IoU analysis), not a
dependency of the live demo.

## Layout

- `hardware/` — bill of materials, pinout, wiring notes (no code)
- `vision/` — camera capture, preprocessing, lane detection
- `detection/` — YOLO inference now; BDD100K prep + fine-tuning later
- `control/` — `state_machine.py` (pure decision logic) and, once
  hardware arrives, `motor_control.py` (the only file that touches GPIO)
- `sim/` — mock frames/sensor readings for testing without hardware
- `eval/` — mAP/IoU/latency scripts and logged results
- `docs/` — final report and demo materials
- `main.py` — ties the pipeline together
