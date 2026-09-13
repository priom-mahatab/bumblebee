# av-comp-vis — Mini Autonomous Vehicle Using Computer Vision

COSC 4358 course project. A small autonomous vehicle (Raspberry Pi, camera,
motors, ultrasonic sensor) that combines classical lane detection with a
YOLO object detector to navigate a controlled track — responding to lanes,
obstacles, and stop signs. See `docs/` for the full proposal once it's
added; summary below is enough to work from day to day.

## How to work in this repo

**I write the code myself. Claude should not write or edit files unless I
explicitly ask it to.** Default mode: explain the approach/concepts for
whatever I'm working on, answer questions, and review or help debug code
I've written — don't generate files proactively, even for broadly-scoped
requests like "let's build X." If something concrete needs writing (like
this file), I'll ask for it directly.

## Status

Hardware (Pi, camera, motors, ultrasonic sensor) was ordered through my
office and takes 2-3 weeks to arrive (ordered ~2026-09-05). Until then,
work is scoped to be hardware-independent: control logic, lane-detection
algorithm, mock sensor inputs, and pretrained-model inference all run and
are testable on a laptop with just a webcam.

## Key technical decisions

- **Object detection uses pretrained YOLOv8n (COCO weights) on the actual
  car — no training required for the live demo.** COCO already includes
  `person`, `car`, `truck`, `bus`, and `stop sign`, which covers what the
  track needs.
- **BDD100K fine-tuning is a separate experiment**, not a dependency of
  the robot working: fine-tune YOLOv8n on BDD100K, evaluate mAP@0.50 /
  IoU-based precision-recall, and compare against the pretrained baseline
  and across model sizes. This is the graded ML-rigor deliverable the
  proposal asks for.
- Reason for the split: BDD100K has no dedicated `stop sign` class (only
  a generic `traffic sign`), so it wouldn't actually improve the one
  detection that matters most for the physical track's stop-sign
  behavior — COCO's pretrained weights are arguably better for that
  specific case.

## Setup

```bash
python3.12 -m venv .cv       # NOT the Homebrew python3.14 — ML wheels
                              # (torch, ultralytics) aren't reliable on it yet
source .cv/bin/activate
pip install -r requirements.txt
```

`requirements-pi.txt` lists the extra packages needed once code runs on
the Raspberry Pi itself (installed there, on top of `requirements.txt`).
`picamera2` in particular is not pip-installable on macOS — it goes in via
`apt` on the Pi, since it depends on system libcamera libraries.

## Running things now (no hardware needed)

```bash
python -m control.state_machine   # decision logic against scripted scenarios
python -m sim.mock_inputs         # fake camera frame + distance readings
python -m vision.lane_detection   # lane-offset calc against a synthetic frame
python -m detection.infer         # live YOLOv8n on the laptop webcam
python main.py                    # full dry-run: webcam -> lane + object
                                   # detection -> control decision, printed
                                   # each frame, no motors
```

## Layout

- `hardware/` — bill of materials, pinout, wiring notes (no code)
- `vision/` — camera capture, preprocessing, lane detection
- `detection/` — YOLO inference now; BDD100K prep + fine-tuning later
- `control/` — `state_machine.py` (pure decision logic, no hardware calls)
  and, once hardware arrives, `motor_control.py` (the only file that
  touches GPIO)
- `sim/` — mock frames/sensor readings for testing without hardware
- `eval/` — mAP/IoU/latency scripts and logged results
- `docs/` — final report and demo materials
- `main.py` — ties the pipeline together

## Conventions

- Keep hardware-touching code isolated to `control/motor_control.py` and
  `vision/camera.py`'s `PiCameraSource` — everything else should be
  testable without the physical robot.
- Tunable thresholds (stop/slow distances, lane deadzone, in-path bbox
  range) live as module-level constants in `control/state_machine.py`,
  not hardcoded inline.
