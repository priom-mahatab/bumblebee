# Testing & Launch Commands

Every runnable entry point in the repo. There's no pytest suite — the `python -m …`
modules below are the self-checks. Run all commands from the repo root with the
virtualenv active:

```bash
source .cv/bin/activate      # or prefix each command with .cv/bin/python
```

## Setup

```bash
python3.12 -m venv .cv                    # NOT Homebrew python3.14 (ML wheels unreliable)
source .cv/bin/activate
pip install -r requirements.txt           # core: vision, control, detection
pip install -r requirements-sim.txt       # + the 3D simulator (ursina)
pip install -r requirements-pi.txt        # on the Raspberry Pi only (adds picamera2, etc.)
```

## Logic checks (fast, no camera or hardware)

```bash
python -m control.state_machine    # decision ladder on scripted scenarios: stop sign, lane change, no-lane fallback
python -m vision.lane_detection    # analyze_lanes on synthetic multi-lane frames; prints "all passed"
python -m sim.mock_inputs          # synthetic frame + scripted ultrasonic readings
python -m sim.course               # print the sim course geometry (length, objects, ego start)
```

## Perception on real footage

```bash
python -m detection.lane_test      # lane + adjacency overlay on sample stills + project_video -> datasets/test_images/roads/lane_output/
python -m detection.infer          # live YOLOv8n object detection on the laptop webcam
python main.py path/to/road.mp4    # full pipeline (lanes + YOLO + decision) on a video -> runs/pipeline/
python main.py                     # same, on the live webcam
```

## 3D simulator (closed-loop driving)

```bash
python -m sim.run_sim --frames 900 --no-detect                          # windowed self-drive
python -m sim.run_sim --headless --no-detect --frames 900 --out runs/sim # render to runs/sim/<timestamp>/annotated.mp4
```

Watch for: the ego steers the lanes, stops at the first stop sign (~48 m), then
changes into the right lane to pass the stalled car (~85 m). The annotated video
shows the per-frame action and lane offset.

Options:

| Flag | Effect |
|------|--------|
| `--no-detect` | Skip YOLO (overlay only; faster, no torch load) |
| `--headless` | No window (CI, or rendering straight to file) |
| `--scripted` | Follow the centreline exactly (clean tour / ground-truth capture) |
| `--frames N` | Number of frames (default 900) |
| `--seconds S` | Run for S sim-seconds (sizes a longer road to match) |
| `--out DIR` | Output directory (default `runs/sim`) |

## Detection metrics (ML evaluation)

```bash
python -m eval.evaluate            # recompute v3 + v4 mAP / precision / recall -> eval/*.csv
python -m eval.evaluate --models v4   # a single model
```

Results table and caveats: `eval/detection_metrics.md`.

## Utilities

```bash
python -m sim.textures             # (re)generate the stop / speed-limit sign textures
```
