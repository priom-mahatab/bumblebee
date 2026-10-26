# 3D simulator (Ursina)

A textured 3D driving world to exercise the **whole** pipeline before hardware
arrives: the vehicle's camera renders what it sees → `vision/lane_detection` +
`detection/infer` (YOLO) + `control/state_machine` → the resulting `Action`
steers the car. It also emits **ground-truth** labels per frame so detection and
control can be scored, not just eyeballed.

This complements `sim/worlds/track.wbt` (the static Webots scene): it uses the
same real-metre scale and layout conventions, but adds turns, moving traffic,
a driven camera, and an eval dataset.

## Install

```bash
source .cv/bin/activate
pip install -r requirements.txt -r requirements-sim.txt   # adds ursina + pillow
```

## Run

```bash
PYTHONPATH=. python -m sim.run_sim              # self-drive, windowed
PYTHONPATH=. python -m sim.run_sim --scripted   # follow the centreline (clean tour / GT dataset)
PYTHONPATH=. python -m sim.run_sim --no-detect   # lane + control only (skips YOLO/torch)
PYTHONPATH=. python -m sim.run_sim --headless    # no window (CI / batch eval)
```
Options: `--frames N`, `--out DIR`. Each run writes to `runs/sim/<timestamp>/`:

```
annotated.mp4     # lanes, detections (green), ground truth (cyan), HUD per frame
images/           # frame_00000.jpg … (the camera frames)
labels/           # frame_00000.txt … YOLO format: class cx cy w h (normalised)
classes.txt       # 0 stop sign, 1 car, 2 person, 3 truck, 4 bus
meta.jsonl        # per-frame offset / distance / action / det & gt counts
```

Component self-tests:
```bash
PYTHONPATH=. python -m sim.course      # course geometry + object placement
PYTHONPATH=. python -m sim.textures    # regenerate the sign textures
```

## How it plugs into the existing pipeline

`sim/sim_camera.py:SimCameraSource` implements the same `read()` / `release()`
contract as `vision/camera.py`'s `WebcamSource`, so it's a drop-in frame source.
`run_sim.py` is the closed loop; to instead feed sim frames through the existing
`main.py`, swap in `SimCameraSource()` where it builds `WebcamSource()`.

Beyond a camera it also provides, from the known 3D scene:
- `ground_truth()` → exact 2D boxes + labels for in-view targets (the eval truth),
- `nearest_obstacle_distance_cm()` → an ultrasonic-sensor stand-in feeding
  `decide_action(distance_cm=…)`.

## The world

- A **3-lane** road (`spec.NUM_LANES`, set 4 for four lanes) with solid white
  edge lines and dashed white lane dividers, built as a ribbon mesh that follows
  a centreline with three **turns** (right, left, tighter right).
- Two **stop signs** (real red-octagon STOP textures so YOLO fires on them) and
  two speed-limit signs (scene realism; not a detector class, same as the Webots
  note — COCO/BDD have no speed-limit class).
- **Traffic**: a parked car on the shoulder, a slower car ahead in the ego lane
  (the car should `SLOW` as it closes in), and an oncoming car in the far lane.

Coordinates: X-Z ground plane, +Y up; heading 0 points +Z, increasing toward +X.

## Specs and where to change them (`sim/spec.py`)

The sim is at **real-road scale** (metres) on purpose: YOLO expects real road
scenes, so a road-scale world gives it realistic-looking frames. The camera's
**field of view is the one real spec that transfers unchanged** — set to the
Raspberry Pi Camera v2's 62.2°. Values marked `ASSUMPTION` (mount height, speed,
etc.) are road-scale stand-ins to reconcile with `hardware/` once the bill of
materials exists; they're all module-level constants, like the control
thresholds in `control/state_machine.py`.

## Known limitations / honesty notes

- **macOS shader note:** ursina's built-in shaders are GLSL 130/140, which macOS
  can't compile (it offers only GLSL 120 or 150+), so they render **black**. The
  sim ships its own GLSL-120 unlit shader (`sim/shaders.py`) and sets it as the
  default to avoid this — which is also why no entity passes `unlit=True` (that
  would re-attach ursina's incompatible shader). Rendering, capture, projection,
  the control loop and the dataset output are all verified working; still worth a
  glance on first run that lane detection locks onto the painted lines.
- **Cars are blocky low-poly models.** The stop-sign texture is realistic enough
  for the COCO/BDD detector; the cars may or may not be detected. That's a real
  finding to note — but the **ground-truth boxes are always exact**, so detection
  can still be scored regardless of whether the detector fires.
- Generated sign textures land in `sim/assets/` on first run (git-ignored;
  regenerated if deleted). Drop in a photo at the same path to use a real image.
```
