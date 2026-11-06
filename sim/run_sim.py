"""Closed-loop runner for the 3D sim.

Each frame: render what the vehicle's camera sees -> lane detection + YOLO +
state machine -> feed the resulting Action back into the vehicle's steering, so
the car actually drives the simulated course responding to lanes, the stop
sign, and traffic. Writes an annotated video plus an eval-ready ground-truth
dataset (images + YOLO-format labels + per-frame log).

    PYTHONPATH=. .cv/bin/python -m sim.run_sim                 # self-drive, windowed
    PYTHONPATH=. .cv/bin/python -m sim.run_sim --headless      # no window (CI / eval)
    PYTHONPATH=. .cv/bin/python -m sim.run_sim --scripted      # follow centreline (clean tour / GT)
    PYTHONPATH=. .cv/bin/python -m sim.run_sim --no-detect     # lane + control only (skip YOLO/torch)

Options: --frames N, --out DIR.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import cv2

from control.state_machine import Action, StopSignController, SpeedLimitTracker
from vision.lane_detection import compute_lane_offset
from vision.speed_limit import read_speed_limit

from . import course as course_mod
from . import spec
from .sim_camera import SimCameraSource
from control.state_machine import Detection

GT_CLASSES = ["stop sign", "car", "person", "truck", "bus", "speed limit"]
DT = 1.0 / spec.SIM_FPS

# Lane-detection ROI tuned to the SIM camera geometry (wider/higher than the
# real-dashcam default, which misses the sim's lane lines). (x_frac, y_frac).
SIM_ROI = [(0.0, 0.88), (1.0, 0.88), (0.80, 0.48), (0.20, 0.48)]


def gt_to_detections(gt):
    """Sim ground-truth tuples -> Detection objects for the control layer.
    Confidence is 1.0: GT is exact, not estimated. This is the perception
    source that DRIVES control in the sim (YOLO stays an optional overlay)."""
    return [Detection(name, 1.0, bbox) for name, bbox in gt]


def annotate(frame, lane_offset, detections, gt_boxes, action, dist, idx, speed_limit=None):
    h, w = frame.shape[:2]
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
    if lane_offset is not None:
        lx = int((w / 2) * (1 + lane_offset))
        cv2.circle(frame, (lx, int(0.9 * h)), 8, (0, 255, 255), -1)
    for name, (x1, y1, x2, y2) in gt_boxes:            # ground truth: thin cyan
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 1)
    for d in detections:                               # detections: green
        x1, y1, x2, y2 = map(int, d.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{d.class_name} {d.confidence:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    off_txt = f"{lane_offset:+.3f}" if lane_offset is not None else "none"
    dist_txt = f"{dist:.0f}cm" if dist is not None else "--"
    lim_txt = f"{speed_limit}" if speed_limit is not None else "--"
    cv2.putText(frame, f"[{idx}] off {off_txt}  dist {dist_txt}  lim {lim_txt}  {action.value}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame

def gt_to_detections(gt):
    """Sim ground-truth tuples -> Detection objects for the control layer.
    Confidence is 1.0: GT is exact, not estimated."""
    return [Detection(class_name, 1.0, bbox) for class_name, bbox in gt]

class Vehicle:
    """Minimal kinematics: the control Action sets speed and yaw rate."""

    def __init__(self, ego):
        self.ego = ego
        self.speed = spec.CRUISE_SPEED_MPS
        self.stop_timer = 0.0
        self.traveled = 0.0

    def apply(self, action: Action, lane_offset: float | None = None,
              speed_limit: int | None = None):
        # Cruise speed obeys the recognized speed limit when we have one, else the
        # default. (Mapping lives in spec; speed_limit is the READ number in mph.)
        cruise = (spec.CRUISE_SPEED_MPS if speed_limit is None
                  else speed_limit * spec.SPEED_LIMIT_MPS_PER_MPH)
        # --- speed comes from the discrete action (stop / slow / cruise) ---
        if action == Action.STOP:
            if self.stop_timer <= 0:
                self.stop_timer = spec.STOP_HOLD_SECONDS
            self.speed = 0.0
        elif self.stop_timer > 0:
            self.speed = 0.0                           # still serving a stop
        elif action == Action.SLOW:
            self.speed = spec.SLOW_SPEED_MPS
        else:                                          # FORWARD / TURN_* all cruise
            self.speed = cruise

        if self.stop_timer > 0:
            self.stop_timer -= DT

        # --- steering is PROPORTIONAL to the raw lane offset (P-controller) ---
        # yaw_rate = gain * offset, clamped to the max turn rate. This follows a
        # curve smoothly instead of the old on/off TURN_LEFT/RIGHT (which under-
        # steered through the deadzone). None (lane lost) -> hold heading straight.
        yaw = 0.0
        if lane_offset is not None and self.speed > 0:
            yaw = spec.STEER_GAIN_DPS * lane_offset
            yaw = max(-spec.TURN_YAW_RATE_DPS, min(spec.TURN_YAW_RATE_DPS, yaw))

        self.ego.rotation_y += yaw * DT
        dx, dz = course_mod._dir(self.ego.rotation_y)
        self.ego.x += dx * self.speed * DT
        self.ego.z += dz * self.speed * DT
        self.traveled += self.speed * DT


def scripted_pose(ego, course, dist):
    """Drop the vehicle exactly on the centreline at `dist` (clean tour / GT)."""
    x, z, h = course.sample(dist)
    ego.x, ego.z, ego.rotation_y = x, z, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="no window (CI/eval)")
    ap.add_argument("--scripted", action="store_true", help="follow the centreline")
    ap.add_argument("--no-detect", action="store_true", help="skip YOLO (lane+control only)")
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--seconds", type=float, default=None,
                    help="run for this many sim-seconds; sizes a long road to match")
    ap.add_argument("--out", default="runs/sim")
    args = ap.parse_args()

    # --seconds overrides the frame count and builds a road long enough that the
    # ego won't reach the end before time's up (1.3x cruise distance + margin).
    n_frames = int(args.seconds * spec.SIM_FPS) if args.seconds else args.frames
    min_length = (spec.CRUISE_SPEED_MPS * args.seconds * 1.3 + 60) if args.seconds else None

    detect = None
    if not args.no_detect:
        from detection.infer import detect  # lazy: avoids loading torch otherwise

    out_dir = os.path.join(args.out, datetime.now().strftime("%Y%m%d_%H%M%S"))
    img_dir, lbl_dir = os.path.join(out_dir, "images"), os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    with open(os.path.join(out_dir, "classes.txt"), "w") as f:
        f.write("\n".join(GT_CLASSES) + "\n")

    src = SimCameraSource(headless=args.headless, min_length=min_length)
    veh = Vehicle(src.ego)
    controller = StopSignController()  # holds stop-sign pause-and-go state across frames
    speed_tracker = SpeedLimitTracker()  # holds the active speed limit across frames
    writer = None
    meta = open(os.path.join(out_dir, "meta.jsonl"), "w")
    print(f"course {src.course.length:.0f} m; writing to {out_dir}")

    try:
        for idx in range(n_frames):
            frame = src.read()
            if frame is None:
                print("render returned no frame (GL back-end can't draw here)")
                break
            h, w = frame.shape[:2]

            lane_offset = compute_lane_offset(frame, roi=SIM_ROI)
            dist = src.nearest_obstacle_distance_cm()
            detections = detect(frame) if detect else []   # YOLO: optional overlay only
            gt = src.ground_truth()
            sim_detections = gt_to_detections(gt)          # GT drives control in the sim
            # Read the speed limit off the nearest sign's PIXELS (classical CV), not
            # from GT; the tracker's vote keeps a misread from jerking the speed.
            sl_boxes = [bb for name, bb in gt if name == "speed limit"]
            reading = (read_speed_limit(frame, max(sl_boxes, key=lambda b: b[3] - b[1]))
                       if sl_boxes else None)
            speed_limit = speed_tracker.update(reading)
            action = controller.update(lane_offset=lane_offset,
                                       detections=sim_detections, distance_cm=dist,
                                       frame_width=w, frame_height=h,
                                       odometry_m=veh.traveled,
                                       stop_sign_distance_m=src.next_stop_sign_distance_m())
            cv2.imwrite(os.path.join(img_dir, f"frame_{idx:05d}.jpg"), frame)
            with open(os.path.join(lbl_dir, f"frame_{idx:05d}.txt"), "w") as lf:
                for name, (x1, y1, x2, y2) in gt:
                    cid = GT_CLASSES.index(name)
                    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                    bw, bh = (x2 - x1) / w, (y2 - y1) / h
                    lf.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            meta.write(json.dumps({"frame": idx, "offset": lane_offset,
                                   "dist_cm": dist, "action": action.value,
                                   "speed_limit": speed_limit, "sl_read": reading,
                                   "n_det": len(detections), "n_gt": len(gt)}) + "\n")

            annotated = annotate(frame, lane_offset, detections, gt, action, dist, idx,
                                 speed_limit=speed_limit)
            if writer is None:
                writer = cv2.VideoWriter(os.path.join(out_dir, "annotated.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         spec.SIM_FPS, (w, h))
            writer.write(annotated)

            if args.scripted:
                veh.traveled += spec.CRUISE_SPEED_MPS * DT
                scripted_pose(src.ego, src.course, veh.traveled)
            else:
                veh.apply(action, lane_offset, speed_limit=speed_limit)
            src.step_world(DT)

            if idx % 20 == 0:
                print(f"[{idx:4d}] off={lane_offset} dist={dist} det={len(detections)} "
                      f"gt={len(gt)} -> {action.value}  (travelled {veh.traveled:.0f} m)")
            if veh.traveled > src.course.length - 6:
                print("reached end of course"); break
    finally:
        meta.close()
        if writer is not None:
            writer.release()
        src.release()
        print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
