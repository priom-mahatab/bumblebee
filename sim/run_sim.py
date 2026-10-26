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

from control.state_machine import Action, StopSignController
from vision.lane_detection import compute_lane_offset

from . import course as course_mod
from . import spec
from .sim_camera import SimCameraSource

GT_CLASSES = ["stop sign", "car", "person", "truck", "bus"]
DT = 1.0 / spec.SIM_FPS


def annotate(frame, lane_offset, detections, gt_boxes, action, dist, idx):
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
    cv2.putText(frame, f"[{idx}] off {off_txt}  dist {dist_txt}  {action.value}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame


class Vehicle:
    """Minimal kinematics: the control Action sets speed and yaw rate."""

    def __init__(self, ego):
        self.ego = ego
        self.speed = spec.CRUISE_SPEED_MPS
        self.stop_timer = 0.0
        self.traveled = 0.0

    def apply(self, action: Action):
        yaw = 0.0
        if action == Action.STOP:
            if self.stop_timer <= 0:
                self.stop_timer = spec.STOP_HOLD_SECONDS
            self.speed = 0.0
        elif self.stop_timer > 0:
            self.speed = 0.0                           # still serving a stop
        elif action == Action.SLOW:
            self.speed = spec.SLOW_SPEED_MPS
        elif action == Action.TURN_LEFT:
            self.speed = spec.CRUISE_SPEED_MPS; yaw = -spec.TURN_YAW_RATE_DPS
        elif action == Action.TURN_RIGHT:
            self.speed = spec.CRUISE_SPEED_MPS; yaw = +spec.TURN_YAW_RATE_DPS
        else:
            self.speed = spec.CRUISE_SPEED_MPS

        if self.stop_timer > 0:
            self.stop_timer -= DT

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
    ap.add_argument("--out", default="runs/sim")
    args = ap.parse_args()

    detect = None
    if not args.no_detect:
        from detection.infer import detect  # lazy: avoids loading torch otherwise

    out_dir = os.path.join(args.out, datetime.now().strftime("%Y%m%d_%H%M%S"))
    img_dir, lbl_dir = os.path.join(out_dir, "images"), os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    with open(os.path.join(out_dir, "classes.txt"), "w") as f:
        f.write("\n".join(GT_CLASSES) + "\n")

    src = SimCameraSource(headless=args.headless)
    veh = Vehicle(src.ego)
    controller = StopSignController()  # holds stop-sign pause-and-go state across frames
    writer = None
    meta = open(os.path.join(out_dir, "meta.jsonl"), "w")
    print(f"course {src.course.length:.0f} m; writing to {out_dir}")

    try:
        for idx in range(args.frames):
            frame = src.read()
            if frame is None:
                print("render returned no frame (GL back-end can't draw here)")
                break
            h, w = frame.shape[:2]

            lane_offset = compute_lane_offset(frame)
            dist = src.nearest_obstacle_distance_cm()
            detections = detect(frame) if detect else []
            action = controller.update(lane_offset=lane_offset or 0.0,
                                       detections=detections, distance_cm=dist, frame_width=w)

            gt = src.ground_truth()
            cv2.imwrite(os.path.join(img_dir, f"frame_{idx:05d}.jpg"), frame)
            with open(os.path.join(lbl_dir, f"frame_{idx:05d}.txt"), "w") as lf:
                for name, (x1, y1, x2, y2) in gt:
                    cid = GT_CLASSES.index(name)
                    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                    bw, bh = (x2 - x1) / w, (y2 - y1) / h
                    lf.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            meta.write(json.dumps({"frame": idx, "offset": lane_offset,
                                   "dist_cm": dist, "action": action.value,
                                   "n_det": len(detections), "n_gt": len(gt)}) + "\n")

            annotated = annotate(frame, lane_offset, detections, gt, action, dist, idx)
            if writer is None:
                writer = cv2.VideoWriter(os.path.join(out_dir, "annotated.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"),
                                         spec.SIM_FPS, (w, h))
            writer.write(annotated)

            if args.scripted:
                veh.traveled += spec.CRUISE_SPEED_MPS * DT
                scripted_pose(src.ego, src.course, veh.traveled)
            else:
                veh.apply(action)
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
