"""Closed-loop runner for the 3D sim: render camera -> lane + YOLO + state machine -> drive the vehicle, writing an annotated video and an eval-ready ground-truth dataset."""
from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime

import cv2

from control.state_machine import (Action, DrivingController, LaneChangeController,
                                    lead_vehicle_closeness)
from vision.lane_detection import analyze_lanes

from . import course as course_mod
from . import spec
from .sim_camera import SimCameraSource
from control.state_machine import Detection

GT_CLASSES = ["stop sign", "car", "person", "truck", "bus", "speed limit"]
DT = 1.0 / spec.SIM_FPS

# ROI tuned to the sim camera geometry; the real-dashcam default misses the sim's lane lines. (x_frac, y_frac)
SIM_ROI = [(0.0, 0.88), (1.0, 0.88), (0.80, 0.48), (0.20, 0.48)]


def gt_to_detections(gt):
    """Sim ground-truth tuples -> Detection objects (confidence 1.0; GT is exact and drives control, YOLO stays an overlay)."""
    return [Detection(name, 1.0, bbox) for name, bbox in gt]


def annotate(frame, lane_offset, detections, gt_boxes, action, dist, idx):
    h, w = frame.shape[:2]
    cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
    if lane_offset is not None:
        lx = int((w / 2) * (1 + lane_offset))
        cv2.circle(frame, (lx, int(0.9 * h)), 8, (0, 255, 255), -1)
    for name, (x1, y1, x2, y2) in gt_boxes:
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 1)
    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{d.class_name} {d.confidence:.2f}", (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    off_txt = f"{lane_offset:+.3f}" if lane_offset is not None else "none"
    dist_txt = f"{dist:.0f}cm" if dist is not None else "--"
    cv2.putText(frame, f"[{idx}] off {off_txt}  dist {dist_txt}  {action.value}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame

def gt_to_detections(gt):
    """Sim ground-truth tuples -> Detection objects (confidence 1.0; GT is exact)."""
    return [Detection(class_name, 1.0, bbox) for class_name, bbox in gt]

def _ego_lane_center() -> float:
    """Road-frame lateral (m, right +) of the ego's starting lane centre."""
    return (spec.EGO_LANE_INDEX - (spec.NUM_LANES - 1) / 2) * spec.LANE_WIDTH_M


def _clamp_lane(lateral: float) -> float:
    """Clamp a lane-centre lateral to the outermost lanes (a safety net)."""
    half = (spec.NUM_LANES - 1) / 2 * spec.LANE_WIDTH_M
    return max(-half, min(half, lateral))


class Vehicle:
    """Minimal kinematics: the Action sets speed, and steering is a P-controller on the lane offset except during a committed pure-pursuit lane change."""

    def __init__(self, ego, course):
        self.ego = ego
        self.course = course
        self.speed = spec.CRUISE_SPEED_MPS
        self.stop_timer = 0.0
        self.traveled = 0.0
        self.lane_lateral = _ego_lane_center()
        self.changing = False

    def _steer_to_lane(self, target_lateral: float):
        """Pure pursuit: aim at a look-ahead point in the target lane and turn toward it (clamped)."""
        tx, tz = self.course.place(self.traveled + spec.LANE_CHANGE_LOOKAHEAD_M, target_lateral)
        desired = math.degrees(math.atan2(tx - self.ego.x, tz - self.ego.z))
        err = (desired - self.ego.rotation_y + 180) % 360 - 180
        step = spec.LANE_CHANGE_YAW_RATE_DPS * DT
        self.ego.rotation_y += max(-step, min(step, err))

    def apply(self, action: Action, lane_offset: float | None = None):
        cruise = spec.CRUISE_SPEED_MPS
        if action == Action.STOP:
            if self.stop_timer <= 0:
                self.stop_timer = spec.STOP_HOLD_SECONDS
            self.speed = 0.0
        elif self.stop_timer > 0:
            self.speed = 0.0
        elif action == Action.SLOW:
            self.speed = spec.SLOW_SPEED_MPS
        else:
            self.speed = cruise

        if self.stop_timer > 0:
            self.stop_timer -= DT

        if action in (Action.CHANGE_LEFT, Action.CHANGE_RIGHT):
            if not self.changing:
                self.changing = True
                shift = -spec.LANE_WIDTH_M if action == Action.CHANGE_LEFT else spec.LANE_WIDTH_M
                self.lane_lateral = _clamp_lane(self.lane_lateral + shift)
            if self.speed > 0:
                self._steer_to_lane(self.lane_lateral)
        else:
            self.changing = False
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

    n_frames = int(args.seconds * spec.SIM_FPS) if args.seconds else args.frames
    min_length = (spec.CRUISE_SPEED_MPS * args.seconds * 1.3 + 60) if args.seconds else None

    detect = None
    if not args.no_detect:
        from detection.infer import detect  # lazy import: avoids loading torch under --no-detect

    out_dir = os.path.join(args.out, datetime.now().strftime("%Y%m%d_%H%M%S"))
    img_dir, lbl_dir = os.path.join(out_dir, "images"), os.path.join(out_dir, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    with open(os.path.join(out_dir, "classes.txt"), "w") as f:
        f.write("\n".join(GT_CLASSES) + "\n")

    src = SimCameraSource(headless=args.headless, min_length=min_length)
    veh = Vehicle(src.ego, src.course)
    controller = DrivingController(LaneChangeController(
        trigger_closeness=spec.LANE_CHANGE_TRIGGER_CLOSENESS,
        change_distance_m=spec.LANE_CHANGE_DISTANCE_M,
        settle_distance_m=spec.LANE_SETTLE_DISTANCE_M,
        blocked_closeness=spec.TARGET_LANE_BLOCKED_CLOSENESS,
        preferred_side=spec.PREFERRED_SIDE,
    ))
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

            lane = analyze_lanes(frame, roi=SIM_ROI)
            lane_offset = lane.offset
            dist = src.nearest_obstacle_distance_cm()
            detections = detect(frame) if detect else []
            gt = src.ground_truth()
            sim_detections = gt_to_detections(gt)          # GT drives object detection in the sim
            # Adjacency uses GT lane occupancy (vision boundaries are too sparse at sim scale); offset steering stays vision-based.
            lane_idx = round(veh.lane_lateral / spec.LANE_WIDTH_M + (spec.NUM_LANES - 1) / 2)
            left_exists = lane_idx > 0
            right_exists = lane_idx < spec.NUM_LANES - 1
            action = controller.update(lane_offset=lane_offset,
                                       detections=sim_detections, distance_cm=dist,
                                       frame_width=w, frame_height=h,
                                       odometry_m=veh.traveled,
                                       stop_sign_distance_m=src.next_stop_sign_distance_m(),
                                       left_lane_exists=left_exists,
                                       right_lane_exists=right_exists)
            cv2.imwrite(os.path.join(img_dir, f"frame_{idx:05d}.jpg"), frame)
            with open(os.path.join(lbl_dir, f"frame_{idx:05d}.txt"), "w") as lf:
                for name, (x1, y1, x2, y2) in gt:
                    cid = GT_CLASSES.index(name)
                    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                    bw, bh = (x2 - x1) / w, (y2 - y1) / h
                    lf.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            closeness = lead_vehicle_closeness(sim_detections, w, h)
            meta.write(json.dumps({"frame": idx, "offset": lane_offset,
                                   "dist_cm": dist, "action": action.value,
                                   "left_lane": lane.left_lane_exists,   # vision (logged for comparison)
                                   "right_lane": lane.right_lane_exists,  # vision (logged for comparison)
                                   "gt_left": left_exists, "gt_right": right_exists,  # GT (actually used)
                                   "lc_phase": controller.lane_change.phase.value,
                                   "closeness": round(closeness, 3) if closeness is not None else None,
                                   "lane_lateral": round(veh.lane_lateral, 2),
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
                veh.apply(action, lane_offset)
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
