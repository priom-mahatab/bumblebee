"""Closed-loop rover driver on the Pi: camera -> lane + colour-obstacle detection -> control -> motors."""
import argparse
import time

from control.motor_control import (DRIVE_TRIM, SPIN_PULSE_S, Motors, Ultrasonic, lane_change_drive, lane_keep_drive,
                                   trim_drive)
from control.state_machine import Action, DrivingController, LaneChangeController, RoverLaneChange
from runlog import RunLogger
from vision.camera import PiCameraSource
from vision.lane_detection import LANE_PROFILES, LaneTracker, analyze_lanes, forward_tape_cm
from vision.obstacles import detect_obstacles

CRUISE_SPEED_MPS = 0.2   # assumed forward speed; dead-reckoned odometry for logging and the stop-sign controller
LOOP_HZ = 15


def main():
    parser = argparse.ArgumentParser(description="Run the rover control loop with logging.")
    parser.add_argument("--no-frames", action="store_true", help="log telemetry only, don't save frames")
    parser.add_argument("--frame-every", type=int, default=1, help="save every Nth frame (default 1)")
    parser.add_argument("--roi", default="rover", choices=sorted(LANE_PROFILES), help="lane profile (default rover)")
    parser.add_argument("--trim", type=float, default=DRIVE_TRIM,
                        help=f"motor imbalance correction, left up / right down by this fraction (default {DRIVE_TRIM}; 0 disables)")
    parser.add_argument("--lane", choices=("left", "right"),
                        help="starting lane; enables sonar-triggered lane changes (off without it)")
    parser.add_argument("--test-change", type=float, metavar="SECONDS",
                        help="with --lane, start one lane change this many seconds in, without an obstacle")
    args = parser.parse_args()
    if args.test_change is not None and not args.lane:
        parser.error("--test-change needs --lane")

    cam = PiCameraSource()
    motors = Motors()
    sonar = Ultrasonic()
    lane_change = RoverLaneChange(args.lane) if args.lane else None
    controller = DrivingController(lane_change=LaneChangeController(trigger_closeness=None, trigger_distance_cm=None),
                                   lane_lost_action=Action.STOP)
    tracker = LaneTracker()
    logger = RunLogger(save_frames=not args.no_frames, frame_every=args.frame_every)
    period = 1.0 / LOOP_HZ
    start = time.monotonic()
    test_pending = args.test_change is not None
    pulsed = False
    try:
        while True:
            t0 = time.monotonic()
            frame = cam.read()
            if pulsed:
                frame = cam.read()
                pulsed = False
            if frame is None:
                continue
            h, w = frame.shape[:2]
            lanes = analyze_lanes(frame, roi=args.roi, tracker=tracker)
            detections = detect_obstacles(frame)
            distance_cm = sonar.read_cm()
            odometry_m = CRUISE_SPEED_MPS * (time.monotonic() - start)
            ray_cm = forward_tape_cm(frame, args.roi) if lane_change else None
            now = time.monotonic()

            step_before = lane_change.step if lane_change else None
            if test_pending and now - start >= args.test_change and step_before is None:
                lane_change.begin(now)
                test_pending = False
            change_action = lane_change.update(distance_cm, ray_cm, now) if lane_change else None
            action = change_action or controller.update(
                lane_offset=lanes.offset,
                detections=detections,
                distance_cm=distance_cm,
                frame_width=w,
                frame_height=h,
                odometry_m=odometry_m,
                left_lane_exists=lanes.left_lane_exists,
                right_lane_exists=lanes.right_lane_exists,
                now=now,
            )
            step = lane_change.step if lane_change else None
            if step_before == "spin_back" and step is None:
                tracker = LaneTracker()
            if action is Action.STOP:
                l, r = 0, 0
            elif step not in (None, "aborted"):
                l, r = lane_change_drive(lane_change.direction, step, lanes.offset)
            else:
                l, r = lane_keep_drive(action, lanes.offset)
            l, r = trim_drive(l, r, args.trim)
            motors.drive(l, r)
            if step in ("spin_out", "spin_back") and action is not Action.STOP:
                time.sleep(SPIN_PULSE_S)
                motors.stop()
                pulsed = True

            logger.log({
                "frame": logger.count,
                "t": round(time.monotonic() - start, 3),
                "offset": lanes.offset,
                "left_lane": lanes.left_lane_exists,
                "right_lane": lanes.right_lane_exists,
                "dist_cm": round(distance_cm, 1),
                "obstacles": len(detections),
                "ray_cm": None if ray_cm is None else round(ray_cm, 1),
                "action": action.value,
                "lane": lane_change.current_lane if lane_change else None,
                "lc_step": step,
                "drive_l": l,
                "drive_r": r,
                "odometry_m": round(odometry_m, 3),
            }, frame)
            print(f"[{logger.count:5d}] offset={lanes.offset} dist={distance_cm:5.1f}cm ray={ray_cm} "
                  f"step={step} -> {action.value:11s} drive=({l},{r})")

            dt = time.monotonic() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        pass
    finally:
        motors.close()
        sonar.close()
        cam.release()
        logger.close()


if __name__ == "__main__":
    main()
