"""Closed-loop rover driver on the Pi: camera -> lane + colour-obstacle detection -> control -> motors."""
import argparse
import time

from control.motor_control import DRIVE_TRIM, Motors, Ultrasonic, lane_keep_drive, s_curve_drive, trim_drive
from control.state_machine import (ROVER_CAMERA_TRIGGER_CM, ROVER_IN_PATH_HALF_WIDTH_CM, ROVER_SONAR_CONFIRM_CM,
                                   S_CURVE_SECONDS, Action, DrivingController, LaneChangeController,
                                   LaneChangePhase)
from runlog import RunLogger
from vision.camera import PiCameraSource
from vision.lane_detection import LANE_PROFILES, LaneTracker, analyze_lanes
from vision.obstacles import detect_obstacles_with_distance

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
                        help="starting lane; enables camera-triggered, sonar-confirmed lane changes (off without it)")
    parser.add_argument("--s-curve", type=float, nargs=3, default=S_CURVE_SECONDS, metavar=("OUT", "STRAIGHT", "BACK"),
                        help=f"lane-change S-curve segment durations in seconds (default {' '.join(map(str, S_CURVE_SECONDS))})")
    parser.add_argument("--test-change", type=float, metavar="SECONDS",
                        help="with --lane, start one lane change this many seconds in, without an obstacle (S-curve tuning)")
    args = parser.parse_args()
    if args.test_change is not None and not args.lane:
        parser.error("--test-change needs --lane")

    cam = PiCameraSource()
    motors = Motors()
    sonar = Ultrasonic()
    lane_change = LaneChangeController(
        trigger_closeness=None,
        trigger_distance_cm=None,
        camera_trigger_cm=ROVER_CAMERA_TRIGGER_CM if args.lane else None,
        sonar_confirm_cm=ROVER_SONAR_CONFIRM_CM,
        current_lane=args.lane,
        s_curve=tuple(args.s_curve))
    controller = DrivingController(lane_change=lane_change, lane_lost_action=Action.STOP)
    tracker = LaneTracker()
    logger = RunLogger(save_frames=not args.no_frames, frame_every=args.frame_every)
    period = 1.0 / LOOP_HZ
    start = time.monotonic()
    test_pending = args.test_change is not None
    try:
        while True:
            t0 = time.monotonic()
            frame = cam.read()
            if frame is None:
                continue
            h, w = frame.shape[:2]
            lanes = analyze_lanes(frame, roi=args.roi, tracker=tracker)
            detections, obstacle_cm = detect_obstacles_with_distance(frame, ROVER_IN_PATH_HALF_WIDTH_CM)
            distance_cm = sonar.read_cm()
            odometry_m = CRUISE_SPEED_MPS * (time.monotonic() - start)

            now = time.monotonic()
            if (test_pending and now - start >= args.test_change and lanes.offset is not None
                    and lane_change.phase is LaneChangePhase.IN_LANE):
                lane_change.begin("right" if lane_change.current_lane == "left" else "left", odometry_m, now)
                test_pending = False
            phase_before = lane_change.phase
            action = controller.update(
                lane_offset=lanes.offset,
                detections=detections,
                distance_cm=distance_cm,
                frame_width=w,
                frame_height=h,
                odometry_m=odometry_m,
                left_lane_exists=lanes.left_lane_exists,
                right_lane_exists=lanes.right_lane_exists,
                obstacle_cm=obstacle_cm,
                now=now,
            )
            if phase_before is LaneChangePhase.CHANGING and lane_change.phase is LaneChangePhase.IN_LANE:
                tracker = LaneTracker()
            if action in (Action.CHANGE_LEFT, Action.CHANGE_RIGHT):
                l, r = s_curve_drive(lane_change.direction, lane_change.segment)
            else:
                l, r = lane_keep_drive(action, lanes.offset)
            l, r = trim_drive(l, r, args.trim)
            motors.drive(l, r)

            logger.log({
                "frame": logger.count,
                "t": round(time.monotonic() - start, 3),
                "offset": lanes.offset,
                "heading": None if lanes.heading is None else round(lanes.heading, 3),
                "centre_cm": None if lanes.centre_cm is None else round(lanes.centre_cm, 1),
                "left_lane": lanes.left_lane_exists,
                "right_lane": lanes.right_lane_exists,
                "dist_cm": round(distance_cm, 1),
                "obstacles": len(detections),
                "obstacle_cm": None if obstacle_cm is None else round(obstacle_cm, 1),
                "action": action.value,
                "lane": lane_change.current_lane,
                "lc_phase": lane_change.phase.value,
                "lc_segment": lane_change.segment,
                "drive_l": l,
                "drive_r": r,
                "odometry_m": round(odometry_m, 3),
            }, frame)
            print(f"[{logger.count:5d}] offset={lanes.offset} dist={distance_cm:5.1f}cm "
                  f"obstacles={len(detections)} cam={obstacle_cm} -> {action.value:11s} drive=({l},{r})")

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
