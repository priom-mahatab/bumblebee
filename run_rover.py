"""Closed-loop rover driver on the Pi: camera -> lane + colour-obstacle detection -> control -> motors."""
import argparse
import time

from control.motor_control import Motors, Ultrasonic, lane_keep_drive
from control.state_machine import Action, DrivingController
from runlog import RunLogger
from vision.camera import PiCameraSource
from vision.lane_detection import LANE_PROFILES, LaneTracker, analyze_lanes
from vision.obstacles import detect_obstacles

CRUISE_SPEED_MPS = 0.2   # assumed forward speed; dead-reckons odometry for lane-change progress (no encoders)
LOOP_HZ = 15


def main():
    parser = argparse.ArgumentParser(description="Run the rover control loop with logging.")
    parser.add_argument("--no-frames", action="store_true", help="log telemetry only, don't save frames")
    parser.add_argument("--frame-every", type=int, default=1, help="save every Nth frame (default 1)")
    parser.add_argument("--roi", default="rover", choices=sorted(LANE_PROFILES), help="lane profile (default rover)")
    args = parser.parse_args()

    cam = PiCameraSource()
    motors = Motors()
    sonar = Ultrasonic()
    controller = DrivingController(lane_lost_action=Action.STOP)
    tracker = LaneTracker()
    logger = RunLogger(save_frames=not args.no_frames, frame_every=args.frame_every)
    period = 1.0 / LOOP_HZ
    start = time.monotonic()
    try:
        while True:
            t0 = time.monotonic()
            frame = cam.read()
            if frame is None:
                continue
            h, w = frame.shape[:2]
            lanes = analyze_lanes(frame, roi=args.roi, tracker=tracker)
            detections = detect_obstacles(frame)
            distance_cm = sonar.read_cm()
            odometry_m = CRUISE_SPEED_MPS * (time.monotonic() - start)

            action = controller.update(
                lane_offset=lanes.offset,
                detections=detections,
                distance_cm=distance_cm,
                frame_width=w,
                frame_height=h,
                odometry_m=odometry_m,
                left_lane_exists=lanes.left_lane_exists,
                right_lane_exists=lanes.right_lane_exists,
            )
            l, r = lane_keep_drive(action, lanes.offset)
            motors.drive(l, r)

            logger.log({
                "frame": logger.count,
                "t": round(time.monotonic() - start, 3),
                "offset": lanes.offset,
                "left_lane": lanes.left_lane_exists,
                "right_lane": lanes.right_lane_exists,
                "dist_cm": round(distance_cm, 1),
                "obstacles": len(detections),
                "action": action.value,
                "drive_l": l,
                "drive_r": r,
                "odometry_m": round(odometry_m, 3),
            }, frame)
            print(f"[{logger.count:5d}] offset={lanes.offset} dist={distance_cm:5.1f}cm "
                  f"obstacles={len(detections)} -> {action.value:11s} drive=({l},{r})")

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
