"""Closed-loop rover driver on the Pi: camera -> lane + colour-obstacle detection -> control -> motors."""
import time

from control.motor_control import Motors, Ultrasonic, action_to_drive
from control.state_machine import DrivingController
from vision.camera import PiCameraSource
from vision.lane_detection import analyze_lanes
from vision.obstacles import detect_obstacles

CRUISE_SPEED_MPS = 0.3   # assumed forward speed; dead-reckons odometry for lane-change progress (no encoders)
LOOP_HZ = 15


def main():
    cam = PiCameraSource()
    motors = Motors()
    sonar = Ultrasonic()
    controller = DrivingController()
    period = 1.0 / LOOP_HZ
    start = time.monotonic()
    try:
        while True:
            t0 = time.monotonic()
            frame = cam.read()
            if frame is None:
                continue
            h, w = frame.shape[:2]
            lanes = analyze_lanes(frame)
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
            l, r = action_to_drive(action)
            motors.drive(l, r)
            print(f"offset={lanes.offset} dist={distance_cm:5.1f}cm "
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


if __name__ == "__main__":
    main()
