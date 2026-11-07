"""Overlay analyze_lanes output on real footage to eyeball lane offset and adjacency flags."""
import os

import cv2

from vision.lane_detection import analyze_lanes

ROADS = "datasets/test_images/roads"
VIDEO = "datasets/test_videos/project_video.mp4"
OUT = f"{ROADS}/lane_output"
IMAGES = ["dashcam_lane.png", "lane_2.jpg", "lane_3.jpg"]

# Wide ROI so adjacent-lane boundaries fall inside it (narrow ROI -> no adjacency).
WIDE_ROI = [(0.03, 0.95), (0.97, 0.95), (0.62, 0.60), (0.38, 0.60)]


def annotate(frame, roi=WIDE_ROI):
    """Draw detected boundaries + center line, and the offset/adjacency read."""
    h, w = frame.shape[:2]
    a = analyze_lanes(frame, roi=roi)
    out = frame.copy()
    cv2.line(out, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
    for fit in a.boundaries:
        (xb, yb), (xt, yt) = fit
        cv2.line(out, (xb, yb), (xt, yt), (0, 255, 0), 4)
    off = f"{a.offset:+.2f}" if a.offset is not None else "None"
    cv2.putText(out, f"offset={off}  L={a.left_lane_exists}  R={a.right_lane_exists}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return out, a


def main():
    os.makedirs(OUT, exist_ok=True)

    print("--- sample road stills ---")
    for name in IMAGES:
        frame = cv2.imread(f"{ROADS}/{name}")
        if frame is None:
            print(f"{name:22s} -> MISSING")
            continue
        out, a = annotate(frame)
        print(f"{name:22s} -> offset={a.offset}  L={a.left_lane_exists} "
              f"R={a.right_lane_exists}  ({len(a.boundaries)} boundaries)")
        cv2.imwrite(f"{OUT}/{name.rsplit('.', 1)[0]}_result.jpg", out)

    print(f"\n--- {VIDEO} (sampled frames) ---")
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("could not open video")
    else:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1200
        for i, f in enumerate(range(100, total, max(1, total // 5))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                continue
            out, a = annotate(frame)
            print(f"frame {f:5d} -> offset={a.offset}  L={a.left_lane_exists} "
                  f"R={a.right_lane_exists}  ({len(a.boundaries)} boundaries)")
            cv2.imwrite(f"{OUT}/project_video_f{f:05d}_result.jpg", out)
        cap.release()

    print(f"\nsaved overlays to {OUT}/")


if __name__ == "__main__":
    main()
