"""Draw an ROI profile and the detected lane boundaries/offset on a still frame for ROI tuning."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from vision.lane_detection import LANE_PROFILES, analyze_lanes, get_profile


def render(frame, profile):
    """Return a copy of frame annotated with the ROI polygon, fitted boundaries, lane centre and offset, plus the analysis."""
    h, w = frame.shape[:2]
    a = analyze_lanes(frame, roi=profile)
    out = frame.copy()
    poly = np.array([[(int(fx * w), int(fy * h)) for fx, fy in get_profile(profile).roi]], dtype=np.int32)
    cv2.polylines(out, poly, True, (255, 0, 255), 2)
    cv2.line(out, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
    for (xb, yb), (xt, yt) in a.boundaries:
        cv2.line(out, (xb, yb), (xt, yt), (0, 255, 0), 3)
    if a.offset is not None:
        cv2.circle(out, (int((w / 2) * (1 + a.offset)), int(0.9 * h)), 8, (0, 255, 255), -1)
    off = f"{a.offset:+.3f}" if a.offset is not None else "none"
    cv2.putText(out, f"roi={profile} offset={off} L={a.left_lane_exists} R={a.right_lane_exists} "
                f"boundaries={len(a.boundaries)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return out, a


def main():
    parser = argparse.ArgumentParser(description="Preview a lane ROI profile on still frames.")
    parser.add_argument("images", nargs="+", help="frame image(s), e.g. frame.jpg or RunLogger dumps")
    parser.add_argument("--roi", default="rover", choices=sorted(LANE_PROFILES))
    parser.add_argument("--out-dir", default=None, help="where to save (default: next to each image)")
    args = parser.parse_args()

    for path in map(Path, args.images):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"{path}: could not read")
            continue
        out, a = render(frame, args.roi)
        out_dir = Path(args.out_dir) if args.out_dir else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}_roi_{args.roi}.jpg"
        cv2.imwrite(str(out_path), out)
        bottoms = [fit[0][0] for fit in a.boundaries]
        print(f"{path}: offset={a.offset} L={a.left_lane_exists} R={a.right_lane_exists} "
              f"boundary_bottom_x={bottoms} -> {out_path}")


if __name__ == "__main__":
    main()
