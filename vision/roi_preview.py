"""Draw an ROI profile and the detected lane boundaries/offset on a still frame for ROI tuning."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from vision.birdseye import floor_to_topdown, image_to_floor, warp_topdown
from vision.lane_detection import BEV_LOOKAHEAD_CM, LANE_PROFILES, LaneTracker, analyze_lanes, get_profile


def _topdown_panel(frame, analysis, height):
    """Top-down floor view with the tracked tapes (green) and the steering target (yellow), scaled to the given height."""
    td = warp_topdown(frame)
    for seg in analysis.boundaries:
        a, b = floor_to_topdown(image_to_floor(np.array(seg, dtype=np.float32)))
        cv2.line(td, tuple(map(int, a)), tuple(map(int, b)), (0, 255, 0), 3)
    rover = floor_to_topdown(np.array([[0.0, 0.0]]))[0]
    if analysis.offset is not None:
        target = floor_to_topdown(np.array([[analysis.offset * BEV_LOOKAHEAD_CM, BEV_LOOKAHEAD_CM]]))[0]
        cv2.line(td, tuple(map(int, rover)), tuple(map(int, target)), (0, 255, 255), 2)
        cv2.circle(td, tuple(map(int, target)), 6, (0, 255, 255), -1)
    scale = height / td.shape[0]
    return cv2.resize(td, (int(td.shape[1] * scale), height))


def render(frame, profile, tracker=None):
    """Return a copy of frame annotated with the ROI polygon, fitted boundaries, lane centre and offset (plus a top-down panel for bird's-eye profiles), and the analysis."""
    h, w = frame.shape[:2]
    a = analyze_lanes(frame, roi=profile, tracker=tracker)
    out = frame.copy()
    poly = np.array([[(int(fx * w), int(fy * h)) for fx, fy in get_profile(profile).roi]], dtype=np.int32)
    cv2.polylines(out, poly, True, (255, 0, 255), 2)
    cv2.line(out, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
    for (xb, yb), (xt, yt) in a.boundaries:
        cv2.line(out, (xb, yb), (xt, yt), (0, 255, 0), 3)
    if a.offset is not None and not get_profile(profile).birdseye:
        cv2.circle(out, (int((w / 2) * (1 + a.offset)), int(0.9 * h)), 8, (0, 255, 255), -1)
    off = f"{a.offset:+.3f}" if a.offset is not None else "none"
    cv2.putText(out, f"roi={profile} offset={off} L={a.left_lane_exists} R={a.right_lane_exists} "
                f"boundaries={len(a.boundaries)}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if get_profile(profile).birdseye:
        out = np.hstack([out, _topdown_panel(frame, a, h)])
    return out, a


def main():
    parser = argparse.ArgumentParser(description="Preview a lane ROI profile on still frames.")
    parser.add_argument("images", nargs="+", help="frame image(s), e.g. frame.jpg or RunLogger dumps")
    parser.add_argument("--roi", default="rover", choices=sorted(LANE_PROFILES))
    parser.add_argument("--out-dir", default=None, help="where to save (default: next to each image)")
    parser.add_argument("--sequence", action="store_true",
                        help="treat images as consecutive frames of one run and carry tape memory between them")
    args = parser.parse_args()
    tracker = LaneTracker() if args.sequence else None

    for path in map(Path, args.images):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"{path}: could not read")
            continue
        out, a = render(frame, args.roi, tracker)
        out_dir = Path(args.out_dir) if args.out_dir else path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{path.stem}_roi_{args.roi}.jpg"
        cv2.imwrite(str(out_path), out)
        bottoms = [fit[0][0] for fit in a.boundaries]
        print(f"{path}: offset={a.offset} L={a.left_lane_exists} R={a.right_lane_exists} "
              f"boundary_bottom_x={bottoms} -> {out_path}")


if __name__ == "__main__":
    main()
