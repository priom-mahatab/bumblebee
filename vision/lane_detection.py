"""Turns a raw frame into lane geometry: current-lane center offset plus adjacent-lane flags."""
from dataclasses import dataclass, field, replace

import numpy as np
import cv2


@dataclass
class LaneAnalysis:
    """One frame's lane read: signed offset (fraction of half-frame-width, negative = left of center, None if no lane), adjacent-lane flags, and fitted boundaries left->right."""
    offset: float | None
    left_lane_exists: bool = False
    right_lane_exists: bool = False
    boundaries: list = field(default_factory=list)

def _fit_lane_line(segments, y_bottom, y_top):
    if not segments:
        return None
    xs, ys = [], []
    for x1, y1, x2, y2 in segments:
        xs += [x1, x2]
        ys += [y1, y2]
    m, b = np.polyfit(ys, xs, 1)          # x = m*y + b
    return (int(m * y_bottom + b), y_bottom), (int(m * y_top + b), y_top)


# Half lane width as fraction of frame width; reconstructs center from one boundary.
LANE_HALF_WIDTH_FRAC = 0.215

# Segments within this many px at the ROI bottom are the same boundary.
LANE_LINE_MERGE_PX = 40
MIN_SEGMENTS_PER_BOUNDARY = 1
MIN_ABS_SLOPE = 0.5

ROVER_TAPE_HSV = ((95, 80, 80), (125, 255, 255))


@dataclass(frozen=True)
class LaneProfile:
    """Per-camera lane settings: ROI corners (bottom-left, bottom-right, top-right, top-left as frame fractions), min |dy/dx| per segment, boundary merge distance, half lane width for one-boundary frames, and an optional HSV range that replaces grayscale edges with a colour mask."""
    roi: tuple
    min_abs_slope: float = MIN_ABS_SLOPE
    merge_px: int = LANE_LINE_MERGE_PX
    lane_half_width_frac: float = LANE_HALF_WIDTH_FRAC
    hsv_range: tuple | None = None


LANE_PROFILES = {
    "dashcam": LaneProfile(roi=((0.10, 0.82), (0.95, 0.82), (0.58, 0.60), (0.42, 0.60))),
    "sim": LaneProfile(roi=((0.0, 0.88), (1.0, 0.88), (0.80, 0.48), (0.20, 0.48))),
    "rover": LaneProfile(roi=((0.0, 0.98), (1.0, 0.98), (1.0, 0.58), (0.0, 0.58)),
                         min_abs_slope=0.1, merge_px=150, lane_half_width_frac=0.27,
                         hsv_range=ROVER_TAPE_HSV),
}
DEFAULT_PROFILE = "dashcam"


def get_profile(name: str) -> LaneProfile:
    """Look up a named lane profile, raising ValueError for unknown names."""
    try:
        return LANE_PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown lane profile {name!r}; choose from {sorted(LANE_PROFILES)}") from None


def _lane_pixels(frame: np.ndarray, hsv_range) -> np.ndarray:
    """Binary lane-candidate image: Canny edges of the grayscale frame, or the denoised HSV colour mask when a range is given."""
    if hsv_range is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color = cv2.inRange(hsv, np.array(hsv_range[0]), np.array(hsv_range[1]))
    return cv2.morphologyEx(color, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _segment_bottom_x(seg: tuple[int, int, int, int], y_bottom: int) -> float:
    """x where the line through this segment crosses y_bottom (the ROI's near edge)."""
    x1, y1, x2, y2 = seg
    m = (x2 - x1) / (y2 - y1)          # y2 != y1: horizontals already dropped
    return m * (y_bottom - y1) + x1


def _cluster_boundaries(segments, y_bottom, y_top, merge_px, min_segments):
    """Group segments into lane boundaries by bottom x-intercept, fit each, and return fits left->right."""
    keyed = sorted((_segment_bottom_x(s, y_bottom), s) for s in segments)
    clusters, cluster, last_x = [], [], None
    for xb, seg in keyed:
        if last_x is not None and xb - last_x > merge_px:
            clusters.append(cluster)
            cluster = []
        cluster.append(seg)
        last_x = xb
    if cluster:
        clusters.append(cluster)

    fits = []
    for cl in clusters:
        if len(cl) < min_segments:
            continue
        fit = _fit_lane_line(cl, y_bottom, y_top)
        if fit is not None:
            fits.append(fit)
    fits.sort(key=lambda f: f[0][0])
    return fits


def analyze_lanes(frame: np.ndarray, roi=DEFAULT_PROFILE, lookahead: float = 0.85,
                  merge_px: int | None = None,
                  min_segments: int = MIN_SEGMENTS_PER_BOUNDARY) -> LaneAnalysis:
    """Full lane read (roi is a profile name, or a corner list using dashcam settings); adjacency needs an ROI wide enough to contain neighbouring boundaries."""
    profile = get_profile(roi) if isinstance(roi, str) else replace(get_profile(DEFAULT_PROFILE), roi=roi)
    roi = profile.roi
    merge_px = profile.merge_px if merge_px is None else merge_px
    edges = _lane_pixels(frame, profile.hsv_range)

    h, w = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[(int(fx * w), int(fy * h)) for fx, fy in roi]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked, 2, np.pi / 180, 50, minLineLength=40, maxLineGap=100)
    if lines is None:
        return LaneAnalysis(None)

    segments = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if x2 == x1:
            continue
        if y2 == y1 or abs((y2 - y1) / (x2 - x1)) < profile.min_abs_slope:
            continue
        segments.append((x1, y1, x2, y2))

    fys = [fy for _, fy in roi]
    y_bottom, y_top = int(max(fys) * h), int(min(fys) * h)
    boundaries = _cluster_boundaries(segments, y_bottom, y_top, merge_px, min_segments)
    if not boundaries:
        return LaneAnalysis(None)

    center_x = w / 2
    left_idx = right_idx = None
    for i, fit in enumerate(boundaries):
        if fit[0][0] <= center_x:
            left_idx = i
        else:
            right_idx = i
            break

    def sample_x(fit):
        return fit[0][0] + lookahead * (fit[1][0] - fit[0][0])

    half_w = profile.lane_half_width_frac * w
    left_fit = boundaries[left_idx] if left_idx is not None else None
    right_fit = boundaries[right_idx] if right_idx is not None else None
    if left_fit is not None and right_fit is not None:
        lane_center_x = (sample_x(left_fit) + sample_x(right_fit)) / 2
    elif left_fit is not None:
        lane_center_x = sample_x(left_fit) + half_w
    elif right_fit is not None:
        lane_center_x = sample_x(right_fit) - half_w
    else:
        return LaneAnalysis(None, boundaries=boundaries)

    offset = (lane_center_x - center_x) / center_x
    left_lane_exists = left_idx is not None and left_idx > 0
    right_lane_exists = right_idx is not None and right_idx < len(boundaries) - 1
    return LaneAnalysis(offset, left_lane_exists, right_lane_exists, boundaries)


def compute_lane_offset(frame: np.ndarray, roi=DEFAULT_PROFILE, lookahead: float = 0.85) -> float | None:
    """Current-lane center offset only, a thin wrapper over analyze_lanes for existing callers."""
    return analyze_lanes(frame, roi, lookahead).offset


if __name__ == "__main__":
    from sim.mock_inputs import make_lane_frame

    WIDE_ROI = [(0.02, 0.96), (0.98, 0.96), (0.62, 0.52), (0.38, 0.52)]
    cases = [
        ("1 lane, centered",        [0.35, 0.65],             False, False),
        ("2 lanes, in left lane",   [0.30, 0.55, 0.80],       False, True),
        ("2 lanes, in right lane",  [0.20, 0.45, 0.70],       True,  False),
        ("3 lanes, in center lane", [0.15, 0.38, 0.62, 0.85], True,  True),
    ]
    print("analyze_lanes on synthetic multi-lane frames (wide ROI):")
    all_ok = True
    for label, fracs, exp_l, exp_r in cases:
        a = analyze_lanes(make_lane_frame(fracs), roi=WIDE_ROI)
        passed = (a.left_lane_exists, a.right_lane_exists) == (exp_l, exp_r)
        all_ok &= passed
        off = f"{a.offset:+.2f}" if a.offset is not None else "  None"
        print(f"  {label:26s} boundaries={len(a.boundaries)} offset={off} "
              f"L={a.left_lane_exists} R={a.right_lane_exists}  "
              f"{'ok' if passed else f'FAIL expected L={exp_l} R={exp_r}'}")
    print("all passed" if all_ok else "SOME FAILED")
