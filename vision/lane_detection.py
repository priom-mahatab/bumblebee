"""Turns a raw frame into lane geometry: current-lane center offset plus adjacent-lane flags."""
from dataclasses import dataclass, field

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


# ROI trapezoid as (x_frac, y_frac) corners; camera-specific (sim passes its own).
REAL_ROI = [(0.10, 0.82), (0.95, 0.82), (0.58, 0.60), (0.42, 0.60)]

# Half lane width as fraction of frame width; reconstructs center from one boundary.
LANE_HALF_WIDTH_FRAC = 0.215

# Segments within this many px at the ROI bottom are the same boundary.
LANE_LINE_MERGE_PX = 40
MIN_SEGMENTS_PER_BOUNDARY = 1


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


def analyze_lanes(frame: np.ndarray, roi=REAL_ROI, lookahead: float = 0.85,
                  merge_px: int = LANE_LINE_MERGE_PX,
                  min_segments: int = MIN_SEGMENTS_PER_BOUNDARY) -> LaneAnalysis:
    """Full lane read: current-lane offset plus adjacent-lane flags; adjacency needs an ROI wide enough to contain neighbouring boundaries."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

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
        if abs((y2 - y1) / (x2 - x1)) < 0.5:           # drop near-horizontal non-lane lines
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

    half_w = LANE_HALF_WIDTH_FRAC * w
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


def compute_lane_offset(frame: np.ndarray, roi=REAL_ROI, lookahead: float = 0.85) -> float | None:
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
