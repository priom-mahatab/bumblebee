"""Turns a raw frame into lane geometry the control loop can steer on:
the current lane's center offset, plus whether an adjacent lane exists on each
side (so the lane-change controller knows where it may legally move)."""
from dataclasses import dataclass, field

import numpy as np
import cv2


@dataclass
class LaneAnalysis:
    """One frame's lane read.

    offset keeps the EXACT meaning/units of the old compute_lane_offset: the
    signed fraction of half-frame-width the current lane's center sits from image
    center (negative = lane center is left of image center), or None if no lane
    was found. left/right_lane_exists say whether another boundary was found
    beyond the current lane's boundary on that side. boundaries holds every
    detected boundary as a fitted ((x_bottom, y_bottom), (x_top, y_top)) line,
    left->right, for overlay/debugging (the controller ignores it)."""
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


# ROI trapezoid as (x_frac, y_frac) corners. The ROI is CAMERA-SPECIFIC:
# this default suits real dashcam clips; the sim has different geometry and
# passes its own trapezoid (see SIM_ROI in sim/run_sim.py).
REAL_ROI = [(0.10, 0.82), (0.95, 0.82), (0.58, 0.60), (0.42, 0.60)]

# Half the lane width at the look-ahead sample point, as a fraction of frame
# width. Used to reconstruct the lane center from ONE line when the other leaves
# view mid-curve (measured ~0.215 in the sim; only used in the one-sided case).
LANE_HALF_WIDTH_FRAC = 0.215

# Two Hough segments whose x-intercepts at the bottom of the ROI are within this
# many pixels are treated as the same lane boundary. Separate boundaries sit a
# lane-width apart, well beyond it. Tune to the track's lane width in pixels.
LANE_LINE_MERGE_PX = 40
# A boundary cluster backed by fewer segments than this is dropped as noise.
# Start at 1 (synthetic frames yield few segments); raise on real footage if
# spurious boundaries flicker the adjacency flags.
MIN_SEGMENTS_PER_BOUNDARY = 1


def _segment_bottom_x(seg: tuple[int, int, int, int], y_bottom: int) -> float:
    """x where the line through this segment crosses y_bottom (the ROI's near
    edge). Grouping on this — where a boundary would meet the car — is stable
    under perspective, since lines that converge far away are still separated
    up close."""
    x1, y1, x2, y2 = seg
    m = (x2 - x1) / (y2 - y1)          # x as a function of y; y2 != y1 (horizontals dropped)
    return m * (y_bottom - y1) + x1


def _cluster_boundaries(segments, y_bottom, y_top, merge_px, min_segments):
    """Group segments into individual lane boundaries by bottom x-intercept and
    fit a line to each; returns the fits sorted left->right.

    This replaces the old slope-sign bucketing, which lumped every left-leaning
    segment into one line and every right-leaning one into another — correct for
    a single lane, but on a multi-lane road it averages several boundaries into a
    phantom pair. Here each real boundary stays its own cluster."""
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
    fits.sort(key=lambda f: f[0][0])   # left -> right by bottom x
    return fits


def analyze_lanes(frame: np.ndarray, roi=REAL_ROI, lookahead: float = 0.85,
                  merge_px: int = LANE_LINE_MERGE_PX,
                  min_segments: int = MIN_SEGMENTS_PER_BOUNDARY) -> LaneAnalysis:
    """Full lane read: current-lane offset plus adjacent-lane flags (see
    LaneAnalysis). Finds EVERY boundary in the ROI, takes the pair straddling the
    car as the current lane, and reports whether more boundaries lie beyond them.

    NOTE: adjacency only works if the ROI is wide enough to contain the
    neighbouring boundaries. An ROI framed tightly to one lane (good for
    lane-keeping) will always report no adjacent lanes — widen it for adjacency."""
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
        if x2 == x1:                                   # vertical: no finite slope
            continue
        if abs((y2 - y1) / (x2 - x1)) < 0.5:           # near-horizontal: not a lane line
            continue
        segments.append((x1, y1, x2, y2))

    fys = [fy for _, fy in roi]
    y_bottom, y_top = int(max(fys) * h), int(min(fys) * h)
    boundaries = _cluster_boundaries(segments, y_bottom, y_top, merge_px, min_segments)
    if not boundaries:
        return LaneAnalysis(None)

    center_x = w / 2
    # Current lane = nearest boundary left of center, nearest boundary right of it.
    left_idx = right_idx = None
    for i, fit in enumerate(boundaries):
        if fit[0][0] <= center_x:
            left_idx = i                               # keep the last one <= center
        else:
            right_idx = i                              # first one > center
            break

    # Sample the lane center at a LOOK-AHEAD point along each boundary (fit[0] is
    # the near end, fit[1] the far end) so the car steers INTO curves rather than
    # reacting only once the bend reaches the bumper.
    def sample_x(fit):
        return fit[0][0] + lookahead * (fit[1][0] - fit[0][0])

    half_w = LANE_HALF_WIDTH_FRAC * w
    left_fit = boundaries[left_idx] if left_idx is not None else None
    right_fit = boundaries[right_idx] if right_idx is not None else None
    if left_fit is not None and right_fit is not None:
        lane_center_x = (sample_x(left_fit) + sample_x(right_fit)) / 2
    elif left_fit is not None:                         # only one boundary visible:
        lane_center_x = sample_x(left_fit) + half_w    # offset it by the known half-width
    elif right_fit is not None:
        lane_center_x = sample_x(right_fit) - half_w
    else:
        return LaneAnalysis(None, boundaries=boundaries)

    offset = (lane_center_x - center_x) / center_x
    left_lane_exists = left_idx is not None and left_idx > 0
    right_lane_exists = right_idx is not None and right_idx < len(boundaries) - 1
    return LaneAnalysis(offset, left_lane_exists, right_lane_exists, boundaries)


def compute_lane_offset(frame: np.ndarray, roi=REAL_ROI, lookahead: float = 0.85) -> float | None:
    """Current-lane center offset only — a thin wrapper over analyze_lanes, kept
    so existing callers (main.py, the sim, lane_test) stay unchanged. Same signed
    fraction as before; None when no lane is found. NOTE: on a multi-lane frame
    this now returns the EGO lane's offset (nearest boundaries straddling the
    car), not the old slope-averaged phantom — an intended fix, but re-verify on
    real clips since the number can shift from the previous behavior."""
    return analyze_lanes(frame, roi, lookahead).offset


if __name__ == "__main__":
    from sim.mock_inputs import make_lane_frame

    # Wide ROI so neighbouring boundaries fall inside it (adjacency needs this).
    WIDE_ROI = [(0.02, 0.96), (0.98, 0.96), (0.62, 0.52), (0.38, 0.52)]
    #  label, boundary x-fractions (left->right), expected (left_lane, right_lane)
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
