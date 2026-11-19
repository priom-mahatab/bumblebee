"""Turns a raw frame into lane geometry: current-lane center offset plus adjacent-lane flags."""
from dataclasses import dataclass, field, replace

import numpy as np
import cv2

from vision.birdseye import floor_to_image, image_to_floor


@dataclass
class LaneAnalysis:
    """One frame's lane read: signed offset (fraction of half-frame-width, or lateral/lookahead on the floor plane for bird's-eye profiles; negative = steer left, None if no lane), adjacent-lane flags, fitted boundaries left->right, and for bird's-eye profiles the lane centre's slope dx/dz (negative = rover angled right of the lane) and its lateral position at the rover in cm (positive = centre to the right)."""
    offset: float | None
    left_lane_exists: bool = False
    right_lane_exists: bool = False
    boundaries: list = field(default_factory=list)
    heading: float | None = None
    centre_cm: float | None = None

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

TRACK_LANE_WIDTH_CM = 41.0
BEV_Z_RANGE_CM = (25.0, 120.0)
BEV_BIN_CM = 5.0
BEV_LOOKAHEAD_CM = 60.0
BEV_CLUSTER_GAP_CM = 4.0
BEV_MAX_TAPE_WIDTH_CM = 10.0
BEV_TRACK_TOL_CM = 8.0
BEV_LANE_WIDTH_TOL = 0.4
PAIR_WIDTH_TOL = 0.25
BEV_MIN_BINS = 2
TRACK_MEMORY_FRAMES = 5
TRACK_MEMORY_TOL_CM = 10.0
STEEP_TAPE_SLOPE = 0.7
PAIR_SLOPE_TOL = 0.35
LONE_TAPE_MIN_BINS = 4
MIN_TRACK_BINS = 3
FRAME_EDGE_MARGIN_PX = 12


@dataclass(frozen=True)
class LaneProfile:
    """Per-camera lane settings: ROI corners (bottom-left, bottom-right, top-right, top-left as frame fractions), min |dy/dx| per segment, boundary merge distance, half lane width for one-boundary frames, an optional HSV range that replaces grayscale edges with a colour mask, and whether to track tape on the calibrated floor plane instead of fitting image lines."""
    roi: tuple
    min_abs_slope: float = MIN_ABS_SLOPE
    merge_px: int = LANE_LINE_MERGE_PX
    lane_half_width_frac: float = LANE_HALF_WIDTH_FRAC
    hsv_range: tuple | None = None
    birdseye: bool = False


LANE_PROFILES = {
    "dashcam": LaneProfile(roi=((0.10, 0.82), (0.95, 0.82), (0.58, 0.60), (0.42, 0.60))),
    "sim": LaneProfile(roi=((0.0, 0.88), (1.0, 0.88), (0.80, 0.48), (0.20, 0.48))),
    "rover": LaneProfile(roi=((0.0, 0.98), (1.0, 0.98), (1.0, 0.58), (0.0, 0.58)),
                         min_abs_slope=0.1, merge_px=150, lane_half_width_frac=0.27,
                         hsv_range=ROVER_TAPE_HSV, birdseye=True),
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


def _row_tapes(xs: np.ndarray) -> list[float]:
    """Split one depth bin's floor x values into tape crossings, dropping clusters too wide to be a tape running forward."""
    xs = np.sort(xs)
    clusters = np.split(xs, np.nonzero(np.diff(xs) > BEV_CLUSTER_GAP_CM)[0] + 1)
    return [float(c.mean()) for c in clusters if c[-1] - c[0] <= BEV_MAX_TAPE_WIDTH_CM]


def _seed_sides(tapes: list[float]) -> dict:
    """Pick the ego lane's left/right tape from the nearest bin: a pair about one lane width apart, else a lone tape by which side of the rover it is on."""
    pairs = [(a, b) for i, a in enumerate(tapes) for b in tapes[i + 1:]
             if abs(b - a - TRACK_LANE_WIDTH_CM) <= BEV_LANE_WIDTH_TOL * TRACK_LANE_WIDTH_CM]
    if pairs:
        a, b = min(pairs, key=lambda p: abs((p[0] + p[1]) / 2))
        return {"left": a, "right": b}
    lone = min(tapes, key=abs)
    return {"left" if lone < 0 else "right": lone}


def _shift_line(line: tuple[float, float] | None, dx: float) -> tuple[float, float] | None:
    """Move a floor line x = m*z + b sideways by dx cm measured perpendicular to it."""
    if line is None:
        return None
    m, b = line
    return m, b + dx * float(np.hypot(1, m))


def _predict(track: list[tuple[float, float]], z: float) -> float:
    if len(track) < 2:
        return track[-1][1]
    (z0, x0), (z1, x1) = track[-2], track[-1]
    return x1 + (x1 - x0) / (z1 - z0) * (z - z1)


class LaneTracker:
    """Remembers each ego tape's floor line (x = m*z + b) for a few frames so a lone tape keeps its side; use one per camera stream."""

    def __init__(self, memory_frames: int = TRACK_MEMORY_FRAMES):
        self.memory_frames = memory_frames
        self.lines = {"left": None, "right": None}
        self.age = {"left": 0, "right": 0}

    def predict(self, side: str, z: float) -> float | None:
        """Remembered x of this side's tape at depth z, or None if forgotten."""
        line = self.lines[side]
        return None if line is None else line[0] * z + line[1]

    def shift_lane(self, direction: str, width: float = TRACK_LANE_WIDTH_CM) -> None:
        """Relabel the remembered tapes one lane over toward direction, so lane keeping steers into that lane."""
        sign = 1 if direction == "right" else -1
        ahead, behind = ("right", "left") if direction == "right" else ("left", "right")
        divider = self.lines[ahead] or _shift_line(self.lines[behind], sign * width)
        if divider is None:
            return
        self.lines[behind] = divider
        self.lines[ahead] = _shift_line(divider, sign * width)
        self.age = {"left": 0, "right": 0}

    def update(self, fits: dict) -> None:
        """Store this frame's fitted tape lines and age out sides that were not seen."""
        for side in ("left", "right"):
            if side in fits:
                self.lines[side], self.age[side] = fits[side], 0
            elif self.lines[side] is not None:
                self.age[side] += 1
                if self.age[side] > self.memory_frames:
                    self.lines[side] = None


def _hugs_frame_edge(track: list[tuple[float, float]], frame_w: int) -> bool:
    """True if most of a track's points sit at the image's left/right border, i.e. a tape cut off by the frame edge rather than one running forward."""
    img_x = floor_to_image(np.array([(x, z) for z, x in track]))[:, 0]
    near_edge = (img_x < FRAME_EDGE_MARGIN_PX) | (img_x > frame_w - FRAME_EDGE_MARGIN_PX)
    return near_edge.mean() > 0.5


def _analyze_birdseye(mask: np.ndarray, tracker: LaneTracker | None) -> LaneAnalysis:
    """Bird's-eye lane read that also feeds the tracker its tape lines for the next frame."""
    analysis, fits = _track_birdseye(mask, tracker)
    if tracker is not None:
        tracker.update(fits)
    return analysis


def _track_birdseye(mask: np.ndarray, tracker: LaneTracker | None) -> tuple[LaneAnalysis, dict]:
    """Track the ego tapes bin by bin outward on the floor plane, seeded from tracker memory when available, and steer toward the lane centre at the lookahead distance; a guessed-side tape steeper than STEEP_TAPE_SLOPE counts only as part of a parallel pair, a lone tape must span LONE_TAPE_MIN_BINS bins, and a lone tape puts the centre half a lane perpendicular to it; a left/right pair not about a lane width apart keeps only its longer track."""
    ys, xs = np.nonzero(mask[::2, ::2])
    if not len(xs):
        return LaneAnalysis(None), {}
    pts = image_to_floor(np.c_[xs * 2, ys * 2])
    z_near, z_far = BEV_Z_RANGE_CM
    pts = pts[(pts[:, 1] >= z_near) & (pts[:, 1] < z_far)]

    width, half = TRACK_LANE_WIDTH_CM, TRACK_LANE_WIDTH_CM / 2
    tracks = {"left": [], "right": []}
    row_tapes = {}

    origin = {}

    def expect(side, z):
        other, sign = ("right", -1) if side == "left" else ("left", 1)
        if tracks[side]:
            return _predict(tracks[side], z), BEV_TRACK_TOL_CM, origin[side]
        remembered = tracker.predict(side, z) if tracker is not None else None
        if remembered is not None:
            return remembered, TRACK_MEMORY_TOL_CM, "memory"
        if tracks[other]:
            return _predict(tracks[other], z) + sign * width, BEV_LANE_WIDTH_TOL * width, "other"
        return None, None, None

    for z in np.arange(z_near + BEV_BIN_CM / 2, z_far, BEV_BIN_CM):
        row = pts[np.abs(pts[:, 1] - z) < BEV_BIN_CM / 2, 0]
        tapes = _row_tapes(row) if len(row) else []
        if not tapes:
            continue
        row_tapes[z] = tapes
        expected = {side: expect(side, z) for side in ("left", "right")}
        if all(pred is None for pred, _, _ in expected.values()):
            hits = _seed_sides(tapes)
            sources = {side: "pair" if len(hits) == 2 else "lone" for side in hits}
        else:
            hits, sources = {}, {}
            candidates = sorted((abs(x - pred), side, x, src) for side, (pred, tol, src) in expected.items()
                                if pred is not None for x in tapes if abs(x - pred) <= tol)
            for _, side, x, src in candidates:
                if side not in hits and x not in hits.values():
                    hits[side], sources[side] = x, src
        for side, x in hits.items():
            tracks[side].append((z, x))
            origin.setdefault(side, sources[side])

    fits = {side: tuple(np.polyfit(*np.array(t).T, 1)) for side, t in tracks.items()
            if len(t) >= MIN_TRACK_BINS and not _hugs_frame_edge(t, mask.shape[1])}
    if len(fits) == 2:
        (ml, bl), (mr, br) = fits["left"], fits["right"]
        zm = np.mean([z for z, _ in tracks["left"] + tracks["right"]])
        gap = ((mr - ml) * zm + br - bl) / np.hypot(1, (ml + mr) / 2)
        if abs(gap - width) > PAIR_WIDTH_TOL * width:
            del fits[min(fits, key=lambda side: len(tracks[side]))]
    paired = (len(fits) == 2 and abs(fits["left"][0] - fits["right"][0]) <= PAIR_SLOPE_TOL)
    if not paired:
        guessed = {side for side, src in origin.items()
                   if src == "lone" or (src == "other" and origin.get("right" if side == "left" else "left") == "lone")}
        fits = {side: f for side, f in fits.items() if side not in guessed or abs(f[0]) <= STEEP_TAPE_SLOPE}
    if len(fits) == 1 and len(tracks[next(iter(fits))]) < LONE_TAPE_MIN_BINS:
        fits = {}
    if not fits:
        return LaneAnalysis(None), {}

    points = {side: dict(tracks[side]) for side in fits}
    centres = []
    left_lane = right_lane = False
    for z, tapes in row_tapes.items():
        xl, xr = points.get("left", {}).get(z), points.get("right", {}).get(z)
        if xl is not None and xr is not None:
            centre = (xl + xr) / 2
        elif xl is not None:
            centre = xl + half * np.hypot(1, fits["left"][0])
        elif xr is not None:
            centre = xr - half * np.hypot(1, fits["right"][0])
        else:
            continue
        centres.append((z, centre))
        left_lane |= any(x < centre - 1.5 * half - BEV_TRACK_TOL_CM for x in tapes)
        right_lane |= any(x > centre + 1.5 * half + BEV_TRACK_TOL_CM for x in tapes)

    if len(centres) < BEV_MIN_BINS:
        return LaneAnalysis(None), fits
    zs, cs = np.array(centres).T
    z_look = float(np.clip(BEV_LOOKAHEAD_CM, zs.min(), zs.max()))
    centre_fit = np.polyfit(zs, cs, 1)
    offset = float(np.polyval(centre_fit, z_look)) / z_look

    boundaries = []
    for side, (m, b) in fits.items():
        tz = np.array(tracks[side])[:, 0]
        ends = floor_to_image(np.array([[m * tz.min() + b, tz.min()], [m * tz.max() + b, tz.max()]]))
        boundaries.append(tuple((int(x), int(y)) for x, y in ends))
    return LaneAnalysis(offset, left_lane, right_lane, boundaries, float(centre_fit[0]), float(centre_fit[1])), fits


def analyze_lanes(frame: np.ndarray, roi=DEFAULT_PROFILE, lookahead: float = 0.85,
                  merge_px: int | None = None,
                  min_segments: int = MIN_SEGMENTS_PER_BOUNDARY,
                  tracker: LaneTracker | None = None) -> LaneAnalysis:
    """Full lane read (roi is a profile name, or a corner list using dashcam settings); adjacency needs an ROI wide enough to contain neighbouring boundaries, and tracker carries tape memory across frames for bird's-eye profiles."""
    profile = get_profile(roi) if isinstance(roi, str) else replace(get_profile(DEFAULT_PROFILE), roi=roi)
    roi = profile.roi
    merge_px = profile.merge_px if merge_px is None else merge_px
    edges = _lane_pixels(frame, profile.hsv_range)

    h, w = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[(int(fx * w), int(fy * h)) for fx, fy in roi]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(edges, mask)
    if profile.birdseye:
        return _analyze_birdseye(masked, tracker)

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
