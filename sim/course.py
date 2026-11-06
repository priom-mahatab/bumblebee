"""The driving course as pure geometry — no rendering, so it can be reasoned
about and unit-tested on its own.

Convention matches the sim's Ursina world: ground is the X-Z plane, +Y is up,
and a heading of 0 deg points along +Z, increasing toward +X (a right turn).
Distances are metres (road-scale, like sim/worlds/track.wbt).

The course is a centreline built from straight and arc segments (the arcs are
the turns), sampled at a fixed step. Objects (signs, traffic) are placed by
their distance *along* the centreline and a lateral offset, so they always sit
correctly relative to the road however the road bends.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import spec

STEP_M = 0.5  # centreline sampling resolution


@dataclass
class Placed:
    """An object positioned relative to the road."""
    kind: str                     # 'stop_sign' | 'speed_limit' | 'car'
    x: float
    z: float
    heading_deg: float
    class_name: str | None = None  # detector/GT label, or None if not a target
    value: int | None = None       # speed-limit number
    speed_mps: float = 0.0         # >0 for moving traffic (along the road)
    extent: tuple = (0, 0, 0)      # half-extents (hx,hy,hz) for GT boxes
    road_dist: float = 0.0         # arc-length position along the centreline
    lateral: float = 0.0           # lane offset from centreline (right +, left -)
    direction: int = 1             # +1 travels with the road, -1 oncoming


# Course shape: (segment_type, *params). Arc angle >0 turns right, <0 left.
_SEGMENTS = [
    ("straight", 30.0),
    ("arc", 25.0, 60.0),     # sweeping right-hander
    ("straight", 25.0),
    ("arc", 25.0, -70.0),    # left-hander
    ("straight", 35.0),
    ("arc", 20.0, 50.0),     # tighter right
    ("straight", 25.0),
]


def _dir(h_deg: float) -> tuple[float, float]:
    """Unit forward (x,z) for a heading."""
    h = math.radians(h_deg)
    return math.sin(h), math.cos(h)


def _right(h_deg: float) -> tuple[float, float]:
    """Unit rightward (x,z) for a heading (90 deg clockwise from forward)."""
    h = math.radians(h_deg)
    return math.cos(h), -math.sin(h)


def _build_centerline(n_tiles: int = 1):
    """Build the centreline, repeating the segment pattern `n_tiles` times
    end-to-end (the accumulators carry over, so tiles join seamlessly)."""
    xs, zs, hdgs, cum = [0.0], [0.0], [0.0], [0.0]
    x, z, h, s = 0.0, 0.0, 0.0, 0.0
    for _ in range(n_tiles):
        for seg in _SEGMENTS:
            if seg[0] == "straight":
                length = seg[1]
                n = max(1, round(length / STEP_M))
                for _ in range(n):
                    dx, dz = _dir(h)
                    x += dx * STEP_M
                    z += dz * STEP_M
                    s += STEP_M
                    xs.append(x); zs.append(z); hdgs.append(h); cum.append(s)
            else:  # arc
                _, radius, angle = seg
                arc_len = abs(math.radians(angle)) * radius
                n = max(1, round(arc_len / STEP_M))
                dh = angle / n
                for _ in range(n):
                    h += dh
                    dx, dz = _dir(h)
                    x += dx * STEP_M
                    z += dz * STEP_M
                    s += STEP_M
                    xs.append(x); zs.append(z); hdgs.append(h); cum.append(s)
    return xs, zs, hdgs, cum


@dataclass
class Course:
    xs: list = field(default_factory=list)
    zs: list = field(default_factory=list)
    hdgs: list = field(default_factory=list)
    cum: list = field(default_factory=list)
    objects: list = field(default_factory=list)

    @property
    def length(self) -> float:
        return self.cum[-1]

    def sample(self, dist: float):
        """(x, z, heading_deg) at arc-length `dist` along the centreline."""
        dist = max(0.0, min(dist, self.cum[-1]))
        i = min(range(len(self.cum)), key=lambda k: abs(self.cum[k] - dist))
        return self.xs[i], self.zs[i], self.hdgs[i]

    def place(self, dist: float, lateral: float):
        """World (x, z) at `dist` along the road, offset `lateral` m to the
        right (negative = left)."""
        x, z, h = self.sample(dist)
        rx, rz = _right(h)
        return x + rx * lateral, z + rz * lateral


def build_course(min_length: float | None = None) -> Course:
    """Build the course centreline and place its signs and traffic.

    min_length: if given, the segment pattern is tiled end-to-end until the road
    is at least this long, and every sign/car is placed once per tile — so a
    longer run has road, signs, and traffic the whole way (used by the sim's
    --seconds option). Default (None) = one tile, the original course.
    """
    tile_len = _build_centerline(1)[3][-1]
    n_tiles = 1 if not min_length else max(1, math.ceil(min_length / tile_len))
    xs, zs, hdgs, cum = _build_centerline(n_tiles)
    c = Course(xs, zs, hdgs, cum)

    road_half = spec.NUM_LANES * spec.LANE_WIDTH_M / 2
    sign_lat = road_half + 0.9      # just clear of the road edge
    shoulder_lat = road_half + 1.6

    def sign(dist, kind, **kw):
        x, z = c.place(dist, sign_lat)
        _, _, h = c.sample(dist)
        # face back toward oncoming traffic (road heading + 180)
        c.objects.append(Placed(kind, x, z, h + 180,
                                road_dist=dist, lateral=sign_lat, **kw))

    def car(dist, lateral, direction=1, **kw):
        """A car in-lane at `dist` along the road (direction -1 = oncoming).
        road_dist/lateral/direction let world.update keep it ON the road."""
        x, z = c.place(dist, lateral)
        _, _, h = c.sample(dist)
        c.objects.append(Placed("car", x, z, h + (0 if direction > 0 else 180),
                                class_name="car", road_dist=dist, lateral=lateral,
                                direction=direction, **kw))

    # Place signs + traffic once per tile so the whole (possibly long) road is
    # populated; offsets are the original single-tile positions.
    for k in range(n_tiles):
        off = k * tile_len
        # speed-limit signs are GT *targets* (so the camera gets their bbox), but
        # the number is NOT exposed to perception — it's read from pixels (OCR).
        sign(off + 16, "speed_limit", class_name="speed limit", value=35,
             extent=(0.3, 0.45, 0.03))
        sign(off + 48, "stop_sign", class_name="stop sign", extent=(0.4, 0.4, 0.05))
        sign(off + 92, "speed_limit", class_name="speed limit", value=25,
             extent=(0.3, 0.45, 0.03))
        sign(off + tile_len - 20, "stop_sign", class_name="stop sign",
             extent=(0.4, 0.4, 0.05))
        # parked car on the right shoulder (static obstacle, off the path)
        car(off + 62, shoulder_lat, extent=(0.9, 0.7, 2.2))
        # a slower car ahead in the ego lane — ego should SLOW as it closes in
        car(off + 78, 0.0, speed_mps=2.5, extent=(0.9, 0.7, 2.2))
        # oncoming car in the far lane
        car(off + 110, -spec.LANE_WIDTH_M, direction=-1, speed_mps=4.0,
            extent=(0.9, 0.7, 2.2))
    return c


def ego_start(c: Course):
    """(x, z, heading_deg) a little way into the course, in the ego lane."""
    x, z, h = c.sample(2.0)
    return x, z, h


if __name__ == "__main__":
    c = build_course()
    print(f"course length: {c.length:.1f} m, {len(c.xs)} centreline points")
    print(f"objects: {len(c.objects)}")
    for o in c.objects:
        tag = o.class_name or o.kind
        print(f"  {tag:12s} at ({o.x:6.1f}, {o.z:6.1f})  hdg={o.heading_deg:6.1f}"
              f"  speed={o.speed_mps}")
    print("ego start:", tuple(round(v, 2) for v in ego_start(c)))
