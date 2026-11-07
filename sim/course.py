"""The driving course as pure geometry (no rendering): an X-Z-ground centreline of straight and arc segments, sampled at a fixed step, with objects placed by distance along it and a lateral offset."""
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
    ("arc", 25.0, 60.0),
    ("straight", 25.0),
    ("arc", 25.0, -70.0),
    ("straight", 35.0),
    ("arc", 20.0, 50.0),
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
    """Build the centreline, repeating the segment pattern n_tiles times end-to-end."""
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
            else:
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
        """World (x, z) at dist along the road, offset lateral m to the right (negative = left)."""
        x, z, h = self.sample(dist)
        rx, rz = _right(h)
        return x + rx * lateral, z + rz * lateral


def build_course(min_length: float | None = None) -> Course:
    """Build the course centreline and place its signs and traffic, tiling the pattern to at least min_length (one tile if None)."""
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
        """A car in-lane at dist along the road (direction -1 = oncoming)."""
        x, z = c.place(dist, lateral)
        _, _, h = c.sample(dist)
        c.objects.append(Placed("car", x, z, h + (0 if direction > 0 else 180),
                                class_name="car", road_dist=dist, lateral=lateral,
                                direction=direction, **kw))

    for k in range(n_tiles):
        off = k * tile_len
        sign(off + 48, "stop_sign", class_name="stop sign", extent=(0.4, 0.4, 0.05))
        sign(off + tile_len - 20, "stop_sign", class_name="stop sign",
             extent=(0.4, 0.4, 0.05))
        car(off + 62, shoulder_lat, extent=(0.9, 0.7, 2.2))
        # stalled car blocking the ego lane — the ego must change into the open right lane to pass
        car(off + 85, 0.0, speed_mps=0.0, extent=(0.9, 0.7, 2.2))
        car(off + 115, -spec.LANE_WIDTH_M, direction=-1, speed_mps=4.0,
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
