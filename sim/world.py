"""Builds the Ursina 3D scene from a Course: a multi-lane road with painted
lane lines (dashed dividers, solid edges), textured stop/speed-limit signs,
and some traffic. Everything is `unlit` so it renders the same across GL
back-ends and so the painted lines keep a constant, detectable brightness.

build_world() returns a World holding:
  - ego:     an Entity the camera rides on (moved by the control loop)
  - labels:  LabeledObject list — the ground-truth targets (signs, cars)
  - a per-frame update(dt) that advances moving traffic

Coordinates follow course.py: X-Z ground plane, +Y up.
"""
from __future__ import annotations

from dataclasses import dataclass

from ursina import Entity, Mesh, Texture, Vec2, Vec3, color

from . import course as course_mod
from . import spec, textures


def _c(r, g, b):
    """Color from 0-255 ints (ursina's color.rgb wants 0-1 floats)."""
    return color.rgb(r / 255, g / 255, b / 255)


ROAD_COLOR = _c(58, 58, 62)
LINE_COLOR = _c(235, 235, 235)
GRASS_COLOR = _c(92, 115, 61)
CAR_COLORS = [_c(196, 60, 55), _c(55, 92, 168), _c(225, 225, 225)]


@dataclass
class LabeledObject:
    entity: Entity           # origin sits at the object's centre
    class_name: str          # GT / detector label
    extent: tuple            # half-sizes (hx, hy, hz) in the entity's frame


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def _road_mesh(c: course_mod.Course) -> Mesh:
    half = spec.NUM_LANES * spec.LANE_WIDTH_M / 2
    verts, tris, uvs = [], [], []
    for i in range(len(c.xs)):
        rx, rz = course_mod._right(c.hdgs[i])
        lx, lz = c.xs[i] - rx * half, c.zs[i] - rz * half
        rrx, rrz = c.xs[i] + rx * half, c.zs[i] + rz * half
        verts.append(Vec3(lx, 0, lz))
        verts.append(Vec3(rrx, 0, rrz))
        v = c.cum[i] / 4.0
        uvs.append(Vec2(0, v)); uvs.append(Vec2(1, v))
    for i in range(len(c.xs) - 1):
        a, b = 2 * i, 2 * i + 1
        cc, d = 2 * (i + 1), 2 * (i + 1) + 1
        tris += [a, b, d, a, d, cc]
    return Mesh(vertices=verts, triangles=tris, uvs=uvs, mode="triangle")


def _line_mesh(c: course_mod.Course, lateral: float, dashed: bool) -> Mesh:
    """One mesh for a painted line at the given lateral offset. Dashed lines
    emit quads only during the 'on' part of the dash/gap cycle."""
    half_w = spec.EDGE_LINE_WIDTH_M / 2
    period = spec.DASH_LENGTH_M + spec.DASH_GAP_M
    verts, tris, uvs = [], [], []

    def edge(i):
        rx, rz = course_mod._right(c.hdgs[i])
        cx, cz = c.xs[i] + rx * lateral, c.zs[i] + rz * lateral
        return (cx - rx * half_w, cz - rz * half_w), (cx + rx * half_w, cz + rz * half_w)

    for i in range(len(c.xs) - 1):
        if dashed and (c.cum[i] % period) > spec.DASH_LENGTH_M:
            continue
        (l0x, l0z), (r0x, r0z) = edge(i)
        (l1x, l1z), (r1x, r1z) = edge(i + 1)
        n = len(verts)
        verts += [Vec3(l0x, 0.02, l0z), Vec3(r0x, 0.02, r0z),
                  Vec3(l1x, 0.02, l1z), Vec3(r1x, 0.02, r1z)]
        uvs += [Vec2(0, 0), Vec2(1, 0), Vec2(0, 1), Vec2(1, 1)]
        tris += [n, n + 1, n + 3, n, n + 3, n + 2]
    if not verts:
        verts, tris, uvs = [Vec3(0, -99, 0)], [0, 0, 0], [Vec2(0, 0)]
    return Mesh(vertices=verts, triangles=tris, uvs=uvs, mode="triangle")


def _build_car(holder: Entity, body_color):
    """A blocky but car-shaped vehicle under `holder` (origin at car centre).
    Half-extents ~ (0.9, 0.6, 2.2)."""
    Entity(parent=holder, model="cube", color=body_color,
           scale=(1.8, 0.7, 4.4), position=(0, -0.2, 0))
    Entity(parent=holder, model="cube", color=body_color,
           scale=(1.6, 0.6, 2.2), position=(0, 0.35, -0.2))
    for sx in (-0.95, 0.95):
        for sz in (-1.5, 1.5):
            Entity(parent=holder, model="cube", color=_c(20, 20, 20),
                   scale=(0.25, 0.5, 0.5), position=(sx, -0.55, sz))


def _build_sign(p: course_mod.Placed) -> Entity:
    holder = Entity(position=(p.x, 0, p.z), rotation_y=p.heading_deg)
    Entity(parent=holder, model="cube", color=_c(120, 120, 120),
           scale=(0.08, 2.0, 0.08), position=(0, 1.0, 0))
    if p.kind == "stop_sign":
        tex, w, hgt = textures.stop_sign(), 0.75, 0.75
    else:
        tex, w, hgt = textures.speed_limit(p.value or 35), 0.6, 0.9
    # wrap in Texture(): ursina mis-resolves a bare absolute path string
    panel = Entity(parent=holder, model="quad", texture=Texture(tex), double_sided=True,
                   scale=(w, hgt), position=(0, 2.2, 0.05))
    return panel


# --------------------------------------------------------------------------
# World
# --------------------------------------------------------------------------

@dataclass
class World:
    ego: Entity
    labels: list
    _moving: list  # (holder_entity, Placed)

    def update(self, dt: float):
        for holder, p in self._moving:
            dx, dz = course_mod._dir(holder.rotation_y)
            holder.x += dx * p.speed_mps * dt
            holder.z += dz * p.speed_mps * dt


def build_world(c: course_mod.Course) -> World:
    # grass groundplane sized to the course bounds
    minx, maxx = min(c.xs), max(c.xs)
    minz, maxz = min(c.zs), max(c.zs)
    Entity(model="plane", color=GRASS_COLOR,
           scale=(maxx - minx + 60, 1, maxz - minz + 60),
           position=((minx + maxx) / 2, -0.02, (minz + maxz) / 2))

    Entity(model=_road_mesh(c), color=ROAD_COLOR)

    half = spec.NUM_LANES * spec.LANE_WIDTH_M / 2
    Entity(model=_line_mesh(c, -half, dashed=False), color=LINE_COLOR)
    Entity(model=_line_mesh(c, +half, dashed=False), color=LINE_COLOR)
    for i in range(1, spec.NUM_LANES):
        off = -half + i * spec.LANE_WIDTH_M
        Entity(model=_line_mesh(c, off, dashed=True), color=LINE_COLOR)

    labels, moving = [], []
    ci = 0
    for p in c.objects:
        if p.kind in ("stop_sign", "speed_limit"):
            panel = _build_sign(p)
            if p.class_name:  # stop signs are targets; speed limits are not
                labels.append(LabeledObject(panel, p.class_name, p.extent))
        elif p.kind == "car":
            holder = Entity(position=(p.x, 0.6, p.z), rotation_y=p.heading_deg)
            _build_car(holder, CAR_COLORS[ci % len(CAR_COLORS)])
            ci += 1
            labels.append(LabeledObject(holder, p.class_name, p.extent))
            if p.speed_mps > 0:
                moving.append((holder, p))

    ex, ez, eh = course_mod.ego_start(c)
    ego = Entity(position=(ex, 0, ez), rotation_y=eh)
    return World(ego=ego, labels=labels, _moving=moving)
