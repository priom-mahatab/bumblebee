"""SimCameraSource: the simulator presented as a camera, exposing vision/camera.py's read()/release() contract plus ground-truth boxes, an ultrasonic distance stand-in, and world/ego handles for the control loop."""
from __future__ import annotations

import cv2
import numpy as np
from panda3d.core import GraphicsOutput, Point2, Point3, Texture, loadPrcFileData

from . import course as course_mod
from . import spec
from .world import build_world


class SimCameraSource:
    def __init__(self, headless: bool = False, min_length: float | None = None):
        # PRC settings must be set before the Ursina/ShowBase app is created.
        loadPrcFileData("", f"win-size {spec.CAM_WIDTH} {spec.CAM_HEIGHT}")
        loadPrcFileData("", "window-type offscreen" if headless else "window-type onscreen")
        loadPrcFileData("", "audio-library-name null")
        loadPrcFileData("", "window-title av-comp-vis 3D sim")

        from ursina import Entity, Ursina, camera  # imported here so PRC applies first

        from .shaders import mac_unlit_shader

        self.app = Ursina(borderless=False, development_mode=False)
        self._camera = camera
        # Flat GLSL-120 unlit shader: ursina's 130/140 shaders fail on macOS and the normal-less meshes would shade black anyway. Entities must NOT pass unlit=True (re-attaches the 130/140 shader).
        Entity.default_shader = mac_unlit_shader

        self.course = course_mod.build_course(min_length=min_length)
        self.world = build_world(self.course)
        self.ego = self.world.ego

        camera.parent = self.ego
        camera.position = (0, spec.CAM_HEIGHT_M, spec.CAM_FORWARD_OFFSET_M)
        camera.rotation = (spec.CAM_PITCH_DEG, 0, 0)  # +x = nose down
        self._lens = self.app.camLens
        self._lens.setFov(spec.CAM_HFOV_DEG, spec.vertical_fov_deg())
        self._lens.setNearFar(0.1, 400)

        self._tex = Texture()
        self.app.win.addRenderTexture(self._tex, GraphicsOutput.RTMCopyRam)
        self.app.graphicsEngine.renderFrame()  # realize the buffer once

    def read(self) -> np.ndarray | None:
        self.app.graphicsEngine.renderFrame()
        ram = self._tex.getRamImageAs("BGR")
        if not ram:
            return None
        w, h = self.app.win.getXSize(), self.app.win.getYSize()
        frame = np.frombuffer(bytes(ram), np.uint8).reshape(h, w, 3)[::-1]
        if (w, h) != (spec.CAM_WIDTH, spec.CAM_HEIGHT):
            frame = cv2.resize(frame, (spec.CAM_WIDTH, spec.CAM_HEIGHT))
        return np.ascontiguousarray(frame)

    def release(self):
        try:
            self.app.userExit()
        except Exception:
            pass

    def _project(self, world_pt) -> tuple[float, float, float] | None:
        """World point -> (px, py, forward_m). None if behind the camera."""
        rel = self.app.cam.getRelativePoint(self.app.render,
                                            Point3(world_pt[0], world_pt[1], world_pt[2]))
        film = Point2()
        if not self._lens.project(rel, film):
            return None
        px = (film.x * 0.5 + 0.5) * spec.CAM_WIDTH
        py = (1 - (film.y * 0.5 + 0.5)) * spec.CAM_HEIGHT
        return px, py, rel.y  # rel.y = forward distance in camera space

    def ground_truth(self) -> list[tuple[str, tuple[float, float, float, float]]]:
        """[(class_name, (x1, y1, x2, y2)), ...] for targets visible this frame."""
        out = []
        for obj in self.world.labels:
            hx, hy, hz = obj.extent
            xs, ys = [], []
            for sx in (-hx, hx):
                for sy in (-hy, hy):
                    for sz in (-hz, hz):
                        world = self.app.render.getRelativePoint(obj.entity, Point3(sx, sy, sz))
                        p = self._project((world.x, world.y, world.z))
                        if p is None:
                            continue
                        xs.append(p[0]); ys.append(p[1])
            if len(xs) < 2:
                continue
            x1, x2 = max(0, min(xs)), min(spec.CAM_WIDTH, max(xs))
            y1, y2 = max(0, min(ys)), min(spec.CAM_HEIGHT, max(ys))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            out.append((obj.class_name, (x1, y1, x2, y2)))
        return out

    def nearest_obstacle_distance_cm(self) -> float | None:
        """Forward distance (cm) to the nearest car/person roughly in the ego path (ultrasonic stand-in), or None if nothing is ahead."""
        best = None
        for obj in self.world.labels:
            if obj.class_name not in ("car", "person", "truck", "bus"):
                continue
            wp = obj.entity.world_position
            p = self._project((wp[0], wp[1], wp[2]))
            if p is None or p[2] <= 0:
                continue
            px_frac = p[0] / spec.CAM_WIDTH
            if 0.35 < px_frac < 0.65:
                d = p[2] * 100
                best = d if best is None else min(best, d)
        return best

    def _road_distance_m(self, x: float, z: float) -> float:
        """Arc length along the centreline of the point nearest to (x, z)."""
        c = self.course
        i = min(range(len(c.xs)),
                key=lambda k: (c.xs[k] - x) ** 2 + (c.zs[k] - z) ** 2)
        return c.cum[i]

    def next_stop_sign_distance_m(self) -> float | None:
        """Curve-aware along-road (arc-length) distance from the ego to the next stop sign ahead, or None if none remain."""
        ego_s = self._road_distance_m(self.ego.x, self.ego.z)
        best = None
        for o in self.course.objects:
            if o.class_name == "stop sign":
                rem = self._road_distance_m(o.x, o.z) - ego_s
                if rem > 0:
                    best = rem if best is None else min(best, rem)
        return best

    def step_world(self, dt: float):
        self.world.update(dt)
