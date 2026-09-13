# Simulated worlds

`track.wbt` — a two-lane road with a dashed centre line, US speed limit
signs, and a 4-way stop. Built so lane detection and YOLO stop-sign
detection have something to run against before the physical track exists.

## What you need to view it

[Webots](https://cyberbotics.com/) (R2025a). It's a single native macOS
app — no ROS, no Docker, no build step.

The Homebrew cask was **disabled on 2026-09-01** (it stopped passing the
macOS Gatekeeper check), so `brew install --cask webots` won't work.
Download the `.dmg` directly instead:

```
https://github.com/cyberbotics/webots/releases/download/R2025a/webots-R2025a.dmg
```

Drag `Webots.app` to `/Applications`. If macOS refuses to open it, either
right-click → Open (once), or clear the quarantine flag:

```bash
xattr -dr com.apple.quarantine /Applications/Webots.app
```

**First load needs an internet connection.** The macOS build ships almost
no PROTO assets locally (exactly one `.proto` file is in the app bundle) —
roads, signs, floor and backgrounds are all fetched from
`raw.githubusercontent.com` on demand and then cached. That's why every
`EXTERNPROTO` and every `signImage` in `track.wbt` is a full pinned
`https://…/R2025a/…` URL rather than a relative or `webots://` path; the
latter silently resolves to nothing and the nodes just don't appear.
Webots' own sample worlds use the same full-URL convention.

Then `File → Open World…` and pick `sim/worlds/track.wbt`. Webots treats
the parent of `worlds/` as the project root, so it will look for
controllers in `sim/controllers/` — that directory doesn't exist yet and
isn't needed to just view the scene.

Why Webots and not Gazebo: Gazebo runs on macOS via Homebrew, but the
server and GUI have to be launched as separate processes and camera
sensor rendering has long-standing problems there — which matters,
because this project is entirely camera-driven.

## Layout

Webots' ENU convention: **+x east, +y north, +z up.**

```
                        north arm
                            |
   x=-4 ────── main road ───┼─── 4-way stop ──── east arm
              (25.5 m)      |     centre x=30      ends x=38.5
                        south arm
```

- Road width 7 m, 2 lanes, dashed white centre line at `y = 0`
  - eastbound (+x) lane: `y ∈ [-3.5, 0]`, centre `y = -1.75`
  - westbound (−x) lane: `y ∈ [0, 3.5]`, centre `y = +1.75`
- Sidewalks at `|y| = 3.5 … 4.3`, 0.15 m high
- Signs at `|y| = 4.8`, just clear of the sidewalk
- Painted stop bars at each of the four approaches

| Object            | Position (x, y) | Faces |
|-------------------|-----------------|-------|
| Speed limit 35    | 2, −4.8         | −x    |
| Speed limit 25    | 14, −4.8        | −x    |
| Speed limit 35    | 10, 4.8         | +x    |
| Stop — west appr. | 26, −4.8        | −x    |
| Stop — east appr. | 34, 4.8         | +x    |
| Stop — north appr.| 25.2, 5         | +y    |
| Stop — south appr.| 34.8, −5        | −y    |

Signs face +x by default, so one governing eastbound traffic is rotated
π about z to face the oncoming driver.

## Tuning

- **Speed limit values** — change `signImage` on the `SpeedLimitPanel`.
  The stock textures are `speed_limit_5.jpg` through `speed_limit_80.jpg`
  (in 5 mph steps) under `.../objects/traffic/protos/textures/signs/us/`.
- **Road length** — `length` on `StraightRoadSegment`. The intersection's
  arms reach 8.5 m out from its centre (3.5 m half-width + 5 m stub), so
  if you move either one, keep `road_end == intersection_x − 8.5` or a
  seam will open up between them.
- **Lane markings** — the `lines` field takes one `RoadLine` per divider
  (`numberOfLanes − 1` of them); `type` is `dashed`, `continuous`,
  `double`, or `none`.

## Not here yet

There's no robot in the world, so it's a static scene — enough to look
at and to shoot reference frames from, but nothing drives. Adding a
camera-equipped robot plus a Webots controller that hands frames to
`vision/lane_detection.py` and `detection/infer.py` is the next step.

One thing to know before wiring that up: COCO has a `stop sign` class, so
pretrained YOLOv8n should pick the stop signs out of this world directly.
It has **no** speed-limit class — the speed limit signs are scene realism
and lane-detection context, not something the pretrained detector will
label.
