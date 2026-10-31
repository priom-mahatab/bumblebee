# Stop-Sign Distance Estimation — Range & Dead-Reckon

**Written 2026-10-31.** Explains how the car figures out *how far away* a stop
sign is and uses that to stop **at** the sign instead of braking the moment it
sees one. Covers `control/state_machine.py` (`StopSignController`,
`stop_sign_visible`) and the sim's range source in `sim/sim_camera.py`
(`next_stop_sign_distance_m`). Learning-first: the goal is to understand *why*
the approach is built this way, not just what it calls.

The job here is the opposite of lane detection's. Lane detection turns a frame
into a steering number every frame. Stop-sign handling is about **timing a single
event** — one clean STOP, at the right place on the road — from a sign that
won't even be in view by the time we get there.

---

## The problem: why you can't just "stop when you see it"

The obvious approach — "if a stop sign is in the frame, STOP" — stops the car far
too early. The reason is pure geometry, and it's worth internalizing because it's
the whole motivation for everything below.

A stop sign sits at the **roadside**, not in your lane. As you drive toward it:

| When the sign is… | its horizontal position | its size in frame |
|---|---|---|
| **far** (approaching) | near the vanishing point — looks **centered** | tiny (a few px tall) |
| **close** (alongside) | slid out to the **frame edge** | large |

So a roadside sign is **never both centered and close**. Any rule based on the
bounding box alone fails:

- *"Stop when it's centered"* → fires when the sign is far and tiny → stops way
  too early (we saw the car brake ~45 m short).
- *"Stop when it's large (close)"* → by then it's at the frame edge, half
  out of view, and you're already level with it — too late, and unreliable.

The fix is to stop reasoning about *where the sign is in the image* and instead
answer one question once: **how many metres ahead is it?** Then drive that far
and stop — even though the sign disappears from view on the way.

---

## The pipeline at a glance

```
stop sign comes into view
  -> ESTIMATE DISTANCE to it           (how many metres ahead?)
       * real car:  pinhole from the bbox height
       * this sim:  exact along-road arc length (ground truth)
  -> LATCH a stop target               (odometry_now + distance - 1 m)
  -> DEAD-RECKON forward                (sign leaves view; drive on odometry)
  -> STOP when odometry >= target       (lands 1 m before the sign)
  -> HOLD, then CLEAR, then RE-ARM      (ready for the next sign down the road)
```

The first box is the "estimating distance" part. The rest is how that distance
is *used* — because a distance is only useful if you can act on it after the sign
is gone.

---

## Step 1 — Estimate the distance

There are two ways to get the range, and this project uses different ones for the
real car vs. the sim. Understanding both is the point.

### 1a — Monocular pinhole (the real-world CV method)

With a single camera you can't measure depth directly, but you *can* infer it
from **apparent size**, because a known-size object shrinks predictably with
distance. This is the pinhole-camera relation:

```
distance  =  real_height_m * focal_length_px / bbox_height_px
```

- `real_height_m` — a stop sign's true height is a known constant (~0.75 m for a
  US sign panel). This is the "known size" that makes the trick work.
- `focal_length_px` — a property of the lens + resolution:
  `f = (frame_height / 2) / tan(VFOV / 2)`. For our camera (62.2° HFOV, 640×480)
  that's ≈ **530 px**.
- `bbox_height_px` — the detected sign's pixel height **this frame**.

So a sign that measures 20 px tall is ≈ `0.75 * 530 / 20 ≈ 20 m` away; at 40 px
it's ≈ 10 m. **Taller box = closer** — the same monocular distance proxy used for
lead vehicles (`lead_vehicle_closeness`), just solved for actual metres because a
sign's real size is standardized (a car's isn't).

This is the honest computer-vision method and is what a real car on the physical
track would use.

### 1b — The sim's exact distance (what we actually drive on)

In the simulator we deliberately **don't** estimate the range from pixels — we
compute it exactly, because the sim *authored* the world and already knows where
every sign is. This matches the project's perception split: **in the sim, bounding
boxes come from ground truth, not YOLO** (YOLO can't reliably see low-poly sim
signs). Using an exact range too means any mis-stop is a *control* bug, not a
noisy-estimate bug — it isolates what we're testing.

`sim/sim_camera.py`:

```python
def _road_distance_m(self, x, z):
    # arc length along the centreline of the nearest point to (x, z)
    c = self.course
    i = min(range(len(c.xs)), key=lambda k: (c.xs[k]-x)**2 + (c.zs[k]-z)**2)
    return c.cum[i]

def next_stop_sign_distance_m(self):
    ego_s = self._road_distance_m(self.ego.x, self.ego.z)
    best = None
    for o in self.course.objects:
        if o.class_name == "stop sign":
            rem = self._road_distance_m(o.x, o.z) - ego_s
            if rem > 0:                       # only signs still ahead
                best = rem if best is None else min(best, rem)
    return best
```

Both the ego and each sign are projected onto the road's centreline to get their
**arc-length position** along the road; the remaining distance is the difference.
Why arc length matters is the next step.

---

## Step 2 — Why arc length, not straight-line distance (curves)

This is the subtle part, and it's why the doc exists.

The pinhole method (1a) gives you the **straight-line** (as-the-crow-flies)
distance from the camera to the sign. But the car doesn't travel in a straight
line to the sign — it follows the **road**, which may **curve** between here and
there. On a bend, the actual driving distance (the **arc length**) is *longer*
than the straight-line chord:

```
        sign •
            /|
  chord   /  |  (straight-line distance — what pixels give you)
        /    |
      /      |
   car ~~~~~~~  (arc length — what you actually drive)
      (road curves)
```

If you estimate the straight-line distance and then drive that many metres, you
**stop short** of the sign on any curve. To stop in the right place you need the
distance measured the same way you measure your own travel — as distance *along
the road*.

The sim's `next_stop_sign_distance_m` returns arc length directly (it walks the
centreline), so curve-awareness is **free**: both the range and the car's
odometry are arc lengths, so comparing them is apples-to-apples. (On the real
car, you'd correct the pinhole chord using the road curvature estimated from lane
detection — a known refinement, not needed in the sim.)

---

## Step 3 — Latch the target, then dead-reckon

Here's the core idea the problem statement forced on us: **the sign leaves the
frame before you reach it**, so you can't keep measuring it. You measure it
*once*, commit to a stopping point, and then drive to that point blind — using
**odometry** (how far you've driven), not the camera.

`control/state_machine.py` runs this as a small state machine with four phases:

```
CRUISING  --(sign seen + ranged, debounced)-->  APPROACHING
APPROACHING  --(odometry reaches target)------>  STOPPED
STOPPED      --(held long enough)-------------->  CLEARING
CLEARING     --(sign has left view)------------>  CRUISING   (re-arms next sign)
```

The latch, inside `StopSignController.update`:

```python
if self.phase is Phase.CRUISING:
    if armed and stop_sign_distance_m is not None:
        self.stop_target_m = odometry_m + stop_sign_distance_m - STOP_LINE_MARGIN_M
        self.phase = Phase.APPROACHING
elif self.phase is Phase.APPROACHING:
    if odometry_m >= self.stop_target_m:          # dead-reckon: no camera needed
        self.phase = Phase.STOPPED
        self.stopped_at = now
```

- `odometry_m` — how far the car has driven (arc length). In the sim this is
  `veh.traveled`; on the real car it'd be wheel encoders.
- `stop_sign_distance_m` — the range from Step 1, supplied by perception.
- The sign only has to be **visible** to arm, not centered or any particular size:
  `stop_sign_visible` just answers "do we see one?". Ranging handles the distance;
  the bbox position is irrelevant.

Why this is robust: after the latch, the decision depends only on odometry, which
keeps counting whether or not the sign is still in frame. The car can lose sight
of the sign completely and still stop in exactly the right place.

### Debounced arming

The sign must be seen for `STOP_SIGN_ARM_FRAMES` (3) consecutive frames before we
latch, and be gone for `STOP_SIGN_CLEAR_FRAMES` (3) before the latch resets. A
single spurious or dropped detection can't flip the phase or trigger a phantom
stop.

---

## Step 4 — Stop 1 m before, hold, and re-arm

```python
STOP_LINE_MARGIN_M = 1.0   # stop this far BEFORE the sign's along-road position
```

We subtract a 1 m margin at latch time so the car halts just **before** the sign
(at a notional stop line), not level with it. The arithmetic is self-correcting:
at the latch frame, `target = odometry + range - 1`, so when `odometry` reaches
`target`, the car has closed all but 1 m of the gap — it stops 1 m short no matter
how early or late the sign was first ranged.

Once `STOPPED`, the controller forces STOP and holds for `STOP_HOLD_SECONDS`, then
goes to `CLEARING` (drives on, ignoring this sign) and finally back to `CRUISING`,
which resets `stop_target_m` so the **next** stop sign down the road arms fresh.
The `CLEARING → CRUISING` reset waits until the sign has been out of view for 3
frames, so the sign we just serviced can't immediately re-arm a second stop.

---

## How it wires together (code map)

```
sim/run_sim.py        controller.update(..., odometry_m=veh.traveled,
                                        stop_sign_distance_m=src.next_stop_sign_distance_m())
      |                         |
      |  range (arc length)     |  odometry (arc length)
      v                         v
sim/sim_camera.py        control/state_machine.py
  next_stop_sign_distance_m()    StopSignController   (phases, latch, margin)
  _road_distance_m()             stop_sign_visible()  (presence only)
```

Note that `decide_action` — the stateless per-frame lane/obstacle logic — **does
not stop for stop signs at all** anymore. All stop-sign timing lives in the
stateful `StopSignController`, keeping `decide_action` simple and unit-testable.

---

## Real-car port

The interface is the same on hardware; only the two inputs change source:

| Input | Sim | Real car |
|---|---|---|
| `stop_sign_distance_m` | exact along-road arc length from the course | pinhole from the bbox height (Step 1a), curvature-corrected |
| `odometry_m` | `veh.traveled` (integrated commanded speed) | wheel encoders |

The controller logic — latch, dead-reckon, 1 m margin, hold, re-arm — is
identical either way.

---

## Summary of known limits (what to improve later)

| Problem | Cause | Fix direction |
|---|---|---|
| Real-car range is noisy when the sign is tiny | pinhole error ∝ 1/bbox_height; a few-px sign gives a bad estimate | only latch once the bbox is big enough to range reliably; vote over frames |
| Straight-line range under-shoots on curves | pixels give the chord, not the arc | curvature-correct using the lane-detection fit (sim avoids this with arc-length GT) |
| Odometry drift over a long dead-reckon | integrating speed/encoders accumulates error | latch closer to the sign (shorter dead-reckon); re-range if the sign stays visible |
| Assumes a known sign size | pinhole needs `real_height_m` | fine for standardized stop signs; wouldn't generalize to unknown objects |
| One stop sign serviced at a time | single-latch state machine | fine for a track; a dense sign environment would need per-sign tracking |
