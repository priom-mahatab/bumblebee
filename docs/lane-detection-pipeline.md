# Lane Detection Pipeline — Canny, Masking & Hough

**Written 2026-10-25.** Explains the classical computer-vision pipeline behind
`vision/lane_detection.py` (`compute_lane_offset`) and the color-masking
experiment in `detection/hls_mask.py`. Learning-first: the goal is to understand
*why* each step is there, not just what it calls.

The job of the whole pipeline is to turn a raw road frame into a single number —
the **lane-center offset** — that tells the car how far left/right the lane
center sits from where the camera is pointing. The control logic
(`control/state_machine.py`) steers on that number.

---

## The pipeline at a glance

```
raw frame
  -> grayscale          (drop color, keep brightness)
  -> Gaussian blur      (kill pixel noise so edges are clean)
  -> Canny edges        (find brightness boundaries)
  -> ROI mask           (keep only the road-shaped region)
  -> HoughLinesP        (turn edge pixels into line segments)
  -> slope split        (left lane vs right lane)
  -> line fit           (one clean line per side)
  -> offset             (lane center vs frame center -> a number)
```

Everything up to Hough is about producing a *clean binary edge image*. Hough and
after is about *turning those edges into geometry and then a steering number*.

---

## Step 1 — Grayscale + Gaussian blur

```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
```

- **Grayscale**: edge detection works on brightness, not color, so we collapse
  the 3 color channels into 1. Faster and simpler.
- **Gaussian blur**: Canny is a derivative operator — it reacts to sharp
  brightness changes. Raw camera frames have per-pixel noise that looks like
  thousands of tiny sharp changes. Blurring averages each pixel with its
  neighbors (the `(5, 5)` is the kernel size — a 5x5 neighborhood), smoothing
  that noise so Canny only fires on *real* edges, not grain.

## Step 2 — Canny edge detection

```python
edges = cv2.Canny(blur, 50, 150)
```

Canny outputs a binary image: white where there's a strong brightness boundary,
black everywhere else. The two numbers are the **hysteresis thresholds**:

- Any gradient **above 150** is definitely an edge.
- Anything **below 50** is definitely not.
- Between 50 and 150 counts *only if it connects to a strong edge* — this is what
  gives Canny clean, continuous lines instead of dotted fragments.

**Canny's weakness (important):** it keys purely on brightness change. A *shadow
edge* on the road looks identical to a *lane-paint edge*. In dappled sunlight
(see `winding_turns_challenge.mp4`), shadows generate so many fake edges that the
real lane lines drown. That weakness is the reason for the color mask below.

## Step 3 — Region of Interest (ROI) mask

```python
h, w = edges.shape
mask = np.zeros_like(edges)                 # all-black image, same size
polygon = np.array([[
    (int(0.10 * w), int(0.82 * h)),         # bottom-left
    (int(0.95 * w), int(0.82 * h)),         # bottom-right
    (int(0.58 * w), int(0.60 * h)),         # top-right
    (int(0.42 * w), int(0.60 * h)),         # top-left
]], dtype=np.int32)
cv2.fillPoly(mask, polygon, 255)            # white trapezoid on black
masked = cv2.bitwise_and(edges, mask)       # keep edges inside trapezoid only
```

The road ahead occupies a **trapezoid**: wide at the bottom (near the car),
narrowing toward the horizon. Everything outside it — sky, trees, signs, other
cars — produces edges we don't want. The ROI is a black image with a white
trapezoid painted on it; `bitwise_and` keeps an edge pixel only where the mask is
white, zeroing out everything else.

The corner fractions are expressed as fractions of width/height so the same
polygon scales to any frame size.

**Tuning caveat we hit:** the trapezoid above is tuned to the original dashcam
test images, where the road sat higher in the frame. On the `winding_turns` clip
the camera points lower, so the `0.82*h` bottom edge **cropped off the lane lines
near the frame bottom** (including a sharp right-curve line). ROI geometry has to
be re-tuned per camera / per footage. This is a parameter to revisit against the
actual demo clips.

## Step 3b — HLS color masking (the robustness upgrade, experimental)

Prototyped in `detection/hls_mask.py`, not yet merged into `lane_detection.py`.
The idea: instead of (or in addition to) "find brightness edges," also "find
lane-*colored* pixels," which survives shadow far better.

```python
hls    = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
white  = cv2.inRange(hls, (0, 200,   0), (179, 255, 255))   # bright -> white paint
yellow = cv2.inRange(hls, (15,  30, 115), (35,  255, 255))   # yellow hue + saturation
color_mask = cv2.bitwise_or(white, yellow)
```

HLS = **H**ue, **L**ightness, **S**aturation. In OpenCV, H is 0–179, L and S are
0–255. `cv2.inRange(src, lo, hi)` returns white where *every* channel is within
`[lo, hi]`.

- **White lines** -> high **L** (`>= 200`), any hue/saturation. Lane paint is
  bright; shadowed asphalt drops below 200 and falls out.
- **Yellow lines** -> hue in the yellow band (~15–35) with decent **S**. Shadows
  change lightness but don't invent saturated yellow, so yellow paint survives
  dappling.

### Combining color mask with Canny

```python
combined = cv2.bitwise_and(edges, color_mask)
```

Color mask alone and Canny alone fail differently:

- **Canny** drowns in shadow edges.
- **Color mask** grabs sunlit asphalt as a solid white *blob* (bare asphalt in
  direct sun is nearly as bright as paint).

`bitwise_and` makes them cover each other: a lane line is **both** lane-colored
**and** an edge, but the sunlit blob is lane-colored while its *interior* has no
edges. ANDing drops the blob interior and keeps the thin painted lines. Then the
ROI from Step 3 is applied on top of `combined` to remove foliage above the road.

**Limit found:** this helps *shadow* frames but **cannot** rescue true *glare*
frames (e.g. windshield reflection) — there the sensor has already blown the
pixels out, so the lane information is simply gone.

## Step 4 — HoughLinesP (edges -> line segments)

```python
lines = cv2.HoughLinesP(masked, 2, np.pi / 180, 50,
                        minLineLength=40, maxLineGap=100)
```

Canny gives us *edge pixels*; Hough connects collinear pixels into actual **line
segments** `(x1, y1, x2, y2)`. The arguments:

- `rho=2` — distance resolution of the accumulator, in pixels. How finely lines
  are distinguished by position. 2 px is a good speed/accuracy balance.
- `theta=np.pi/180` — angle resolution, in radians. `pi/180` = 1 degree.
- `threshold=50` — minimum number of edge-pixel "votes" for a line to count.
  Higher = fewer, more confident lines.
- `minLineLength=40` — segments shorter than 40 px are discarded (kills little
  specks).
- `maxLineGap=100` — gaps up to 100 px along a line are bridged, so a dashed lane
  marking is joined into one segment instead of many.

`HoughLinesP` returns `None` when it finds nothing — which propagates up as a
lost lane (`offset = None`). On this project's OpenCV (5.0.0), the result shape is
`(N, 4)`, so the code uses `lines.reshape(-1, 4)` to iterate robustly across
versions.

## Step 5 — Split into left vs right lane by slope

```python
for x1, y1, x2, y2 in lines.reshape(-1, 4):
    if x2 == x1:
        continue                       # vertical: undefined slope, skip
    slope = (y2 - y1) / (x2 - x1)
    if abs(slope) < 0.5:
        continue                       # too flat to be a lane line, skip
    (left if slope < 0 else right).append((x1, y1, x2, y2))
```

The two lane lines converge toward the horizon, so they have opposite slopes.
**Note image coordinates: y increases downward.** So the left lane line runs
down-to-the-left (**negative** slope) and the right lane line runs
down-to-the-right (**positive** slope). The `abs(slope) < 0.5` filter throws out
near-horizontal segments (crosswalks, shadows, car bumpers) that aren't lanes.

## Step 6 — Fit one clean line per side

```python
def _fit_lane_line(segments, y_bottom, y_top):
    xs, ys = [], []
    for x1, y1, x2, y2 in segments:
        xs += [x1, x2]; ys += [y1, y2]
    m, b = np.polyfit(ys, xs, 1)       # x = m*y + b  (x as a function of y)
    return (int(m*y_bottom + b), y_bottom), (int(m*y_top + b), y_top)
```

Each side has several noisy segments; `np.polyfit(..., 1)` fits one straight line
through all their endpoints (least-squares). We fit **x as a function of y**
(not the usual y = f(x)) because lane lines are steep/near-vertical, where
`x = m*y + b` is numerically stable and `y = m*x + b` would blow up. We then
evaluate that line at a fixed bottom and top y to get two clean endpoints.

**Limit:** a straight-line fit cannot follow a *curve* — on a curved road the fit
averages to a near-straight line, so the offset reads ~0 even mid-curve. Curves
need a higher-order fit (perspective warp + polynomial / sliding window); that's
a future upgrade, not in the current code.

## Step 7 — Compute the offset

```python
lane_center_x  = (left_x + right_x) / 2     # midpoint of the two lanes at the bottom
frame_center_x = w / 2                       # where the camera points
return (lane_center_x - frame_center_x) / frame_center_x
```

Take the midpoint between the two fitted lane lines at the bottom of the image
(nearest the car) and compare it to the image center. Dividing by half the frame
width **normalizes** the result to roughly `[-1, 1]`:

- **negative** -> lane center is left of the car -> steer left
- **~0** -> centered -> go straight
- **positive** -> lane center is right of the car -> steer right

If either lane line is missing, the function returns `None` — "I can't see the
lane," which must be handled honestly downstream (see below).

---

## How it feeds the car

`control/state_machine.py` reads the offset:

- `offset > +0.08` (`LANE_OFFSET_DEADZONE`) -> `TURN_RIGHT`
- `offset < -0.08` -> `TURN_LEFT`
- within the deadzone -> `FORWARD`

**Critical gotcha:** a lost lane returns `None`, not a number. `None` must *not*
be silently turned into `0.0`, because `0.0` means "perfectly centered -> drive
forward" — i.e. a blind car would drive straight. The honest behavior is for a
lost lane to produce a cautious action (`SLOW`). (This is the `None`-handling fix
tracked for the pipeline.)

---

## Summary of known limits (what to improve later)

| Problem | Cause | Fix direction |
|---|---|---|
| Lane lost in shadow | Canny fires on shadow edges | HLS color mask + AND with Canny |
| Lane lost in glare | Sensor pixels blown out | Temporal smoothing; can't recover the frame itself |
| Lines cropped at bottom | ROI trapezoid mistuned for this camera | Re-tune trapezoid per footage |
| Offset ~0 on curves | Straight-line fit can't bend | Perspective warp + polynomial fit |
| Flicker / single dropouts | Per-frame, no memory | Temporal smoothing across frames |
