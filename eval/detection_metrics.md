# Object-Detection Evaluation (BDD100K + stop-sign, YOLOv8n)

IoU-based detection metrics for the fine-tuning runs, per the proposal's criteria:
mean Average Precision at IoU ≥ 0.50 (**mAP@0.50**, primary), mAP averaged over
IoU 0.50–0.95, and **precision/recall at IoU 0.50**. A prediction counts as
correct only when the class is right *and* IoU ≥ 0.50, so localization is scored,
not just recognition.

Model: **YOLOv8n** (3.0 M params, ~8 GFLOPs) fine-tuned on a BDD100K subset.
6 classes: `person, car, truck, bus, traffic sign, stop sign`. The `stop sign`
class is added from a Roboflow close-up set (BDD labels stop signs only as the
generic `traffic sign`). Validation split: **2144 images / 32 208 instances**
(v2–v4); v1 used an earlier 5-class / 2000-image split and is not directly
comparable (see note).

Reproduce (inside the `.cv` venv, from repo root):

```
python -m eval.evaluate            # recomputes v3 & v4 -> eval/*.csv
```

## Model comparison (overall)

Each run changes essentially one variable from the previous, so the gains are
attributable.

| Run | Classes | Train imgs | Epochs | imgsz | mAP@0.50 | mAP@0.50:0.95 | Precision | Recall | Source |
|-----|---------|-----------|--------|-------|----------|---------------|-----------|--------|--------|
| v1  | 5 (no stop sign) | 5 000  | 30 | 640 | 0.477 | 0.280 | 0.601 | 0.434 | run log¹ |
| v2  | 6 | 6 910  | 80 | 640 | 0.578 | 0.397 | 0.691 | 0.541 | run log¹ |
| v3  | 6 | 26 910 | 80 | 640 | 0.635 | 0.437 | 0.714 | 0.587 | recomputed² |
| **v4** | 6 | 26 910 | 80 | **960** | **0.701** | **0.468** | **0.764** | **0.635** | recomputed² |

**v4 is the deployed model** (`detection/models/bdd_v4_960.pt`).

### What each step isolates

- **v1 → v2** — added the `stop sign` class and trained longer (30 → 80 epochs).
  mAP@0.50 0.477 → 0.578. v1's curves were still climbing at 30 epochs
  (undertrained, not overfit), so the extra epochs, not just the new class,
  drove the gain; every original class improved.
- **v2 → v3** — scaled training data 6.9 k → 26.9 k images (same 80 epochs / 640).
  mAP@0.50 0.578 → 0.635, recall 0.541 → 0.587. The biggest gains landed on the
  previously weak, under-represented classes (truck, bus, traffic sign),
  consistent with more data fixing class imbalance and small-object recall.
- **v3 → v4** — raised input resolution 640 → 960, **same data and epochs**.
  mAP@0.50 0.635 → 0.701, recall 0.587 → 0.635. A clean resolution ablation:
  higher resolution helps the small-object recall that was the bottleneck.

Trend across all four: recall rises faster than precision, confirming recall on
small objects — not class confusion — is the limiting factor.

## Per-class mAP@0.50

| Class | v2 | v3 | v4 | v4 Precision | v4 Recall |
|-------|------|------|------|------|------|
| person       | 0.440 | 0.493 | 0.612 | 0.738 | 0.523 |
| car          | 0.688 | 0.730 | 0.792 | 0.803 | 0.712 |
| truck        | 0.458 | 0.550 | 0.593 | 0.671 | 0.536 |
| bus          | 0.411 | 0.491 | 0.551 | 0.661 | 0.448 |
| traffic sign | 0.478 | 0.557 | 0.671 | 0.741 | 0.600 |
| stop sign    | 0.994 | 0.991 | 0.990 | 0.969 | 0.993 |

Per-class precision/recall for v3 and v4 are in
[`detection_metrics_per_class.csv`](detection_metrics_per_class.csv); overall rows
in [`detection_metrics.csv`](detection_metrics.csv). PR curves and confusion
matrices per run are under `docs/train_1..4/` (train_1 = v1 … train_4 = v4).

## Caveats (important for interpreting these numbers)

1. **The `stop sign` ~0.99 is misleading.** Its training/val data are Roboflow
   close-ups (the sign fills most of the frame), so the val set is easy and
   in-distribution. This does **not** show the car will catch small/distant stop
   signs in real street scenes — a domain mismatch a street-scene stop-sign
   dataset would be needed to close.
2. **`traffic sign` vs `stop sign` label conflict (the dangerous failure).** On
   real street photos, actual stop signs were confidently detected but labeled
   `traffic sign` (BDD's label for them), not `stop sign`. The car sees the sign
   but mislabels it, so the stop behavior wouldn't trigger — and mAP@0.50 is
   blind to this, since both are valid classes. Qualitative check:
   `datasets/test_images/pred/`.
3. **Recall on small objects is the bottleneck, not class confusion.** The
   confusion matrices show most misses are "predicted background" (missed
   detections), concentrated on person and traffic sign.
4. **Residual imbalance.** A minority of trucks/buses are still called `car`
   (car dominates instance counts), visible in the confusion matrices.

## Latency / FPS (deliverable — pending)

Inference latency (ms/frame) and FPS are device-specific and collected
separately: laptop and simulation now, Raspberry Pi once the hardware is wired.
To be filled in `eval/latency.py` + a column here next.

---

¹ **run log** — reported from the Colab training run's validation output
(recorded in project notes); the exact v1/v2 checkpoints aren't disambiguated in
this repo, so they are not recomputed here. v1 is on a different (5-class,
2000-image) val split and is included for the progression story, not as a
like-for-like comparison with v2–v4.
² **recomputed** — regenerated in this repo via `eval/evaluate.py`
(`ultralytics` val) on the 2144-image val split; v3's recomputed mAP@0.50
(0.635) matches the recorded/plotted 0.634, cross-validating the method.
