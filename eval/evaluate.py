"""Reproducible detection-metrics evaluation for the BDD/AV YOLO models.

Runs ultralytics val on the held-out val split and records IoU-based metrics
(the proposal's criteria): mAP@0.50, mAP@0.50:0.95, and precision/recall at
IoU 0.50 — overall and per class. Writes two CSVs under eval/ that
detection_metrics.md is built from, so the table is regenerable, not hand-typed.

Usage (from repo root, inside the .cv venv):
    python -m eval.evaluate                 # evaluate the default model set
    python -m eval.evaluate --models v4     # just one tag

Latency/FPS are measured separately (eval/latency.py, TODO) since those are
device-specific (laptop / Pi / sim) and collected later.
"""
from __future__ import annotations

import argparse
import csv
import os

DATA = "datasets/av/dataset.yaml"

# tag -> (weights, imgsz used at train/val time). v1/v2 weights aren't
# disambiguated locally, so their metrics come from recorded run logs instead
# (see detection_metrics.md); v3/v4 are recomputed here from weights.
MODELS = {
    "v3": ("detection/models/bdd_v3_best.pt", 640),
    "v4": ("detection/models/bdd_v4_960.pt", 960),
}

OVERALL_CSV = "eval/detection_metrics.csv"
PERCLASS_CSV = "eval/detection_metrics_per_class.csv"


def evaluate(tag: str, weights: str, imgsz: int):
    from ultralytics import YOLO

    model = YOLO(weights)
    # plots off and a scratch project dir so a run doesn't clutter the repo.
    res = model.val(data=DATA, imgsz=imgsz, split="val", verbose=False, plots=False,
                    project=os.environ.get("VAL_OUT", "/tmp/av_val"), name=tag, exist_ok=True)
    b = res.box
    names = model.names
    overall = {
        "model": tag,
        "weights": os.path.basename(weights),
        "imgsz": imgsz,
        "mAP50": round(float(b.map50), 4),
        "mAP50_95": round(float(b.map), 4),
        "precision": round(float(b.mp), 4),
        "recall": round(float(b.mr), 4),
    }
    per_class = []
    for i, c in enumerate(b.ap_class_index):
        per_class.append({
            "model": tag,
            "class": names[int(c)],
            "mAP50": round(float(b.ap50[i]), 4),
            "precision": round(float(b.p[i]), 4),
            "recall": round(float(b.r[i]), 4),
        })
    return overall, per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS), help="tags to evaluate")
    args = ap.parse_args()

    os.makedirs("eval", exist_ok=True)
    overall_rows, perclass_rows = [], []
    for tag in args.models:
        weights, imgsz = MODELS[tag]
        print(f"\n=== evaluating {tag} ({weights} @ imgsz {imgsz}) ===")
        o, pc = evaluate(tag, weights, imgsz)
        print("overall:", o)
        for row in pc:
            print("  ", row)
        overall_rows.append(o)
        perclass_rows.extend(pc)

    with open(OVERALL_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(overall_rows[0]))
        w.writeheader()
        w.writerows(overall_rows)
    with open(PERCLASS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(perclass_rows[0]))
        w.writeheader()
        w.writerows(perclass_rows)
    print(f"\nwrote {OVERALL_CSV} and {PERCLASS_CSV}")


if __name__ == "__main__":
    main()
