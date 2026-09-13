# Week 1 Learning Plan — Training, Evaluating & Debugging a CV Model

**Week of 2026-09-09.** Hardware not yet arrived; everything here runs on the
laptop with the `.cv` venv. This week is **learning-first**: the goal is that by
the end I can write the AV's detection/training code myself, understanding what
each part does — not copy-pasting.

## Guiding principle (scope guardrails)

Three tiers, in priority order. Nothing in a lower tier is ever allowed to
become a dependency of a higher one.

1. **Car works (critical path):** pretrained YOLOv8n drives the AV. Never
   blocked by any training work.
2. **Real ML learning (this week's focus):** train my own model properly,
   evaluate it honestly, debug it when it's wrong. This is the graded
   ML-rigor deliverable.
3. **From-scratch mini-YOLO (optional stretch):** only if tiers 1–2 are
   solid and time remains. For learning + the report only, never the demo.

This plan is tier 2. The arc is deliberately: **Train → Evaluate → Debug/Fix.**

---

## Learning objectives

Phrased as checkable outcomes. By end of week I should be able to honestly tick
each one. These are what let me write the AV code without asking for the code
directly.

### A. Training foundations
- [ ] I can explain the training loop conceptually: forward pass → loss →
      backprop → optimizer step, and what each does.
- [ ] I understand where PyTorch sits vs. Ultralytics/YOLO (engine vs. wrapper),
      and can name what runs where.
- [ ] I can write **one** minimal training loop in raw PyTorch (small problem)
      so training is no longer a black box.
- [ ] I can fine-tune YOLOv8n on a chosen dataset with `model.train(...)` and
      explain the key hyperparameters I set (epochs, batch, imgsz, lr).

### B. Object-detection specifics
- [ ] I can explain what a detector outputs (variable # of box+class+confidence)
      and why that's harder than classification.
- [ ] I can compute and explain **IoU** and why it's used everywhere.
- [ ] I understand the grid-prediction idea (YOLO) at a high level.
- [ ] I know what NMS does and why raw model output needs it.

### C. "Is it actually accurate for CV?" (evaluation)
- [ ] I can explain **precision vs. recall** and read a precision-recall curve.
- [ ] I understand **mAP@0.5** and **mAP@0.5:0.95** and can read them off a
      training run.
- [ ] I can read **per-class** metrics and identify which classes are weak.
- [ ] I evaluate only on a **held-out** set, never on training data.
- [ ] I do a **qualitative** check: run the model on real webcam frames and
      look at the boxes, not just the numbers.
- [ ] I can measure **latency / FPS** and reason about whether it's fast enough
      for a real-time AV.

### D. Debugging & fixing a model
- [ ] I can run the **overfit-one-batch** sanity check and explain what it
      proves (that the model + loss + loop are wired correctly).
- [ ] I can **visualize the data pipeline** — draw the boxes on the augmented
      images — to catch the #1 silent bug (box transforms gone wrong).
- [ ] I can read **loss curves** and diagnose overfitting vs. underfitting vs.
      a broken run (loss flat / NaN / exploding).
- [ ] I have a checklist of common failure modes and their fixes: LR too
      high/low, bad/mismatched labels, class imbalance, too little data,
      wrong augmentation, wrong confidence/NMS thresholds.
- [ ] Given a low mAP, I can form a hypothesis, test it, and know what to
      change — instead of randomly tweaking.

### E. Data (the thing everything depends on)
- [ ] I can pick/justify a dataset that includes a **stop-sign** class (since
      BDD100K lacks one) so the trained model keeps the AV's key behavior.
- [ ] I understand the annotation format (YOLO `.txt` / COCO JSON) and the
      train/val split.
- [ ] I understand why detection augmentation must transform boxes too.

---

## Daily schedule

Each day: a focus, concrete tasks (I write the code), and a checkpoint to prove
the objective landed. ~2–3 focused hours/day assumed; compress or stretch as
needed.

### Day 1 — Orient + PyTorch mechanics
- **Focus:** A (foundations).
- Tasks: confirm `.cv` venv + `torch`/`ultralytics` import and see GPU/CPU
  device. Read a short PyTorch tensors/autograd refresher. Write a 20-line
  script: a tensor, a simple function, `.backward()`, inspect `.grad`.
- **Checkpoint:** I can explain what autograd just computed.

### Day 2 — Write one training loop from scratch
- **Focus:** A (demystify training).
- Tasks: train a tiny CNN classifier on CIFAR-10 (or MNIST) with a
  hand-written loop — forward, loss, backward, `optimizer.step()`, eval.
  Watch train/val accuracy move.
- **Checkpoint:** the loop runs and accuracy climbs; I can point to each line
  and say what it does.

### Day 3 — Detection concepts + IoU
- **Focus:** B.
- Tasks: read the YOLOv1 paper (short) for the grid idea. Implement an `iou()`
  function from scratch and test it on a couple of hand-checked boxes.
  Read what NMS does.
- **Checkpoint:** my IoU matches hand calculations; I can explain grid
  prediction and NMS in a sentence each.

### Day 4 — Data: pick dataset + build/inspect the pipeline
- **Focus:** E + D (visualization).
- Tasks: choose a stop-sign-inclusive dataset (e.g. a Roboflow stop-sign set,
  LISA, or BDD100K + a stop-sign source). Get it into YOLO format. Write a
  script that loads a few samples **and draws the boxes on the images** —
  including after augmentation.
- **Checkpoint:** boxes land correctly on objects in the visualized images
  (this is the bug-catcher for the whole week).

### Day 5 — Fine-tune YOLOv8n + the overfit-one-batch check
- **Focus:** A + D.
- Tasks: first run the **overfit-one-batch** sanity check (loss should crash
  toward zero). Then kick off a real fine-tune (`model.train`) with sensible
  epochs/batch/imgsz. Note the hyperparameters and why.
- **Checkpoint:** overfit check passes; a real training run completes and
  produces metrics.

### Day 6 — Evaluate: "is it actually accurate?"
- **Focus:** C.
- Tasks: read mAP@0.5 / mAP@0.5:0.95 and per-class AP off the run. Look at the
  PR curve. Run the model on **live webcam frames** and eyeball the boxes.
  Measure FPS.
- **Checkpoint:** I can state my model's mAP, name its weakest class, say
  whether it's fast enough for real-time, and whether the qualitative output
  matches the numbers.

### Day 7 — Debug/fix + write it up
- **Focus:** D + consolidation.
- Tasks: take the weakest result from Day 6, form a hypothesis (data? LR?
  imbalance? threshold?), make **one** targeted change, retrain/re-eval, and
  compare. Write a short note: what I changed, what happened, why.
- **Checkpoint:** a documented before/after — evidence I can debug, not just
  train.

---

## Common failure-mode cheat sheet (for Day 7 and beyond)

| Symptom | Likely cause | First thing to try |
|---|---|---|
| Loss is NaN / explodes | LR too high, bad data | Lower LR; check for corrupt labels |
| Loss flat, never drops | LR too low, broken loop, frozen weights | Raise LR; run overfit-one-batch |
| Train good, val bad | Overfitting / too little data | More augmentation/data; fewer epochs |
| Train and val both bad | Underfitting / not enough capacity or epochs | Train longer; check data pipeline |
| One class much worse | Class imbalance / few examples | Add examples; check label counts |
| Boxes offset / wrong size | Box transform bug in augmentation | Re-run the Day 4 visualization |
| Lots of false positives | Confidence threshold too low | Raise conf threshold; check NMS |
| Missing obvious objects | Confidence too high / bad recall | Lower conf; inspect PR curve |

---

## End-of-week success criteria

- I trained my own detector, know its mAP, and looked at its predictions on
  real frames.
- When it's wrong, I have a **method** (sanity checks → visualize → read curves
  → hypothesis → one change) instead of guessing.
- The pretrained YOLOv8n path still exists and can drive the car — tier 1 was
  never at risk.
- I'm ready to write the AV's detection glue code myself, understanding each
  piece.
