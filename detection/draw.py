from pathlib import Path
import cv2

label_path = Path("datasets/av/labels/train/0a0d6e22-f6446b68.txt")
img_path = Path("datasets/100k/train/0a0d6e22-f6446b68.jpg")

with open(label_path) as f:
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)
    
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        cid = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])

        cx_px = cx * 1280
        cy_px = cy * 720
        bw = w * 1280
        bh = h * 720
        x1 = cx_px - bw/2
        y1 = cy_px - bh/2
        x2 = cx_px + bw/2
        y2 = cy_px + bh/2

        cv2.rectangle(img,
                      (int(x1), int(y1)),
                      (int(x2), int(y2)),
                      (0, 255, 0),
                      2
        )

    cv2.imwrite("check.jpg", img)