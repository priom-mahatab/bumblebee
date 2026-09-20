from pathlib import Path
import shutil

for split in ["train", "val"]:
    label_dir = Path("datasets/av/labels") / split
    src_img_dir = Path("datasets/100k") / split
    dst_img_dir = Path("datasets/av/images") / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    for label_file in label_dir.glob("*.txt"):
        stem = label_file.stem
        src = src_img_dir / f"{stem}.jpg"
        dst = dst_img_dir / f"{stem}.jpg"

        if not src.exists():
            print(f"WARNING: no image for label {label_file.name}")
            missing += 1
            continue

        shutil.copy(src, dst)
        copied += 1

    print(f"{split}: copied {copied}, missing {missing}")
