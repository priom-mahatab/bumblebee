from pathlib import Path
import shutil

# Roboflow stop-sign dataset (single class, index 0) -> merge into datasets/av as class 5
SRC = Path("/Users/zmahatab/Downloads/Stop Sign.v2i.yolov8")
DST = Path("datasets/av")
STOP_CLASS_ID = 5

# (roboflow split name, our split name)
SPLITS = [("train", "train"), ("valid", "val")]

for src_split, dst_split in SPLITS:
    src_lbl_dir = SRC / src_split / "labels"
    src_img_dir = SRC / src_split / "images"
    dst_lbl_dir = DST / "labels" / dst_split
    dst_img_dir = DST / "images" / dst_split
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    missing = 0
    for label_file in src_lbl_dir.glob("*.txt"):
        # re-index: set the FIRST token of every line to STOP_CLASS_ID, leave coords untouched
        lines = []
        for line in label_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            parts[0] = str(STOP_CLASS_ID)
            lines.append(" ".join(parts))

        # find the matching image by stem (extension may be .jpg/.jpeg/.png)
        matches = list(src_img_dir.glob(label_file.stem + ".*"))
        if not matches:
            print(f"WARNING: no image for {label_file.name}")
            missing += 1
            continue

        (dst_lbl_dir / label_file.name).write_text("\n".join(lines))
        shutil.copy(matches[0], dst_img_dir / matches[0].name)
        copied += 1

    print(f"{src_split} -> {dst_split}: merged {copied}, missing {missing}")
