from pathlib import Path
import json

CATMAP = {
    "person": 0,
    "car": 1,
    "truck": 2,
    "bus": 3,
    "traffic sign": 4
}

training_labels_data_folder = Path("datasets/100k_labels/train")
validation_labels_data_folder = Path("datasets/100k_labels/val")
training_output_dir = Path("datasets/av/labels/train")
val_output_dir = Path("datasets/av/labels/val")

def convert_bdd_to_yolo(input_folder, output_folder, limit):
    output_folder.mkdir(parents=True, exist_ok=True)
    files = [f for f in input_folder.rglob("*.json") if f.is_file()]
    for json_file in files[:limit]:
        with open(json_file, 'r') as file:
            d = json.load(file)
            name = d["name"]
            objs = d["frames"][0]["objects"]
            lines = []
            for o in objs:
                if "box2d" not in o: continue
                if o["category"] not in CATMAP: continue
                cid = CATMAP[o["category"]]
                b = o["box2d"]

                cx = ((b["x1"]+b["x2"])/2) / 1280
                cy = ((b["y1"]+b["y2"])/2) / 720
                w  = (b["x2"]-b["x1"]) / 1280
                h  = (b["y2"]-b["y1"]) / 720
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            with open(output_folder / f'{name}.txt', 'w') as out_file:
                out_file.write('\n'.join(lines))

if __name__ == "__main__":
    convert_bdd_to_yolo(training_labels_data_folder, training_output_dir, limit=5000)
    convert_bdd_to_yolo(validation_labels_data_folder, val_output_dir, limit=2000)