from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="datasets/av/dataset.yaml", epochs=1, imgsz=640)