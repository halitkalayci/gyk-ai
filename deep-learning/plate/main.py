# YoloV8
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(data="data.yaml", epochs=10, imgsz=640, batch=8)

# bu modeli bütün veri setiyle eğitelim.

# modeli nasıl kaydederiz? yeni img ile nasıl test ederiz?