# YoloV8
from ultralytics import YOLO
# yolov8n => bir computer vision modeli

def train():
    model = YOLO("yolov8n.pt")
    model.train(data="data.yaml", epochs=10, imgsz=640, batch=8)

# Pythonda starndart koruma yapısı.
# python {main.py} => bu kodu çalıştırır.
# ama diğer dosyalar import ettiğinde burayı çalıştırmaz.
if __name__ == '__main__':
    train()

# bu modeli bütün veri setiyle eğitelim.

# modeli nasıl kaydederiz? yeni img ile nasıl test ederiz?

#GPU => Ekran kartı
#CPU => işlemci