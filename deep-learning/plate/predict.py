from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Modeli yükle
model = YOLO("best.pt")

img = cv2.imread("dataset/images/973.jpg")
# source => tahmin yapılacak kaynak (img yolu veya arrayi)
# conf (confident) => güven skoru (0.5 %50'den düşük güven duyduğun tahminleri görmezden gel.)
# verbose => Tahmin detaylarını konsola yazsın mı?
results = model.predict(source=img, conf=0.5, verbose=False)


# 1den fazla tahmin göster? for döngüsü 
boxes = results[0].boxes # Tahminler listesi

annot_img = img.copy()

for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])

    cv2.rectangle(annot_img, (x1,y1), (x2,y2), color=(0,255,0), thickness=2)
    cv2.putText(annot_img, 
                f"Guven Skoru: {conf}", (x1,y1 - 10), 
                fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 
                fontScale=0.5,
                color=(0,255,0), 
                thickness=1)

plt.figure(figsize=(10,7))
plt.imshow(annot_img)
plt.show()