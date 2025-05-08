from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import pytesseract

# opsiyonel bir satır.
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# macos
# /usr/local/bin/tesseract - /opt/homebrew/bin/tesseract

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
    # Her tahminin x1,x2 y1,y2 alanları var
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])


    cropped_plate = img[y1:y2, x1:x2]

    #ocr ile resmi texte döndür.
    # config="--psm 7" -> Tek satırlık bir okuma yapıyosun.
    plate_text = pytesseract.image_to_string(cropped_plate, config="--psm 7")

    print(plate_text)

    plt.figure(figsize=(10,7))
    plt.imshow(cropped_plate)
    plt.show()
    




# OCR => Optical Character Recognition
# 