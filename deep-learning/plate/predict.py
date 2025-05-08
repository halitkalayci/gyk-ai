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

img = cv2.imread("dataset/images/758.jpg")
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

    h, w, _ = img.shape

    pad = 30 #her kenarı 10 piksel geniş keselim.

    x1p = max(x1-pad, 0)
    y1p = max(y1-pad, 0)
    x2p = min(x2+pad, w)
    y2p = min(y2+pad, h)

    cropped_plate = img[y1p:y2p, x1p:x2p]

    gray_plate = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2GRAY)

    # Threshold uygula
    _, threshold = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #ocr ile resmi texte döndür.
    # config="--psm 7" -> Tek satırlık bir okuma yapıyosun.
    plate_text = pytesseract.image_to_string(cropped_plate)

    print(f"OCR:{plate_text}")

    #Post-processing -> 34 JH 023] -> ] KALDIRMAK

    plt.figure(figsize=(10,7))
    plt.imshow(cropped_plate)
    plt.show()
    




# OCR => Optical Character Recognition
# 