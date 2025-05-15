from ultralytics import YOLO
import cv2
import pytesseract
import matplotlib.pyplot as plt
import re

# (Opsiyonel) Windows için tesseract yolunu belirt
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Modeli yükle
model = YOLO("best.pt")

# Görseli yükle
img = cv2.imread("dataset/images/1.jpg")

# YOLO tahmini
results = model.predict(source=img, conf=0.5, verbose=False)

boxes = results[0].boxes
annot_img = img.copy()

for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    h, w, _ = img.shape

    pad = 30  # Kenarlardan geniş kırp
    x1p = max(x1 - pad, 0)
    y1p = max(y1 - pad, 0)
    x2p = min(x2 + pad, w)
    y2p = min(y2 + pad, h)

    cropped = img[y1p:y2p, x1p:x2p]

    # --- Preprocessing ---
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    scaled = cv2.resize(inverted, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- OCR ---
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(thresh, config=config)
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())

    print(f"OCR: {cleaned}")

    # --- Görselleştir ---
    plt.figure(figsize=(12, 6))
    plt.imshow(thresh, cmap='gray')
    plt.title(f"OCR: {cleaned}")
    plt.axis("off")
    plt.show()
