import cv2
import os

image_dir = "dataset/images"
output_dir = "labels"
os.makedirs(output_dir, exist_ok=True)

max_display_width = 1000  # Görüntü genişliği ekranı geçmesin diye

images = [img for img in os.listdir(image_dir) if img.endswith(".jpg")]

ix, iy = -1, -1
drawing = False
rect = []

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, rect
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = display_img.copy()
            cv2.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Etiketle", img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect = [ix, iy, x, y]
        cv2.rectangle(display_img, (ix, iy), (x, y), (0, 255, 0), 2)
        cv2.imshow("Etiketle", display_img)

for image_name in images:
    image_path = os.path.join(image_dir, image_name)
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # Görüntüyü uygun ölçekte yeniden boyutla
    scale_factor = 1.0
    if w > max_display_width:
        scale_factor = max_display_width / w
        display_img = cv2.resize(img, (int(w * scale_factor), int(h * scale_factor)))
    else:
        display_img = img.copy()

    rect = []
    cv2.namedWindow("Etiketle")
    cv2.setMouseCallback("Etiketle", draw_rectangle)

    print(f"Etiketleniyor: {image_name} - Çiz ve ESC'ye bas")
    cv2.imshow("Etiketle", display_img)

    key = cv2.waitKey(0)
    if key == 27 and rect:  # ESC ile geç
        x1, y1, x2, y2 = rect

        # Koordinatları orijinale geri çevir
        x1, x2 = int(x1 / scale_factor), int(x2 / scale_factor)
        y1, y2 = int(y1 / scale_factor), int(y2 / scale_factor)

        # Normalize YOLO formatı
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = abs(x2 - x1) / w
        height = abs(y2 - y1) / h

        label_name = image_name.replace(".jpg", ".txt")
        with open(os.path.join(output_dir, label_name), "w") as f:
            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

cv2.destroyAllWindows()
