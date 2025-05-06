import cv2
import matplotlib.pyplot as plt


img = cv2.imread("dataset/images/9.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Gürültü azaltma
blur = cv2.bilateralFilter(gray, 11, 17, 17)
#

# Kenar Algılama 
edged = cv2.Canny(blur, 30, 200)
#

contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

contours = sorted( contours, key=cv2.contourArea, reverse=True )[:10]

plate = None

for c in contours:
    # Çokgenleri bul.
    perimeter = cv2.arcLength(c, True)

    approx = cv2.approxPolyDP(c, 0.02*perimeter, True) #polygon -> 0.02 hassasiyet

    if len(approx) == 4:
        plate = approx
        print("Dikdortgen algılandı.")
        break

if plate is not None:
    cv2.drawContours(img, [plate], -1, (0,255,0), 3)

plt.figure(figsize=(10,6))
plt.imshow(img)
plt.show()