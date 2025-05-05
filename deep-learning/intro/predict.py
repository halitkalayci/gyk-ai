import tensorflow as tf 
from tensorflow.keras.models import load_model
import cv2
import numpy as np

model = load_model("my_model.h5")

img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28,28))
img = 255 - img # 
img = img / 255.0
img_flat = img.reshape(1, 28*28)

prediction = model.predict(img_flat)
predicted_label = np.argmax(prediction)
print(prediction)
print(predicted_label)