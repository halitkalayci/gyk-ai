# scikit-learn -> ML kütüphanesi
# pyTorch-tensorFlow -> DL Kütüphaneleri

from tensorflow.keras.datasets import mnist # hazır rakam veri seti
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = mnist.load_data()

def img_show_save():
    img = x_train[0]
    label = y_train[0]

    plt.imsave(f"{label}.png", img, cmap="gray")
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"{label} numarası img olarak kaydedildi.")


x_train = x_train.reshape(-1, 28 * 28) / 255.0 # Flattening
x_test = x_test.reshape(-1, 28*28) / 255.0
# 2x2 => [[5, 10]
#         [15, 20]]
# [5,10,15,20]

# Giriş verisi
# Gizli katman (nöron)
# Çıkış katmanı (0-9 arası 10 rakamın olasılığı.)
model = models.Sequential([
    tf.keras.Input(shape=(784,)), # Girdi verisi 784 boyutunda (28*28)
    layers.Dense(64, activation="relu"), # Gizli katman - 64 -> Nöron sayısı # Activation'a geleceğiz.
    layers.Dense(10, activation="softmax") # Çıkış katmanı - 10 olasılık var..
])
# 64 tane yapay nöron var. z = w1x1 + w2x2 + ..... w6000x6000

# model compile

#