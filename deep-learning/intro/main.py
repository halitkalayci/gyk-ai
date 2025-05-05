# scikit-learn -> ML kütüphanesi
# pyTorch-tensorFlow -> DL Kütüphaneleri

from tensorflow.keras.datasets import mnist # hazır rakam veri seti
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def img_show_save():
    img = x_train[0]
    label = y_train[0]

    plt.imsave(f"{label}.png", img, cmap="gray")
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"{label} numarası img olarak kaydedildi.")
#
print(x_train.shape)
x_train = x_train.reshape(-1, 28 * 28) / 255.0 # Flattening
x_test = x_test.reshape(-1, 28*28) / 255.0
# 2x2 => [[5, 10]
#         [15, 20]]

# [5,10,15,20]

print(x_train.shape)