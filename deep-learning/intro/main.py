# scikit-learn -> ML kütüphanesi
# pyTorch-tensorFlow -> DL Kütüphaneleri

from tensorflow.keras.datasets import mnist # hazır rakam veri seti
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import tensorflow as tf
import numpy as np

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
# epoch
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# Adaptive Moment Estimation
# 0-9 arası sayısal etiketler için 

# kaç kere veriyi sıfırdan alarak eğitilsin?
model.fit(x_train, y_train, epochs=5, batch_size=64) 

# epoch -> eğitim turu
# batch_size => her turdaki verinin oransal olarak kaçını alacağım 60000/64
# iteration => her batch'in yaptığı bir adımı temsil eder.
# 60.000/64 => 938
# epochs = 5, iteration=938, batch_size=64
test_loss, test_acc = model.evaluate(x_test, y_test) # predict()
print(f"Test doğruluk oranı {test_acc}")


# Rastgele test örneği
# veriyi test etmek için => EĞİTİM NASILSA TEST VERİSİ O FORMATA UYMAK ZORUNDA!

index = np.random.randint(0, len(x_test))
sample = x_test[index].reshape(1,28*28)

prediction = model.predict(sample)
predicted_label = np.argmax(prediction) # argmax => EN yüksek olasılık olarak al.
print(predicted_label)

plt.imshow(x_test[index].reshape(28,28), cmap="gray")
plt.title(f"Tahmin Edilen: {predicted_label} | Gerçek: {y_test[index]}")
plt.axis('off')
plt.show()
