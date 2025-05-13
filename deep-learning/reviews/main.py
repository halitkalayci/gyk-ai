# ip: 193.203.191.79
# port: 32001
# dbname: testdb
# user: postgres
# password: 1234

import psycopg2
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


connection = psycopg2.connect(
    dbname="testdb",
    user="postgres",
    password="1234",
    host="193.203.191.79",
    port= "32001"
)

query = "SELECT * FROM users" # users tablosundaki verileri oku.
df = pd.read_sql(query, connection)
connection.close()

texts = df['review'].values
ratings = df['rating'].values
#print(texts)
#print(ratings)
X_train, X_test, y_train, y_test = train_test_split(texts,ratings,test_size=0.3, random_state=42)

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=50, padding="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=50, padding="post")

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=50), # kelimeleri vektörlere çevirir
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(32, activation='relu'), # f(x) = max(0,x)
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation='linear'), # hiçbir aktivasyon uygulanmasın demek. f(x)=x
])
# Huber Loss => MSE+MAE karışımı gibi.
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

history = model.fit(X_train_pad, y_train, epochs=1, validation_data=(X_test_pad,y_test), batch_size=2)

loss,mae = model.evaluate(X_test_pad, y_test) # evaluate => verdiğimiz loss function sonucunu ve metric(ler) sonucunu verir.
print(f"MAE Skoru: {mae}, Loss : {loss}")

# val_loss Artıyorsa ⏫ train_loss azalıyorsa ⏬ -> Overfitting
# val_loss ~ train_loss (birbirine yakın) ama ikisi de yüksek ise -> Underfitting 
# val_loss Azalıyorsa ⏬ train_loss artıyorsa ⏫ -> Regularization fazla olduğu durum.
# EarlyStopping-Grafik İzleme ?


# Bu modelin overfitting yapmasını önlemek için gereken işlemleri uygulayınız.

