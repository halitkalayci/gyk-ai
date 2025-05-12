import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

#df = pd.read_csv("spam.csv", encoding_errors='ignore')
df = pd.read_csv("spam.csv", encoding='latin1')
df = df[['v1','v2']]
df.columns = ["label","text"]


label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Eğitim - Etiket
texts = df['text'].values
labels = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(texts,labels,test_size=0.2, random_state=42)

tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train) # sözlüğü oluşturma.


X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq  = tokenizer.texts_to_sequences(X_test)


X_train_pad = pad_sequences(X_train_seq, maxlen=100, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=100, padding='post')


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=10000, output_dim=64, input_length=100), # kelimeleri vektörlere çevirir
    # RNN => "Ben kahve içtim" => Ben,kahve,içtim -> Ali okula gitti. Sonra geldi.
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)), # LSTM (Long Short-Term Memory)=> GRU,SimpleRNN
    # LSTM'den gelen karmaşık temsili biraz daha "yoğunlaştırmak" için.
    # Tam Bağlantılı Katman
    tf.keras.layers.Dense(32, activation='relu'),
    # Ezberlemeyi önlemek (Overfitting) %50 oranında rastgele nöronları eğitimsiz bırakır.
    tf.keras.layers.Dropout(0.5),
    # 1 çıktı oldugu için Dense(1)
    tf.keras.layers.Dense(1, activation="sigmoid") # sigmoid => 0-1 arasında olasılık üretir (0.94 -> %94 spam)
])

# spam-değil
# türü, cins -> Memeli-Kedi



# Padding ["0, 15, 16, 18", "15, 17, 19, 20"] (Derin öğrenme modelleri her girdiyi sabit uzunlukta bekler.)
# pre-post => 0'ı öne mi arkaya mı ekleyelim?
#
# maxlen = Çok uzun cümleleri istenilen maksimum uzunluğa göre kesmek için.

#bilmediklerim->1
#ben->3
#kahve->2

# 3-2-1

# Tokenizer -> verilen bir metni daha küçük parçalara (token) ayıran bir yapıdır.

# Ben, sabah kahve içtim. -> Tokenlara ayırma ve sayı atama. 
# [ {"Ben"->1}, {"Sabah"->2}, {"Kahve"->3}, {"İçtim"->4}]

# Subword -> Playing -> play->2 ##ing->3 [playfully -> play, ##ful, ##ly]
# Fantastic => [fan, ##tas ##tic]
# gel -> gelemediklerimizden


# Ben sabah kahve içtim, sonra öğlen tekrar kahve içtim.
# [1,2,3,4,5,6,7,3,4]

# Ben sabah kahve içtim, üzerimde kahve renk bir tişört var.
# context-aware (BERT) -> Kullanım noktasının öncesi ve sonrasına bakarak anlam çıkartılır.

# Merhaba 😂
# Merhaba 😡

# num_words => maximum tutalacak kelime sayısı
# oov_token => Bilinmeyen kelimeler için özel token <OOV>