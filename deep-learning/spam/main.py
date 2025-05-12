import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer

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

sentences = ["Ben kahve içtim.","Kahve çok güzeldi."]
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')

tokenizer.fit_on_texts(sentences)
print(tokenizer.word_index)
# {'<OOV>': 1, 'kahve': 2, 'ben': 3, 'içtim': 4, 'çok': 5, 'güzeldi': 6}

test_sentences = ["Ben kahve içmedim."]
sequences = tokenizer.texts_to_sequences(test_sentences)
print(sequences)