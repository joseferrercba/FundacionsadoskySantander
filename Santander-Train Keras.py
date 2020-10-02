# In[ ]:
import sys
import mlflow
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Bidirectional, LSTM, SpatialDropout1D
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from classes.Constans import COLUMNA_PREGUNTAS
from classes.ModelBuilder import ModelBuilder
from classes.Resample import Resample
from classes.Vectorizer import Vectorizer
from classes.Constans import *
def baseline_model():
    print('')

builder = ModelBuilder()
resample = Resample()
vectorizer = Vectorizer()

df_train = shuffle(pd.read_csv('data/train_preprocessed.csv', sep='|'))
df_test = shuffle(pd.read_csv('data/test_santander.csv', usecols=['id','Pregunta']))

# add one more sample because I have one case with just one sample and stratify need at least 2 samples
df_train = resample.apply_resample(df_train, 'Pregunta', 5, 100)

print('sentence: {}'.format(df_train['Pregunta'][:1].values))
print('label: {}'.format(df_train['Intencion'][:1].values))

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df_train['Pregunta'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

X = tokenizer.texts_to_sequences(df_train['Pregunta'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)

y = pd.get_dummies(df_train['Intencion']).values
print('Shape of label tensor:', y.shape)

X_train, X_test, y_train, y_test = builder.get_train_test_split(X, y)
print(X_train.shape,y_train.shape)
print(X_test.shape,y_test.shape)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(352, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

epochs = 50
batch_size = 64

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

accr = model.evaluate(X_test,y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()

labels = df_train.Intencion.unique()
print('Save predictions of test to CSV...')
seq = tokenizer.texts_to_sequences(df_test['Pregunta'].values)
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)
df_test['Intencion'] = labels[np.argmax(pred)]

SUBMIT_FILE = 'data/submit_{}.csv'.format('keras')
df_test.to_csv(SUBMIT_FILE, mode='w', header=False, columns=['id','Intencion'], index=False, sep=',')      