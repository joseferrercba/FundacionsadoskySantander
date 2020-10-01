# In[ ]:
import sys
import mlflow
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Bidirectional, LSTM
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from classes.Constans import COLUMNA_PREGUNTAS
from classes.ModelBuilder import ModelBuilder
from classes.Resample import Resample
from classes.Vectorizer import Vectorizer
from classes.Constans import *

seed = 7
vocab_size = 1000 
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>' #OOV = Out of Vocabulary
training_portion = .8
#np.random.seed(seed)
builder = ModelBuilder()
resample = Resample()
vectorizer = Vectorizer()

df_train = shuffle(pd.read_csv('data/train_preprocessed.csv', sep='|'))
df_test = shuffle(pd.read_csv('data/test_santander.csv', usecols=['id','Pregunta']))

# add one more sample because I have one case with just one sample and stratify need at least 2 samples
df_train = resample.apply_resample(df_train, 'Pregunta', 5, 100)

X = df_train['Pregunta'].values
y = df_train['Intencion'].values

X_train, X_test, y_train, y_test = builder.get_train_test_split(X, y)

# #Join X_train and y_train to add more classes to train
# df_Xtrain = pd.DataFrame(X_train, columns=['Pregunta'])
# df_Xtrain.set_index('Pregunta')
# df_Xtrain['Intencion_cat_label'] = y_train
# #Apply resample to train dataset
# df_Xtrain = resample.apply_resample(df_Xtrain, 'Pregunta', 10, 100)
# X_train = df_Xtrain['Pregunta'].values
# y_train = df_Xtrain['Intencion_cat_label'].values

print(f"train_articles {len(X_train)}")
print("train_labels", len(y_train))
print("validation_articles", len(X_test))
print("validation_labels", len(y_test))

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(X_train)
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

validation_sequences = tokenizer.texts_to_sequences(X_test)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(y)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(y_train))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(y_test))

label_tokenizer.word_index

model = Sequential()

model.add(Embedding(vocab_size, embedding_dim))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(embedding_dim)))
model.add(Dense(2, activation='softmax'))

model.summary()

opt = Adam(lr=0.001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

num_epochs = 2
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)

#import matplotlib.pyplot as plt
#
#def plot_graphs(history, string):
#  plt.plot(history.history[string])
#  plt.plot(history.history['val_'+string])
#  plt.xlabel("Epochs")
#  plt.ylabel(string)
#  plt.legend([string, 'val_'+string])
#  plt.show()
#  
#plot_graphs(history, "accuracy")
#plot_graphs(history, "loss")

print('Calculating Scores...')
#training_score = model.score(train_padded , training_label_seq)
#test_score = model.score(validation_padded  , validation_label_seq)        
pred = model.predict(validation_padded)
#Compute the balanced accuracy
#The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.
#The best value is 1 and the worst value is 0 when adjusted=False.        
#balanced_acc_score = balanced_accuracy_score(validation_label_seq, pred)
#acc_score = accuracy_score(validation_label_seq, pred)
#print('Summary Scores')
#print("Training set score: %f" % training_score)
#print("Testing  set score: %f" % test_score)
#print('balanced_accuracy_score: %f' % balanced_acc_score)
#print('accuracy_score: %f' % acc_score)
#print('--------------------------------------------------------')
# Evaluate your model accuracy on the test data
score = model.evaluate(validation_padded, validation_label_seq)

labels = y
pred_label = []
for text in tqdm(df_test['Pregunta'].values, ascii=True):
  seq = tokenizer.texts_to_sequences(text)  
  padded = pad_sequences(seq, maxlen=max_length)  
  pred = model.predict(padded)  
  label = labels[np.argmax(pred)-1][4:]
  print('Sentence: {}, label: {}'.format(text, label))
  pred_label.append(label)
df_test['Intencion'] = pred_label
print('Save predictions of test to CSV...')
SUBMIT_FILE = 'data/submit_keras.csv'
df_test.to_csv(SUBMIT_FILE, mode='w', header=False, columns=['id','Intencion'], index=False, sep=',')
