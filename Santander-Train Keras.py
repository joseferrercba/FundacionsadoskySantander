# In[ ]:
import sys
import mlflow
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.keras.layers.core import Flatten
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input, Embedding, Dropout, Conv1D, GlobalMaxPooling1D, Bidirectional, LSTM, SpatialDropout1D, Activation
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.metrics import *
from tensorflow.keras.losses import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.activations import *
from classes.Constans import COLUMNA_PREGUNTAS
from classes.ModelBuilder import ModelBuilder
from classes.Resample import Resample
from classes.Vectorizer import Vectorizer
from classes.Constans import *
LOAD_MODEL = False

def baseline_model_LSTM(output_len):    
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(SpatialDropout1D(0.3))
    model.add(Bidirectional(LSTM(EMBEDDING_DIM, dropout=0.3, recurrent_dropout=0.3)))
    model.add(Dense(EMBEDDING_DIM, activation=relu))
    model.add(Dropout(0.8))
    model.add(Dense(EMBEDDING_DIM, activation=relu))
    model.add(Dropout(0.8))
    model.add(Dense(output_len, activation=softmax))    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model = Sequential()
    #model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    #model.add(SpatialDropout1D(0.2))
    #model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))    
    #model.add(Dense(output_len, activation=sigmoid))
    #opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    plot_model(model, to_file='model_{}.png'.format('LSTM'), show_shapes=True, show_layer_names=True)
    return model

def baseline_model_RNN(output_len):    
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH))
    model.add(SpatialDropout1D(0.2))
    model.add(RNN(100))    
    model.add(Dense(output_len, activation=sigmoid))
    opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    print(model.summary())
    plot_model(model, to_file='model_{}.png'.format('LSTM'), show_shapes=True, show_layer_names=True)
    return model

try:
    if mlflow.active_run() != None:
        mlflow.end_run()

    with mlflow.start_run():            
        model_name = 'KerasClassifier'
        mlflow.log_param('Classifier', model_name)            
        mlflow.log_param('resampling', APPLY_RESAMPLE)         
        mlflow.log_param('vect', VECTORIZER_TYPE.name)
        mlflow.log_param('tokenizer', ('None' if (TOKENIZER_TYPE == None) else TOKENIZER_TYPE))    
        mlflow.log_param('resampling_class', RESAMPLER_TYPE.name) 
        mlflow.log_param('sampling_strategy', SAMPLING_STRATEGY)   
        mlflow.log_param('prep_column', COLUMNA_PREGUNTAS)   
        mlflow.log_param('test_size', TEST_SIZE)   
        mlflow.log_param('min_df', MIN_DF)   
        mlflow.log_param('shuffle', SHUFFLE)
        mlflow.log_param('k_neighbors', K_NEIGHBORS)      
        mlflow.log_param('scoring', SCORING)      
        mlflow.log_param('refit', REFIT)      
        mlflow.log_param('cv', CV)
        builder = ModelBuilder()
        resample = Resample()
        vectorizer = Vectorizer()    
        
        df_train = shuffle(pd.read_csv('data/train_preprocessed.csv', sep='|'))
        df_test = shuffle(pd.read_csv('data/test_santander.csv', usecols=['id','Pregunta']))

        # add one more sample because I have one case with just one sample and stratify need at least 2 samples
        df_train = resample.apply_resample(df_train, 'Pregunta', 5, 100)        

        print('sentence: {}'.format(df_train['Pregunta'][:1].values))
        print('label: {}'.format(df_train['Intencion'][:1].values))
        sentences = df_train['Pregunta'].values
        labels = df_train.Intencion.unique()
        labels_values = df_train.Intencion_cat_label.values
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        tokenizer.fit_on_texts(sentences)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        X = tokenizer.texts_to_sequences(sentences)
        X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
        print('Shape of data tensor:', X.shape)

        y = pd.get_dummies(df_train['Intencion']).values
        print('Shape of label tensor:', y.shape)

        if LOAD_MODEL == True:
            model = load_model('data/model.28-0.62-0.92.h5')
        else:
            model = KerasClassifier(build_fn=baseline_model_LSTM, output_len=len(labels), epochs=EPOCHS, batch_size=BATCH_SIZE,
                                    validation_split=TEST_SIZE, 
                                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001),
                                                ModelCheckpoint(save_best_only=True, filepath='data/model.h5')])
            history = model.fit(X, y)
                        
            plt.title('Loss')
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.legend()
            LOSS_FILE = 'data/loss_{}.png'.format(model_name)
            plt.savefig(LOSS_FILE) 
            mlflow.log_artifact(LOSS_FILE)
            plt.title('Accuracy')
            plt.plot(history.history['accuracy'], label='train')
            plt.plot(history.history['val_accuracy'], label='test')
            plt.legend()
            ACCURACY_FILE = 'data/accuracy_{}.png'.format(model_name)
            plt.savefig(ACCURACY_FILE)
            mlflow.log_artifact(ACCURACY_FILE)
                         
        seq = tokenizer.texts_to_sequences(sentences)
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        predictions = model.predict(padded)
        row_pred = []
        row_idx = []
        for pred in predictions:
            max_value_idx = np.argmax(pred)               
            row_pred.append(labels[max_value_idx][4:])
            #print('idx: {}, pred: {}, label: {}'.format(max_value_idx, pred[max_value_idx], labels[max_value_idx][4:]))        
        report = classification_report(labels_values.astype(str), row_pred, target_names=labels)
        print(report)
        
        print('Calculating Scores...')
        #Compute the balanced accuracy
        #The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.
        #The best value is 1 and the worst value is 0 when adjusted=False.        
        balanced_acc_score = balanced_accuracy_score(labels_values.astype(str), row_pred)
        acc_score = accuracy_score(labels_values.astype(str), row_pred)
        print('balanced_accuracy_score: %f' % balanced_acc_score)
        print('accuracy_score: %f' % acc_score)
        print('--------------------------------------------------------')                        
        print('Logging metrics...')            
        mlflow.log_metric("balanced_accuracy_score", balanced_acc_score)
        mlflow.log_metric("accuracy_score", acc_score)

        print('Save predictions of test to CSV...')               
        seq = tokenizer.texts_to_sequences(df_test['Pregunta'].values)
        padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
        predictions = model.predict(padded)
        row_pred = []
        row_idx = []
        for pred in predictions:
            max_value_idx = np.argmax(pred)                           
            row_pred.append(labels[max_value_idx][4:])
            #print('idx: {}, pred: {}, label: {}'.format(max_value_idx, pred[max_value_idx], labels[max_value_idx][4:]))        
        df_test['Intencion'] = row_pred
        SUBMIT_FILE = 'data/submit_{}.csv'.format('keras')
        df_test.to_csv(SUBMIT_FILE, mode='w', header=False, columns=['id','Intencion'], index=False, sep=',')  
        mlflow.log_artifact(SUBMIT_FILE)
except:
   print('There was an error!')
   print(sys.exc_info())
   mlflow.end_run()    
