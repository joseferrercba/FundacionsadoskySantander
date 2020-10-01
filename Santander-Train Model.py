#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[ ]:
import sys
import mlflow
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from classes.Vectorizer import Vectorizer
from classes.Classifier import Classifier
from classes.Resample import Resample
from classes.ModelBuilder import ModelBuilder
from classes.Constans import *

# In[ ]:
classifier = Classifier()
vectorizer = Vectorizer()
resample = Resample()
builder = ModelBuilder()

#--------------------------------------------------#
###                  CLASSIFIERS                 ###
#--------------------------------------------------#
classifier_list = classifier.get_classifier_list()

def build_model(X, y, model, df_test):
    model_name = model.__class__.__name__
    X_train, X_test, y_train, y_test = builder.get_train_test_split(X, y)
    RESAMPLE_FILE = 'data/apply_resample_after_{}.png'.format(model_name)
    if APPLY_RESAMPLE == True:
        #Join X_train and y_train to add more classes to train
        df_Xtrain = pd.DataFrame(X_train, columns=['Pregunta'])
        df_Xtrain.set_index('Pregunta')
        df_Xtrain['Intencion_cat_label'] = y_train
        cnt_pro = df_Xtrain['Intencion_cat_label'].value_counts()
        plt.figure(figsize=(35,4))
        sns.barplot(x=cnt_pro.index, y=cnt_pro.values, data=cnt_pro, alpha=0.8)
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel('Intencion', fontsize=12)
        plt.xticks(rotation=90)
        #Apply resample to train dataset
        df_Xtrain = resample.apply_resample(df_Xtrain, 'Pregunta', 10, 100)
        X_train = df_Xtrain['Pregunta'].values
        y_train = df_Xtrain['Intencion_cat_label'].values
        cnt_pro = df_Xtrain['Intencion_cat_label'].value_counts()
        plt.figure(figsize=(35,4))
        sns.barplot(x=cnt_pro.index, y=cnt_pro.values, data=cnt_pro, alpha=0.8)
        plt.ylabel('Number of Occurrences', fontsize=12)
        plt.xlabel('Intencion', fontsize=12)
        plt.xticks(rotation=90)    
        plt.savefig(RESAMPLE_FILE)    
        
    try:
        if mlflow.active_run() != None:
            mlflow.end_run()

        with mlflow.start_run():            
            mlflow.log_param('Classifier', model_name)    
            mlflow.log_param('resampling', APPLY_RESAMPLE)         
            mlflow.log_param('vect', VECTORIZER_TYPE.name)
            mlflow.log_param('tokenizer', ('None' if (TOKENIZER_TYPE == None) else TOKENIZER_TYPE.name))    
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
            if APPLY_RESAMPLE == True:
                mlflow.log_artifact(RESAMPLE_FILE)
            
            optimized_model, model_best_params = builder.train_model(X_train, y_train, classifier=model, scoring=SCORING, refit=REFIT)
                    
            training_score, test_score, balanced_acc_score, acc_score = builder.score(optimized_model, X_train, X_test, y_train, y_test)
                        
            print('Logging metrics...')            
            mlflow.log_metric('training_score', training_score)
            mlflow.log_metric('test_score', test_score)
            mlflow.log_metric("balanced_accuracy_score", balanced_acc_score)
            mlflow.log_metric("accuracy_score", acc_score)                
            for key, value in model_best_params.items():        
                mlflow.log_param(key, value)                        
            
            #print('plot confusion matrix')
            #CM_FILE = plot_confusion_matrix(model_name, y_test, pred)
            #mlflow.log_artifact(CM_FILE) 
            
            #print('Processing Classification Report...')
            #report = classification_report(y_test, pred)
            #df_report = pd.DataFrame()
            #lines = report.split('\n')
            #for line in lines[2:-5]:    
            #    row_data = line.split('      ')             
            #    row = pd.Series(data={'class': row_data[1].strip(), 
            #                            'precision': row_data[2].strip(), 
            #                            'recall': row_data[3].strip(), 
            #                            'f1-score': row_data[4].strip(), 
            #                            'support': row_data[5].strip()
            #                            }
            #                            )
            #    df_report = df_report.append(row, ignore_index=True)
            #df_report['f1-score'] = df_report['f1-score'].astype(float)
            #df_report['precision'] = df_report['precision'].astype(float)
            #df_report['recall'] = df_report['recall'].astype(float)
            #df_report['class'] = df_report['class'].astype(int)
            #
            #cnt_pro = df_report[df_report['recall'] < 0.50]
            #fig, ax = plt.subplots(nrows=2, sharex=True)
            #fig.set_figheight(5)
            #fig.set_figwidth(15)
            #sns.barplot(x=cnt_pro.index, y=cnt_pro.values, data=cnt_pro, alpha=0.8, ax=ax[0])
            #plt.ylabel('Recall', fontsize=12)
            #plt.xlabel('Intencion', fontsize=12)
            #items = df_train[df_train['Intencion_cat_label'].isin(cnt_pro.index)]['Intencion_cat_label'].value_counts()
            #sns.barplot(x=cnt_pro.index, y=cnt_pro.values, data=cnt_pro, alpha=0.8, ax=ax[1])
            #plt.ylabel('Number of Occurrences', fontsize=12)
            #plt.xticks(rotation=90)
            #REPORT_FILE = 'data/classification_report_{}.png'.format(model_name)
            #plt.savefig(REPORT_FILE)
            #mlflow.log_artifact(REPORT_FILE)
            
            print('Training Model With all train dataset...')
            optimized_model.fit(X, y)
            print('Finished Training Model With all train dataset.')            
            training_score = optimized_model.score(X , y)
            print("Training score for all data: %f" % training_score)
            
            mlflow.sklearn.log_model(optimized_model, "model")
            print('Save predictions of test to CSV...')
            pred_det = optimized_model.predict(df_test['Pregunta'].values)
            df_test['Intencion'] = pred_det
            SUBMIT_FILE = 'data/submit_{}.csv'.format(model_name)
            df_test.to_csv(SUBMIT_FILE, mode='w', header=False, columns=['id','Intencion'], index=False, sep=',')
            mlflow.log_artifact(SUBMIT_FILE)            
            return optimized_model
    except:
        print('There was an error!')
        print(sys.exc_info())
        mlflow.end_run()        

# ### Get Info from CSV

# In[ ]:
df_train = shuffle(pd.read_csv('data/train_preprocessed.csv', sep='|'))
df_test = shuffle(pd.read_csv('data/test_santander.csv', usecols=['id','Pregunta']))
print(df_train['Intencion_cat_label'].value_counts())

# add one more sample because I have one case with just one sample and stratify need at least 2 samples
df_train = resample.apply_resample(df_train, 'Pregunta', 5, 100)

#grouped = df_train.groupby('Intencion_cat_label')['Intencion'].count().sort_values(ascending=True)
#major_samples_ids = grouped[(grouped.values > 5)].index.sort_values(ascending=True)
#df_train = df_train[df_train['Intencion_cat_label'].isin(major_samples_ids)]
#print(df_train.shape)

X = df_train[COLUMNA_PREGUNTAS].values
y = df_train['Intencion_cat_label'].values

# ### build model

# In[ ]:
for classifier in classifier_list:
    optimized_model = build_model(X,y, classifier, df_test)