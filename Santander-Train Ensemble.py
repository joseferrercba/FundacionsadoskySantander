#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[ ]:
import sys
import mlflow
import pandas as pd 
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
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
classifier_list = []

# ### Get Info from CSV

# In[ ]:
df_train = shuffle(pd.read_csv('data/train_preprocessed.csv', sep='|'))
df_test = shuffle(pd.read_csv('data/test_santander.csv', usecols=['id','Pregunta']))
print(df_train['Intencion_cat_label'].value_counts())
# add one more sample because I have one case with just one sample and stratify need at least 2 samples
df_train = resample.apply_resample(df_train, 'Pregunta', 5, 100)

# In[ ]:
def plot_occurrences(data, title='Dummy Title'):
    plt.figure(figsize=(35,6))
    sns.countplot(x=data['Intencion_cat_label'], data=data, alpha=0.8, order=data['Intencion_cat_label'].value_counts().index)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Intencion', fontsize=2)
    plt.xticks(rotation=90)
    plt.title(title, fontsize=20)
    REPORT_FILE = 'data/plot_occurrences_{}.png'.format(title)
    plt.savefig(REPORT_FILE)

def build_optimized(X, y, classifier):
    optimized_model, _ = builder.train_model(X, y, classifier)
    X_train, X_test, y_train, y_test = builder.get_train_test_split(X, y)
    builder.score(optimized_model, X_train, X_test, y_train, y_test)
    print('Training Model With all train dataset...')
    optimized_model.fit(X, y)
    print('Finished Training Model With all train dataset.')            
    training_score = optimized_model.score(X , y)
    print("Training score for all data: %f" % training_score)
    return optimized_model

def build(X, y, classifier):
    try:        
        model_name = classifier.__class__.__name__
        X_train, X_test, y_train, y_test = builder.get_train_test_split(X, y)
        
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
                mlflow.log_artifact(RESAMPLE_FILE)
                                    
            optimized_model, model_best_params = builder.train_model(X_train, y_train, stacking_classifier)
            training_score, test_score, balanced_acc_score, acc_score = builder.score(optimized_model, X_train, X_test, y_train, y_test)
            print('Training Model With all train dataset...')
            optimized_model.fit(X, y)
            print('Finished Training Model With all train dataset.')            
            training_score = optimized_model.score(X , y)
            print("Training score for all data: %f" % training_score)        
                                                
            print('Logging metrics...')            
            mlflow.log_metric('training_score', training_score)
            mlflow.log_metric('test_score', test_score)
            mlflow.log_metric("balanced_accuracy_score", balanced_acc_score)
            mlflow.log_metric("accuracy_score", acc_score)                
            for key, value in model_best_params.items():        
                mlflow.log_param(key, value)                        
            
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
    except:
        print('There was an error!')
        print(sys.exc_info())
        mlflow.end_run()

# In[ ]:
plot_occurrences(df_train)

# In[ ]:
grouped = df_train.groupby('Intencion_cat_label')['Intencion'].count().sort_values(ascending=True)
major_samples_ids = grouped[(grouped.values < grouped.max()) & (grouped.values > 50)].index.sort_values(ascending=True)
mid_samples_ids = grouped[(grouped.values <= 50) & (grouped.values > 20)].index.sort_values(ascending=True)
low_samples_ids = grouped[(grouped.values <= 20) & (grouped.values > 0)].index.sort_values(ascending=True)

print('Size each group: Major:{}, Mid:{}, Low:{}'.format(len(major_samples_ids), len(mid_samples_ids), len(low_samples_ids)))

major_samples = df_train[df_train['Intencion_cat_label'].isin(major_samples_ids)]
plot_occurrences(major_samples, 'Major Samples (between less than {} & more than {} Occurrences)'.format(grouped.max(), 50))
mid_samples = df_train[df_train['Intencion_cat_label'].isin(mid_samples_ids)]
plot_occurrences(mid_samples, 'Mid Samples (between less than equal 50 & more than 20 Occurrences)')
low_samples = df_train[df_train['Intencion_cat_label'].isin(low_samples_ids)]
plot_occurrences(low_samples, 'Low Samples (between less than equal 20 & more than 0 Occurrences)')

# In[ ]:
print('Processing major samples model')
X = major_samples[COLUMNA_PREGUNTAS].values
y = major_samples['Intencion_cat_label'].values
major_samples_classifier = LinearSVC(verbose=VERBOSE, random_state=RANDOM_STATE)
major_samples_classifier = build_optimized(X, y, major_samples_classifier)

# In[ ]:
print('Processing mid samples model')
X = mid_samples[COLUMNA_PREGUNTAS].values
y = mid_samples['Intencion_cat_label'].values
mid_samples_classifier = LinearSVC(verbose=VERBOSE, random_state=RANDOM_STATE)
mid_samples_classifier = build_optimized(X, y, mid_samples_classifier)

# In[ ]:
print('Processing low samples model')
X = low_samples[COLUMNA_PREGUNTAS].values
y = low_samples['Intencion_cat_label'].values
low_samples_classifier = LinearSVC(verbose=VERBOSE, random_state=RANDOM_STATE)
low_samples_classifier = build_optimized(X, y, low_samples_classifier)

# In[ ]:
print('Processing stacking with base models')        
X = df_train[COLUMNA_PREGUNTAS].values
y = df_train['Intencion_cat_label'].values
stacking_classifier = classifier.get_stacking()
build(X, y, stacking_classifier)


# In[ ]:
#print('Processing stacking with previous trained models')
#X = df_train[COLUMNA_PREGUNTAS].values
#y = df_train['Intencion_cat_label'].values
#stacking = classifier.get_stacking()
#
##--------------------------------------------------#
#### TO USE STACKING OR VOTING UNCOMMENT ALL THIS ###
##--------------------------------------------------#
## define the base models
#level0 = list()	
#level0.append(('lsvc1', major_samples_classifier))
#level0.append(('lsvc2', mid_samples_classifier))
#level0.append(('lsvc3', low_samples_classifier))        
#level0.append(('lr', LogisticRegression(C=1, class_weight=CLASS_WEIGHT, solver='liblinear',
#                                        dual=False, multi_class='ovr', penalty='l2', 
#                                        random_state=RANDOM_STATE, max_iter=300)))
## define meta learner model
#level1 = LogisticRegression(C=1, class_weight=CLASS_WEIGHT, n_jobs=N_JOBS, random_state=RANDOM_STATE, max_iter=300)
#
## define the stacking ensemble
#stacking_classifier = StackingClassifier(estimators=level0, final_estimator=level1, verbose=VERBOSE, n_jobs=N_JOBS, passthrough=False, cv=CV)
#
#build(X, y, stacking_classifier)
