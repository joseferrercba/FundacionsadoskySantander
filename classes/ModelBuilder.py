import nltk
from functools import partial
import io
import json
import os
import re
from .CustomTokenizer import *
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import *
from imblearn.pipeline import Pipeline as imbPipeline
import mlflow

class ModelBuilder(object): 
    
    def __init__(self):
        self.stopwords = nltk.corpus.stopwords.words('spanish')        

    def __Print(self, text):
        print('')
        print('--------------------------------------------------------')
        print('-- {} --'.format(text))
        print('--------------------------------------------------------')        


    def GetVectorizer(self, vect_type='TfidfVectorizer'):    
        min_df=1            
        #tokenizer = custom_preprocess
        tokenizer = nltk.word_tokenize
        if vect_type == 'TfidfVectorizer':
            vect = TfidfVectorizer(min_df=min_df, tokenizer=tokenizer)
        elif vect_type == 'CountVectorizer':
            vect = CountVectorizer(min_df=min_df, tokenizer=tokenizer)
        mlflow.log_param('vect_stopwords', False)
        mlflow.log_param('vect_min_df', min_df)  
        mlflow.log_param('tokenizer', tokenizer.__name__)
        return vect

    def Summary(self, model, model_name, X_train, X_test, y_train, y_test):        
        self.__Print('Summary')
        training_score = model.score(X_train , y_train)
        test_score = model.score(X_test  , y_test )
        mlflow.log_metric('training_score', training_score)
        mlflow.log_metric('test_score', test_score)
        print("Training set score for " + model_name + " %f" % training_score)
        print("Testing  set score for " + model_name + " %f" % test_score)
        print('--------------------------------------------------------')        

    def SaveBestParamsToDisk(self, model_name, model_best_params):                
        print("Best Params for " + model_name + ": " + str(model_best_params))
        filename = ''.join(['model_best_params/', model_name, '_best_params.json'])
        self.__Print('Saving Best Parameters for {} on {}'.format(model_name, filename))
        with open(filename, 'w') as outfile:
            json.dump(model_best_params, outfile)        

    def GetModelParams(self, model_name):
        filename = 'model_params/{}_params.json'.format(model_name)
        fileexists = os.path.isfile(filename)
        if fileexists == False:
            with io.open(filename, 'w') as json_file:
                json_file.write(json.dumps({}))
        with open(filename) as json_file:
            params_grid = json.load(json_file)
        return params_grid

    def GenerateTrainedModel(self, classifier, X_train, X_test, y_train, y_test, cv=5, resampling = True, vect_type='TfidfVectorizer'):
        model_name = classifier.__class__.__name__
        params_grid = self.GetModelParams(model_name)        
        vect = self.GetVectorizer(vect_type=vect_type)                        
        print('Preprocessing data...')        
        if resampling == True:
            print('Resampling data...')
            sampling_strategy='not majority'
            oversample = SMOTE(random_state=42, n_jobs=-1, sampling_strategy=sampling_strategy, k_neighbors=3)         
            mlflow.log_param('resampling_class', oversample.__class__.__name__) 
            mlflow.log_param('sampling_strategy', sampling_strategy)   
            pipeline = imbPipeline(steps=[('vect', vect),       
                                       ('resample', oversample),                                                                
                                       ('clf', classifier)])
        else: 
            pipeline = Pipeline(steps=[('vect', vect), ('clf', classifier)])        
        mlflow.log_param('vect', vect.__class__.__name__)
        gridsearch = GridSearchCV(pipeline, params_grid, cv=cv, n_jobs=-1, verbose=2)
        print('Training Model...')
        gridsearch.fit(X_train, y_train)                
        optimized_model = gridsearch.best_estimator_
        print('Finished Training Model.')        
        model_best_params = gridsearch.best_params_                
        self.SaveBestParamsToDisk(model_name, model_best_params)
        self.Summary(optimized_model, model_name, X_train, X_test, y_train, y_test)        
        return optimized_model, model_best_params, X_train, X_test, y_train, y_test

