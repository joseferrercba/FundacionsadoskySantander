import nltk
import joblib
from functools import partial
import io
import json
import os
from .CustomTokenizer import CustomTokenizer
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline

class ModelBuilder(object): 
    
    def __init__(self):
        self.stopwords = nltk.corpus.stopwords.words('spanish')        

    def __Print(self, text):
        print('')
        print('--------------------------------------------------------')
        print('-- {} --'.format(text))
        print('--------------------------------------------------------')        


    def GetVectorizer(self):                
        vect = TfidfVectorizer(stop_words=self.stopwords, 
                                tokenizer=nltk.word_tokenize)
        return vect

    def Summary(self, model, model_name, X_train, X_test, y_train, y_test):        
        self.__Print('Summary')
        print("Training set score for " + model_name + " %f" % model.score(X_train , y_train))
        print("Testing  set score for " + model_name + " %f" % model.score(X_test  , y_test ))
        print('--------------------------------------------------------')        

    def SaveModelToDisk(self, model, model_name):                
        filename = ''.join(['models/', model_name, '_model.sav'])
        self.__Print('Saving model on {}'.format(filename))
        joblib.dump(model, filename)

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

    def GenerateTrainedModel(self, classifier, X_train, X_test, y_train, y_test):
        model_name = classifier.__class__.__name__
        params_grid = self.GetModelParams(model_name)
        #count_vect = CountVectorizer()        
        tokenizer = CustomTokenizer()
        vect = self.GetVectorizer()
        tfidf_trans = TfidfTransformer(sublinear_tf=True)

        X_train = [tokenizer.listToString(tokenizer.processAll(sentence)) for sentence in X_train]
        X_test = [tokenizer.listToString(tokenizer.processAll(sentence)) for sentence in X_test]
        #X_train_vect = vect.fit_transform(X_train_tokenize)
        #X_train_trans = tfidf_trans.fit_transform(X_train_vect)
        pipeline = Pipeline(steps=[#('vect', count_vect),
                                   ('vect', vect), 
                                   ('tfidf_transformer', tfidf_trans),                                   
                                   ('clf', classifier)])
        gridsearch = GridSearchCV(pipeline, params_grid, cv=5, n_jobs=-1)
        print('Training Model...')
        gridsearch.fit(X_train, y_train)                
        optimized_model = gridsearch.best_estimator_
        print('Finished Training Model.')        
        model_best_params = gridsearch.best_params_                
        self.SaveBestParamsToDisk(model_name, model_best_params)
        self.Summary(optimized_model, model_name, X_train, X_test, y_train, y_test)
        self.SaveModelToDisk(optimized_model, model_name)
        return optimized_model, model_best_params, X_train, X_test, y_train, y_test

