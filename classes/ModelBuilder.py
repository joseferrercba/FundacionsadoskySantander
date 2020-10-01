import io
import json
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as imbPipeline
from classes.Vectorizer import Vectorizer
from classes.Resample import Resample
from classes.Constans import *
class ModelBuilder(object): 
    
    def __init__(self, n_jobs=-1, random_state=42, verbose=0):
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose       
        self.vectorizer = Vectorizer() 
        self.resample = Resample()

    def get_model_params(self, model_name):
        filename = 'model_params/{}_params.json'.format(model_name)
        fileexists = os.path.isfile(filename)
        if fileexists == False:
            with io.open(filename, 'w') as json_file:
                json_file.write(json.dumps({}))
        with open(filename) as json_file:
            params_grid = json.load(json_file)
        return params_grid  

    def __save_best_params_on_disk(self, model_name, model_best_params):                
        #print("Best Params for " + model_name + ": " + str(model_best_params))
        filename = ''.join(['model_best_params/', model_name, '_best_params.json'])
        #self.__Print('Saving Best Parameters for {} on {}'.format(model_name, filename))
        with open(filename, 'w') as outfile:
            json.dump(model_best_params, outfile)      
    
    def __Print(self, text):
        print('')
        print('--------------------------------------------------------')
        print('-- {} --'.format(text))
        print('--------------------------------------------------------')

    def score(self, model, X_train, X_test, y_train, y_test):        
        print('Calculating Scores...')
        training_score = model.score(X_train , y_train)
        test_score = model.score(X_test  , y_test)        
        pred = model.predict(X_test)
        #Compute the balanced accuracy
        #The balanced accuracy in binary and multiclass classification problems to deal with imbalanced datasets. It is defined as the average of recall obtained on each class.
        #The best value is 1 and the worst value is 0 when adjusted=False.        
        balanced_acc_score = balanced_accuracy_score(y_test, pred)
        acc_score = accuracy_score(y_test, pred)
        self.__Print('Summary Scores')
        print("Training set score: %f" % training_score)
        print("Testing  set score: %f" % test_score)
        print('balanced_accuracy_score: %f' % balanced_acc_score)
        print('accuracy_score: %f' % acc_score)
        print('--------------------------------------------------------')
        return training_score, test_score, balanced_acc_score, acc_score

    def get_train_test_split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(np.array(X), y, test_size = TEST_SIZE, stratify=y, random_state=RANDOM_STATE, shuffle=SHUFFLE)
        #y_train = y_train.astype('int')
        #y_test = y_test.astype('int')
        return X_train, X_test, y_train, y_test

    def train_model(self, X, y, classifier,                   
                    scoring='balanced_accuracy', 
                    refit='balanced_accuracy'):      
        model_name = classifier.__class__.__name__
        pipeline = self.get_pipeline(classifier)
        print('Training Model... {}'.format(model_name))
        params_grid = self.get_model_params(model_name)  
        gridsearch = GridSearchCV(pipeline, params_grid, cv=CV, 
                                  n_jobs=self.n_jobs, 
                                  verbose=self.verbose, 
                                  scoring=scoring, 
                                  refit=refit)        
        gridsearch.fit(X, y)                
        optimized_model = gridsearch.best_estimator_               
        model_best_params = gridsearch.best_params_                
        self.__save_best_params_on_disk(model_name, model_best_params)     
        print('Finished Training Model.')   
        return optimized_model, model_best_params
    
    def get_pipeline(self, classifier):
        vect = self.vectorizer.get_vectorizer(vectorizer_type=VECTORIZER_TYPE, tokenizer_type=TOKENIZER_TYPE)   
        resampler = self.resample.get_resampler(resampler_type=RESAMPLER_TYPE, sampling_strategy=SAMPLING_STRATEGY,
                                                k_neighbors=K_NEIGHBORS, allow_minority=True) 
        imbpipe = imbPipeline(steps=[('vect', vect), ('resample', resampler), ('clf', classifier)], verbose=VERBOSE)
        pipeline = Pipeline(steps=[('vect', vect), ('clf', classifier)], verbose=VERBOSE)
        return imbpipe if (APPLY_RESAMPLE == True) else pipeline      
    
    def plot_confusion_matrix(self, model_name, y_test, y_pred):
        conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
        print('Confusion matrix:\n', conf_mat)

        fig, ax = plt.subplots(figsize=(100,100)) 
        sns.heatmap(conf_mat, annot=True, ax = ax); #annot=True to annotate cells

        # labels, title and ticks
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        ax.set_title('Confusion Matrix')
        #ax.xaxis.set_ticklabels(['business', 'health']) 
        #ax.yaxis.set_ticklabels(['health', 'business'])
        CM_FILE = 'data/plot_confusion_matrix_{}.png'.format(model_name)
        plt.savefig(CM_FILE)
        return CM_FILE
