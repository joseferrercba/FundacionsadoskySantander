import xgboost as xgb
import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from classes.Constans import *

class Classifier(object):
    """
    Return Classifier model to train
    """
    def __init__(self, n_jobs=-1, random_state=42, verbose=0):
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose


    def get_classifier_list(self):
        classifier_list = []
        classifier_list.append(LinearSVC(verbose=self.verbose, random_state=self.random_state)) 
        #classifier_list.append(MultinomialNB()) 
        #classifier_list.append(GaussianNB()) 
        #classifier_list.append(LogisticRegression(n_jobs=self.n_jobs, random_state=self.random_state))
        #classifier_list.append(xgb.XGBClassifier(n_jobs=self.n_jobs)) 
        #classifier_list.append(SVC(verbose=self.verbose)) 
        #classifier_list.append(RandomForestClassifier(n_jobs=self.n_jobs, verbose=self.verbose)) 
        #classifier_list.append(BalancedRandomForestClassifier(n_jobs=self.n_jobs, verbose=self.verbose, random_state=self.random_state)) 
        #classifier_list.append(KNeighborsClassifier(n_jobs=self.n_jobs)) 
        #classifier_list.append(lgb.LGBMClassifier()) 
        #classifier_list.append(AdaBoostClassifier())
        #classifier_list.append(DecisionTreeClassifier()) 
        return classifier_list

    def get_stacking(self):        
        major_samples_classifier = LinearSVC(verbose=2, random_state=self.random_state)
        mid_samples_classifier = LinearSVC(verbose=2, random_state=self.random_state)
        low_samples_classifier = LinearSVC(verbose=2, random_state=self.random_state)

        #--------------------------------------------------#
        ### TO USE STACKING OR VOTING UNCOMMENT ALL THIS ###
        #--------------------------------------------------#
        # define the base models
        level0 = list()	
        level0.append(('lsvc1', major_samples_classifier))
        level0.append(('lsvc2', mid_samples_classifier))
        level0.append(('lsvc3', low_samples_classifier))        
        level0.append(('lr', LogisticRegression(C=1, class_weight=CLASS_WEIGHT, solver='liblinear',
                                                dual=False, multi_class='ovr', penalty='l2', 
                                                n_jobs=self.n_jobs, random_state=self.random_state, max_iter=300)))
        # define meta learner model
        level1 = LogisticRegression(C=1, class_weight=CLASS_WEIGHT, n_jobs=self.n_jobs, random_state=self.random_state, max_iter=300)

        # define the stacking ensemble
        stackingclassifier = StackingClassifier(estimators=level0, final_estimator=level1, verbose=2, n_jobs=self.n_jobs, passthrough=False, cv=CV)        
        return stackingclassifier