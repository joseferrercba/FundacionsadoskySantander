from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, balanced_accuracy_score, confusion_matrix, plot_confusion_matrix
from matplotlib import pyplot as plt
class Accuracy(object):  

    @staticmethod
    def get_balanced_accuracy_score(y, pred):    
        balanced_accuracyscore = balanced_accuracy_score(y,pred)   
        print('')         
        print('--------------------------------------------------------')
        print('-- Summary --')
        print('--------------------------------------------------------')
        print('balanced_accuracy_score: ' + str(round(balanced_accuracyscore,2)))
        return balanced_accuracyscore

    @staticmethod
    def get_accuracy_score(y, pred):            
        accuracyscore = accuracy_score(y,pred)   
        print('') 
        print('--------------------------------------------------------')
        print('-- Summary --')
        print('--------------------------------------------------------')        
        print('accuracy_score: ' + str(round(accuracyscore,2)))
        return accuracyscore

    @staticmethod
    def get_classification_report(y_test_labels, pred_labels):
        report = classification_report(y_test_labels, pred_labels)
        print('')
        print('--------------------------------------------------------')
        print('-- Summary --')
        print('--------------------------------------------------------')
        print(report)
        return report

    @staticmethod
    def get_confusion_matrix(classifier, y_test_labels, pred_labels, class_names):
        report = plot_confusion_matrix(classifier, y_test_labels, pred_labels, display_labels=class_names, cmap=plt.cm.Blues)        
        print(report.confusion_matrix)
        return report
        