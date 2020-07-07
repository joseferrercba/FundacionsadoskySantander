from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report, balanced_accuracy_score
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
        