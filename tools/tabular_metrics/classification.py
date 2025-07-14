import numpy as np

from sklearn.metrics import (multilabel_confusion_matrix, 
                             matthews_corrcoef, 
                             f1_score, 
                             roc_auc_score, 
                             precision_score, 
                             recall_score) 

class ClassificationBenchmark:
    def __init__(self, X, y, model):
        self.target = np.array(y) if not isinstance(y, np.ndarray) else y
        self.pred = np.array(model.predict(X)) if not isinstance(model.predict(X), np.ndarray) else model.predict(X)

        self.dim = X.shape[1] if hasattr(X, 'shape') else 1
        self.n = X.shape[0] if hasattr(X, 'shape') else len(X)
        self.model_name = model.__class__.__name__
        
        
        
    def accuracy(self):
        return np.mean(self.target == self.pred)

    def precision(self, average='macro'):
        return precision_score(self.target, self.pred, average=average)

    def precision(self, average='macro'):
        return precision_score(self.target, self.pred, average=average)

    def recall(self, average='macro'):
        return recall_score(self.target, self.pred, average=average)

    def F1(self, average='macro'):
        return f1_score(self.target, self.pred, average=average)

    def roc_auc(self, pred_proba, average='macro', multi_class='ovr'):
        target = np.array(target) if not isinstance(target, np.ndarray) else target
        pred_proba = np.array(pred_proba) if not isinstance(pred_proba, np.ndarray) else pred_proba

        assert pred_proba.ndim == 2, "pred_proba should be a 2D array for multi-class classification"

        return roc_auc_score(self.target, pred_proba, average=average, multi_class=multi_class)

    def informedness(self):
        pred = np.array(self.pred) if not isinstance(self.pred, np.ndarray) else self.pred

        mc_matrix = multilabel_confusion_matrix(self.target, pred)

        recall = mc_matrix[:, 1, 1] / (mc_matrix[:, 1, 1] + mc_matrix[:, 1, 0])
        
        specificity = mc_matrix[:, 0, 0] / (mc_matrix[:, 0, 0] + mc_matrix[:, 0, 1])
        
        return np.mean(recall + specificity) - 1

    def markedness(self):
        pred = np.array(self.pred) if not isinstance(self.pred, np.ndarray) else self.pred

        mc_matrix = multilabel_confusion_matrix(self.target, pred)
        
        precision = mc_matrix[:, 1, 1] / (mc_matrix[:, 1, 1] + mc_matrix[:, 0, 1])
        
        npv = mc_matrix[:, 0, 0] / (mc_matrix[:, 0, 0] + mc_matrix[:, 1, 0])
        
        return np.mean(precision + npv) - 1

    def matthews(self):
        return matthews_corrcoef(self.target, self.pred)


