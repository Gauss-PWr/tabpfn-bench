import numpy as np

from sklearn.metrics import (
    multilabel_confusion_matrix,
    matthews_corrcoef,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)


class ClassificationBenchmark:
    def __init__(self, X, y, model, use_tensor=False):
        if use_tensor:
            import torch

            assert (
                torch.cuda.is_available()
            ), "CUDA is not available. Ensure you have a compatible GPU and PyTorch installed with CUDA support."
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.int64)
        self.target = np.array(y) if not isinstance(y, np.ndarray) else y
        self.pred = (
            model.predict(X).cpu().numpy()
            if not isinstance(model.predict(X), np.ndarray)
            else model.predict(X)
        )
        try:
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)
                self.pred_proba = (
                    probabilities.cpu().numpy()
                    if not isinstance(probabilities, np.ndarray)
                    else probabilities
                )
            elif hasattr(model, 'decision_function'):
                # For models like SVM that have decision_function
                decision_scores = model.decision_function(X)
                self.pred_proba = (
                    decision_scores.cpu().numpy()
                    if not isinstance(decision_scores, np.ndarray)
                    else decision_scores
                )
            else:
                self.pred_proba = None
                print(f"Warning: {model.__class__.__name__} doesn't support predict_proba or decision_function")
        except Exception as e:
            print(f"Warning: Could not get probabilities from {model.__class__.__name__}: {e}")
            self.pred_proba = None
        
        self.dim = X.shape[1] if hasattr(X, "shape") else 1
        self.n = X.shape[0] if hasattr(X, "shape") else len(X)
        self.model_name = model.__class__.__name__
      
    def accuracy(self):
        return np.mean(self.target == self.pred)

    def precision(self, average="macro"):
        return precision_score(self.target, self.pred, average=average)

    def precision(self, average="macro"):
        return precision_score(self.target, self.pred, average=average)

    def recall(self, average="macro"):
        return recall_score(self.target, self.pred, average=average)

    def F1(self, average="macro"):
        return f1_score(self.target, self.pred, average=average)

    def roc_auc(self, average="macro", multi_class="ovr"):
        if self.pred_proba is None:
            print(f"Warning: Cannot calculate ROC AUC for {self.model_name} - no probabilities available")
            return np.nan
        
        try:
            n_classes = len(np.unique(self.target))
            
            if n_classes == 2:
                if self.pred_proba.ndim == 2 and self.pred_proba.shape[1] == 2:
                    proba = self.pred_proba[:, 1]  
                elif self.pred_proba.ndim == 1:
                    proba = self.pred_proba  
                else:
                    proba = self.pred_proba[:, -1]  
                
                return roc_auc_score(self.target, proba)
            
            else:
                if self.pred_proba.ndim == 1:
                    print(f"Warning: 1D scores for multi-class in {self.model_name}, cannot compute ROC AUC")
                    return np.nan
                
                return roc_auc_score(
                    self.target, 
                    self.pred_proba, 
                    average=average, 
                    multi_class=multi_class
                )
                
        except Exception as e:
            print(f"Warning: ROC AUC calculation failed for {self.model_name}: {e}")
            return np.nan


    def informedness(self):

        mc_matrix = multilabel_confusion_matrix(self.target, self.pred)

        recall = mc_matrix[:, 1, 1] / (mc_matrix[:, 1, 1] + mc_matrix[:, 1, 0])

        specificity = mc_matrix[:, 0, 0] / (mc_matrix[:, 0, 0] + mc_matrix[:, 0, 1])

        return np.mean(recall + specificity) - 1

    def markedness(self):

        mc_matrix = multilabel_confusion_matrix(self.target, self.pred)

        precision = mc_matrix[:, 1, 1] / (mc_matrix[:, 1, 1] + mc_matrix[:, 0, 1])

        npv = mc_matrix[:, 0, 0] / (mc_matrix[:, 0, 0] + mc_matrix[:, 1, 0])

        return np.mean(precision + npv) - 1

    def matthews(self):
        return matthews_corrcoef(self.target, self.pred)
