from typing import Dict

from .classification import *
from .regression import *

#xgboost
#lightgbm
#catboost


def get_classification_metrics(X, y, model) -> Dict[str, float]:
    pred = model.predict(X)

    return {
        'ACC': accuracy_metric(y, pred),
        'BACC': balanced_accuracy_metric(y, pred),
        'PREC': average_precision_metric(y, pred),
        'REC': recall(y, pred),
        'SPEC': specifity(y, pred),
        'INFO': informedness(y, pred),
        'MARK': markedness(y, pred),
        'MATTHEWS': matthews(y, pred),
        'F1': f1_metric(y, pred),
        'CE': cross_entropy(y, pred),
        'ROC_AUC': auc_metric_ovr(y, pred)


    }


def get_regression_metrics(X, y, model) -> Dict[str, float]:
    pred = model.predict(X)
    params = model.get_params()

    if not hasattr(model, 'params'):
        raise NotImplementedError('Model must have "params" atribute')

    return {
        'RMSE': root_mean_squared_error_metric(y, pred),
        'NRMSE': normalized_root_mean_squared_error_metric(y, pred),
        'MSE': mean_squared_error_metric(y, pred),
        'NMSE': mean_squared_error_metric(y, pred),
        'MAE': mean_absolute_error(y, pred),
        'NMAE': normalized_mean_absolute_error_metric(y, pred),
        'R_squared': r2_metric(y, pred),
        'Adj_R_squared': adj_r2(y, pred, params),
        'Spearman': spearman_metric(y, pred),
    }

