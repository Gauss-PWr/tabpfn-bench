import numpy as np
import pandas as pd
import torch
from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from tabpfn import TabPFNClassifier, TabPFNRegressor
from xgboost import XGBClassifier, XGBRegressor

from tools.constants import (catboost_params, lgbm_params, tabpfn_params,
                             xgb_params, int_params)
from sklearn.metrics import mean_squared_error, log_loss

def match_model_params(model_class_name):
    match model_class_name:
        case "XGBClassifier":
            params = xgb_params
        case "XGBRegressor":
            params = xgb_params
        case "LGBMClassifier":
            params = lgbm_params
        case "LGBMRegressor":
            params = lgbm_params
        case "CatBoostClassifier":
            params = catboost_params
        case "CatBoostRegressor":
            params = catboost_params
        case "TabPFNClassifier":
            params = tabpfn_params
        case "TabPFNRegressor":
            params = tabpfn_params
        case _:
            raise ValueError(f"Model {model_class_name} not supported.")
    return params

def create_model_instance(model_class_name, params):
    match model_class_name:
        case "XGBClassifier":
            return XGBClassifier(**params)
        case "XGBRegressor":
            return XGBRegressor(**params)
        case "LGBMClassifier":
            return LGBMClassifier(**params)
        case "LGBMRegressor":
            return LGBMRegressor(**params)
        case "CatBoostClassifier":
            return CatBoostClassifier(**params)
        case "CatBoostRegressor":
            return CatBoostRegressor(**params)
        case "TabPFNClassifier":
            return TabPFNClassifier(**params)
        case "TabPFNRegressor":
            return TabPFNRegressor(**params)
        case _:
            raise ValueError(f"Model {model_class_name} not supported.")

def get_model_params(
    model,
    X_train,
    y_train,
    tune=False,
    max_time=60,
    use_tensor=False,
):
    assert torch.cuda.is_available(), "CUDA is not available. Please check your PyTorch installation."

    model_class_name = model.__class__.__name__
    is_classifier = "Classifier" in model_class_name

    if type(X_train) == torch.Tensor:
        X_train = X_train.cpu().numpy()

    elif type(X_train) == pd.DataFrame:
        X_train = X_train.to_numpy()

    if type(y_train) == torch.Tensor:
        if is_classifier:
            y_train = y_train.cpu().long().numpy()
        else:
            y_train = y_train.cpu().float().numpy()

    elif type(y_train) == pd.Series:
        if is_classifier:
            y_train = y_train.to_numpy().astype(np.int64)
        else:
            y_train = y_train.to_numpy().astype(np.float32)

    if not tune:
        return {}
    params = match_model_params(model_class_name)
    
    unique_values, counts = np.unique(y_train, return_counts=True)
    can_stratify = len(unique_values) >= 2 and np.min(counts) >= 2

    X_train_subset, X_val, y_train_subset, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, 
        stratify=y_train if can_stratify else None
    )
    if use_tensor:
        X_train_subset, y_train_subset = (
            torch.tensor(X_train_subset),
            torch.tensor(y_train_subset),
        )

        X_val, y_val = torch.tensor(X_val), torch.tensor(y_val)
        X_train_subset, y_train_subset = X_train_subset.float(), y_train_subset.long()
        X_val, y_val = X_val.float(), y_val.long()

        if is_classifier:
            y_train_subset = y_train_subset.int()
            y_val = y_val.int()

    def objective(params):
        for param in int_params:
            if param in params:
                params[param] = int(params[param])
        
        model_instance = create_model_instance(model_class_name, params)
        
        model_instance.fit(X_train_subset, y_train_subset)
        
        if hasattr(model_instance, "predict_proba"):
            y_pred = model_instance.predict_proba(X_val)
            if y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]
        else:
            y_pred = model_instance.predict(X_val)
        
        metric_func = (log_loss if "Classifier" in model_class_name else mean_squared_error)
        score = metric_func(y_val, y_pred)
        return {"loss": score, "status": STATUS_OK}

    trials = Trials()
    best = fmin(
        objective,
        params,
        algo=tpe.suggest,
        max_evals=None,
        trials=trials,
        timeout=max_time,
    )
    best_params = space_eval(params, best)
    for param in int_params:
        if param in best_params:
            best_params[param] = int(best_params[param])
    print("Best parameters:", best_params)

    return best_params
