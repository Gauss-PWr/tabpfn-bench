import time

import numpy as np
import pandas as pd
import torch
import tqdm
from hyperopt import STATUS_OK, Trials, fmin, space_eval, tpe
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

import tools.tabular_metrics as tabular_metrics
from tools.constants import (catboost_params, lgbm_params, tabpfn_params,
                             xgb_params)


def get_scoring_direction(metric):
    # not implemented yet 1 for maximization, -1 for minimization
    return 1


def match_model_params(model):
    match model.__class__.__name__:
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
            raise ValueError(f"Model {model.__class__.__name__} not supported.")
    return params


def get_model_params(
    model,
    X_train,
    y_train,
    tune=False,
    tune_metric="f1",
    max_time=60,
    use_tensor=False,
    device=None,
):

    if type(X_train) == torch.Tensor:
        X_train = X_train.cpu().numpy()

    elif type(X_train) == pd.DataFrame:
        X_train = X_train.to_numpy()

    if type(y_train) == torch.Tensor:
        if "Classifier" in model.__class__.__name__:
            y_train = y_train.cpu().long().numpy()
        else:
            y_train = y_train.cpu().float().numpy()

    elif type(y_train) == pd.Series:
        if "Classifier" in model.__class__.__name__:
            y_train = y_train.to_numpy().astype(np.int64)
        else:
            y_train = y_train.to_numpy().astype(np.float32)

    if not tune:
        return {}
    params = match_model_params(model)

    X_train_subset, X_val, y_train_subset, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    if use_tensor:
        X_train_subset, y_train_subset = (
            torch.tensor(X_train_subset),
            torch.tensor(y_train_subset),
        )

        X_val, y_val = torch.tensor(X_val), torch.tensor(y_val)
        X_train_subset, y_train_subset = X_train_subset.float(), y_train_subset.long()
        X_val, y_val = X_val.float(), y_val.long()

        if "Classifier" in model.__class__.__name__:
            y_train_subset = y_train_subset.int()
            y_val = y_val.int()

        if device is not None:
            X_train_subset = X_train_subset.to(device)
            y_train_subset = y_train_subset.to(device)
            X_val = X_val.to(device)
            y_val = y_val.to(device)

    def objective(params):
        for key, value in params.items():
            setattr(model, key, value)

        model.fit(X_train_subset, y_train_subset)
        if hasattr(model, "predict_proba"):
            y_pred = model.predict_proba(X_val)
            if y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]
        else:
            y_pred = model.predict(X_val)
        # metric_func = getattr(tabular_metrics, f"get_{tune_metric}")
        metric_func = accuracy_score
        try:
            score = metric_func(y_val, y_pred)
            score *= get_scoring_direction(tune_metric)
        except Exception as e:
            print(f"Error calculating metric: {e}")
            score = np.nan
            # nan if metric calculation fails, like ROC without predict proba
        return {"loss": score, "status": STATUS_OK}

    trials = Trials()
    best = fmin(
        objective,
        params,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials,
        timeout=max_time,
    )
    best_params = space_eval(params, best)
    print("Best parameters:", best_params)

    return best_params
