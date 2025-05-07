import tqdm
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval, rand
import time
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import torch

from tools.constants import xgb_params, lgbm_params, catboost_params, tabpfn_params
import tools.tabular_metrics as tabular_metrics


def get_scoring_direction(metric):
    # not implemented yet
    return 1  # 1 for maximization, -1 for minimization


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
):
    start = time.time()
    params = match_model_params(model)

    def stop(trial):
        return time.time() - start > max_time, []

    if not tune:
        return {}

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

        metric_func = getattr(tabular_metrics, f"get_{tune_metric}")
        try:
            score = metric_func(y_val, y_pred)
            score *= get_scoring_direction(tune_metric)

        except Exception as e:
            print(f"Error calculating metric: {e}")
            score = (
                np.nan
            )  # nan if metric calculation fails, like ROC without predict proba

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
