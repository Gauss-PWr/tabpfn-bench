from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tabpfn import TabPFNRegressor
from tabular_metrics import get_regression_metrics 
from hyperparameter_tuning import get_model_params 
import torch
import pandas as pd


def match_model(model):
    match model.__class__.__name__:
        case "XGBRegressor":
            return XGBRegressor
        case "LGBMRegressor":
            return LGBMRegressor
        case "CatBoostRegressor":
            return CatBoostRegressor
        case "TabPFNRegressor":
            return TabPFNRegressor
        case _:
            raise ValueError(f"Model {model.__class__.__name__} not supported.")


def benchmark_dataset_regression(
    X_train,
    y_train,
    X_test,
    y_test,
    models=[
        'XGBRegressor',
        'LGBMRegressor',
        'CatBoostRegressor',
        'TabPFNRegressor',
    ],
    csv_path=None,
):  
    if type(X_train) == pd.DataFrame:
        X_train = X_train.to_numpy()
    if type(X_test) == pd.DataFrame:
        X_test = X_test.to_numpy()
    if type(y_train) == pd.Series:
        y_train = y_train.to_numpy()
    if type(y_test) == pd.Series:
        y_test = y_test.to_numpy()
    
    
    results = {}
    for model in models:
        if model == 'TabPFNRegressor':
            X_train = torch.Tensor(X_train)
            X_test = torch.Tensor(X_test)
            y_train = torch.Tensor(y_train)
            y_test = torch.Tensor(y_test)
            X_train, y_train, X_test, y_test = (
                X_train.float(),
                y_train.long(),
                X_test.float(),
                y_test.long(),
            )
        model_default = match_model(model)()
        model_default.fit(X_train, y_train)

        metrics = get_regression_metrics(X_test, y_test, model_default)

        results[model.__class__.__name__+'_default'] = metrics
        
        params = get_model_params(
            model_default,
            X_train,
            y_train,
            tune=True,
            tune_metric="r2",
            max_time=4*60*60, # 4h
            use_tensor=model == 'TabPFNRegressor',
        )
        model_tuned = match_model(model)(**params)
        
        model_tuned.fit(X_train, y_train)
        metrics = get_regression_metrics(X_test, y_test, model_tuned)
        
        results[model.__class__.__name__+'_tuned'] = metrics
        
    if csv_path:
        import pandas as pd
        
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        
    return results