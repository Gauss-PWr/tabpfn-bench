from xgboost import XGBClassifier
from lgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
from tabular_metrics import get_classification_metrics 
from hyperparameter_tuning import get_model_params 
import torch


def match_model(model):
    match model.__class__.__name__:
        case "XGBClassifier":
            return XGBClassifier
        case "LGBMClassifier":
            return LGBMClassifier
        case "CatBoostClassifier":
            return CatBoostClassifier
        case "TabPFNClassifier":
            return TabPFNClassifier
        case _:
            raise ValueError(f"Model {model.__class__.__name__} not supported.")


def benchmark_dataset_classification(
    X_train,
    y_train,
    X_test,
    y_test,
    models=[
        'XGBClassifier',
        'LGBMClassifier',
        'CatBoostClassifier',
        'TabPFNClassifier',
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
    
    
    classification_metrics = get_classification_metrics()
    results = {}
    for model in models:
        if model == 'TabPFNClassifier':
            X_train = torch.Tensor(X_train)
            X_test = torch.Tensor(X_test)
            y_train = torch.Tensor(y_train)
            y_test = torch.Tensor(y_test)
            X_train, y_train, X_test, y_test = (
                X_train.float(),
                y_train.int(),
                X_test.float(),
                y_test.int(),
            )
        model_default = match_model(model)()
        model_default.fit(X_train, y_train)

        y_pred = model_default.predict(X_test)
        metrics = classification_metrics(y_test, y_pred)

        results[model.__class__.__name__+'_default'] = metrics
        
        params = get_model_params(
            model_default,
            X_train,
            y_train,
            tune=True,
            tune_metric="f1",
            max_time=4*60*60, # 4h
            use_tensor=False,
        )
        model_tuned = match_model(model)(**params)
        
        model_tuned.fit(X_train, y_train)
        y_pred = model_tuned.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)
        
        results[model.__class__.__name__+'_tuned'] = metrics
        
    if csv_path:
        import pandas as pd
        
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)
        
    return results