import warnings

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier

from tools.pipe import DataProcessingPipeline

warnings.filterwarnings("ignore")


def dataset_benchmark(
    dataset,
    target,
    n_trials=100,
    metrics=[accuracy_score, precision_score, recall_score, f1_score, roc_auc_score],
    test_size=0.2,
    cat_cols=None,
    num_cols=None,
    date_cols=None,
    random_state=42,
):

    np.random.seed(random_state)
    result = []
    pipeline = DataProcessingPipeline(
        cat_features=cat_cols, num_features=num_cols, date_features=date_cols
    )
    cat_pipeline = DataProcessingPipeline(
        cat_features=cat_cols,
        num_features=num_cols,
        date_features=date_cols,
        for_catboost=True,
    )
    X = dataset.drop(target, axis=1)
    y = dataset[target]

    X_prep = pipeline.fit_transform(X)
    X_prep_cat = cat_pipeline.fit_transform(X)
    y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_prep, y, test_size=test_size)

    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(
        X_prep_cat, y, test_size=test_size
    )

    non_cat_models = [TabPFNClassifier(), XGBClassifier(), LGBMClassifier(verbosity=-1)]
    cat_models = [CatBoostClassifier(cat_features=cat_cols, verbose=0)]

    for model in non_cat_models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(model.__class__.__name__, "\n")
        for metric in metrics:
            print(f"{metric.__name__}: {metric(y_test, y_pred)}")
            result.append(
                {
                    "model": model.__class__.__name__,
                    "metric": metric.__name__,
                    "value": metric(y_test, y_pred),
                }
            )
        print("\n")

    for model in cat_models:
        model.fit(X_train_cat, y_train_cat, cat_features=cat_cols)
        y_pred = model.predict(X_test_cat)
        print(model.__class__.__name__, "\n")
        for metric in metrics:
            print(f"{metric.__name__}: {metric(y_test_cat, y_pred)}")
            result.append(
                {
                    "model": model.__class__.__name__,
                    "metric": metric.__name__,
                    "value": metric(y_test_cat, y_pred),
                }
            )
        print("\n")

    non_cat_models = [TabPFNClassifier(), XGBClassifier(), LGBMClassifier()]
    cat_models = [CatBoostClassifier(cat_features=cat_cols, verbose=0)]

    for model in non_cat_models:

        def create_objective(X_train, y_train, X_train_cat, y_train_cat):
            def objective(trial):
                X_train_split, X_valid, y_train_split, y_valid = (
                    train_test_split(X_train, y_train, test_size=0.25)
                    if model.__class__.__name__ != "CatBoostClassifier"
                    else train_test_split(X_train_cat, y_train_cat, test_size=0.25)
                )

                if model.__class__.__name__ == "TabPFNClassifier":
                    params = {
                        "n_estimators": trial.suggest_int(
                            "n_estimators", 1, 2
                        ),  # 2 należy zmienic na 100, jak zapomne to poprawić
                        "softmax_temperature": trial.suggest_float(
                            "softmax_temperature", 0.1, 10.0
                        ),
                        "random_state": random_state,
                        "n_jobs": -1,
                    }
                else:
                    params = {
                        "n_estimators": trial.suggest_int("n_estimators", 1, 100),
                        "max_depth": trial.suggest_int("max_depth", 1, 100),
                        "learning_rate": trial.suggest_float(
                            "learning_rate", 1e-5, 1e-1
                        ),
                        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
                        "colsample_bytree": trial.suggest_float(
                            "colsample_bytree", 0.1, 1.0
                        ),
                        "reg_alpha": trial.suggest_float("reg_alpha", 1e-5, 1e-1),
                        "reg_lambda": trial.suggest_float("reg_lambda", 1e-5, 1e-1),
                        "verbosity": (
                            -1 if model.__class__.__name__ == "LGBMClassifier" else 0
                        ),
                        "random_state": random_state,
                    }

                model.set_params(**params)
                model.fit(X_train_split, y_train_split)
                y_pred = model.predict_proba(X_valid)
                return -log_loss(y_valid, y_pred)

            return objective

        objective_with_args = create_objective(
            X_train, y_train, X_train_cat, y_train_cat
        )

        study = optuna.create_study(direction="maximize")
        study.optimize(objective_with_args, n_trials=n_trials)
        best_params = study.best_params
        model.set_params(**best_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"{model.__class__.__name__}_tuned", "\n")
        for metric in metrics:
            print(f"{metric.__name__}: {metric(y_test, y_pred)}")
            result.append(
                {
                    "model": model.__class__.__name__ + "_tuned",
                    "metric": metric.__name__,
                    "value": metric(y_test, y_pred),
                }
            )
        print("\n")

    for model in cat_models:

        def create_objective(X_train_cat, y_train_cat):
            def objective(trial):
                X_train, X_valid, y_train, y_valid = train_test_split(
                    X_train_cat, y_train_cat, test_size=test_size
                )

                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 1, 100),
                    "max_depth": trial.suggest_int("max_depth", 1, 16),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-1),
                    "subsample": trial.suggest_float("subsample", 0.1, 1.0),
                    "colsample_bylevel": trial.suggest_float(
                        "colsample_bylevel", 0.1, 1.0
                    ),
                    "cat_features": cat_pipeline.cat_features,
                    "verbose": 0,
                    "random_state": random_state,
                }

                model.set_params(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict_proba(X_valid)
                return -log_loss(y_valid, y_pred)

            return objective

        objective_with_args = create_objective(X_train_cat, y_train_cat)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective_with_args, n_trials=n_trials)
        best_params = study.best_params
        model = CatBoostClassifier(**best_params, cat_features=cat_cols, verbose=0)
        model.fit(X_train_cat, y_train_cat)
        y_pred = model.predict(X_test_cat)
        print(f"{model.__class__.__name__}_tuned", "\n")
        for metric in metrics:
            print(f"{metric.__name__}: {metric(y_test_cat, y_pred)}")
            result.append(
                {
                    "model": model.__class__.__name__ + "_tuned",
                    "metric": metric.__name__,
                    "value": metric(y_test_cat, y_pred),
                }
            )
        print("\n")
    return result
