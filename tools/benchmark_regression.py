import os
import json

import torch
from catboost import CatBoostRegressor
from tools.hyperparameter_tuning import get_model_params
from lightgbm import LGBMRegressor
from tabpfn import TabPFNRegressor
from tools.tabular_metrics import evaluate_regression
from xgboost import XGBRegressor
import gc
import warnings
warnings.filterwarnings("ignore")

def match_model(model):
    match model:
        case "XGBRegressor":
            return XGBRegressor
        case "LGBMRegressor":
            return LGBMRegressor
        case "CatBoostRegressor":
            return CatBoostRegressor
        case "TabPFNRegressor":
            return TabPFNRegressor
        case _:
            raise ValueError(f"Model {model} not supported.")

def get_default_params(model_name):
    if model_name == "TabPFNRegressor":
        return {
            "device": "cuda",
            "ignore_pretraining_limits": True,
        }
    elif model_name == "CatBoostRegressor":
        return {
            "task_type": "GPU",
            "devices": "0",
            "verbose": False,
        }
    elif model_name == "LGBMRegressor":
        return {
            "device": "gpu",
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
            'verbosity': -1,
        }
    elif model_name == "XGBRegressor":
        return {
            "tree_method": "gpu_hist",
            "gpu_id": 0,
            "verbosity": 0,
        }
    else:
        raise ValueError(f"Model {model_name} not supported.")

def benchmark_dataset_regression(
    X_train,
    y_train,
    X_test,
    y_test,
    models=[
        "XGBRegressor",
        "LGBMRegressor",
        "CatBoostRegressor",
        "TabPFNRegressor",
    ],
    json_path=None,
    tune_time=4 * 60 * 60,  # 4 hours
    dataset_id=None,
):
    assert dataset_id is not None, "dataset_id must be provided"

    if json_path:
        assert json_path.endswith(".json"), "json_path must end with .json"
    import pandas as pd

    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.to_numpy()
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_numpy()
    if isinstance(y_test, pd.Series):
        y_test = y_test.to_numpy()

    X_train_orig = X_train.copy()
    y_train_orig = y_train.copy()
    X_test_orig = X_test.copy()
    y_test_orig = y_test.copy()

    results = {}

    for model_name in models:
        print(f"Processing {model_name}...")

        X_train = X_train_orig.copy()
        y_train = y_train_orig.copy()
        X_test = X_test_orig.copy()
        y_test = y_test_orig.copy()

        if model_name == "TabPFNRegressor":
            assert (
                torch.cuda.is_available()
            ), "CUDA is not available. Please check your PyTorch installation."
            X_train = torch.tensor(X_train, dtype=torch.float32)
            X_test = torch.tensor(X_test, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)
        try:

            default_params = get_default_params(model_name)
            model_default = match_model(model_name)(**default_params)
            model_default.fit(X_train, y_train)

            metrics = evaluate_regression(
                X_test,
                y_test,
                model_default,
                use_tensor=model_name == "TabPFNRegressor",
            )

            results[f"{model_name}_default"] = metrics

            del model_default
            gc.collect()

            print(f"Tuning hyperparameters for {model_name}...")
            unfitted_model = match_model(model_name)()
            params = get_model_params(
                unfitted_model,
                X_train,
                y_train,
                tune=True,
                max_time=tune_time,
                use_tensor=model_name == "TabPFNRegressor",
            )

            del unfitted_model
            gc.collect()

            # Tuned model
            model_tuned = match_model(model_name)(**params)
            model_tuned.fit(X_train, y_train)
            metrics = evaluate_regression(
                X_test, y_test, model_tuned, use_tensor=model_name == "TabPFNRegressor"
            )
            results[f"{model_name}_tuned"] = metrics

            del model_tuned

        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            results[f"{model_name}_default"] = {"error": str(e)}
            results[f"{model_name}_tuned"] = {"error": str(e)}

        if model_name == "TabPFNRegressor":
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        gc.collect()
        print(f"Completed {model_name}")

    if json_path:

        file_exists = os.path.exists(json_path) and os.path.getsize(json_path) > 0

        if file_exists:
            try:
                with open(json_path, "r") as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, dict):
                    existing_data = dict(existing_data)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = {}
        else:
            existing_data = {}
        existing_data[dataset_id] = results
        with open(json_path, "w") as f:
            json.dump(existing_data, f, indent=2)

    return results
