import os

from sklearn.model_selection import train_test_split
import torch

from tools.benchmark_regression import benchmark_dataset_regression
from tools.dataset import get_openml_ids_reg, load_dataset
from tools.preprocess import preprocess_data
import warnings
warnings.filterwarnings("ignore")

def main():
    assert torch.cuda.is_available(), "CUDA is not available. Please check your PyTorch installation."
    print("CUDA is available. Proceeding with benchmarking...")

    json_path = os.path.join("results", "regression_results.json")
    print("Loading datasets...")

    ids = get_openml_ids_reg()
    datasets = [load_dataset(id) for id in ids][:1]
    print(f"Loaded {len(datasets)} datasets for regression benchmarking.")

    results = []
    for i, (X, y, categorical_indices) in enumerate(datasets):
        print(f"Processing dataset {i + 1}/{len(datasets)}: {X.shape}, {y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, y_train, X_test, y_test = preprocess_data(
            X_train,
            y_train,
            X_test,
            y_test,
            onehot=True,
            impute=True,
            standardize=True,
            categorical_features=categorical_indices,
        )
        print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        print(f"Benchmarking dataset {i + 1}/{len(datasets)}")

        result = benchmark_dataset_regression(
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
            json_path=json_path,
            tune_time=120,  # 2 min, TODO change to 4 hours set tune_time=4 * 60 * 60
            dataset_id=ids[i],
        )
        results.append(result)

    print("Benchmarking completed.")

    # Print results
    for i, result in enumerate(results):
        print(f"Results for dataset {i + 1}: {result}")


if __name__ == "__main__":
    main()