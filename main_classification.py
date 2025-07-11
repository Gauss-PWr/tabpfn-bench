import os

from sklearn.model_selection import train_test_split

from tools.benchmark_classification import benchmark_dataset_classification
from tools.dataset import get_openml_ids_reg, load_dataset
from tools.preprocess import preprocess_data


def main():
    csv_path = os.path.join("results", "classification_results.csv")
    print("Loading datasets...")

    ids = get_openml_ids_reg()
    datasets = [load_dataset(id) for id in ids]
    print(f"Loaded {len(datasets)} datasets for classification benchmarking.")

    results = []
    for i, (X, y, categorical_indices) in enumerate(datasets):
        print(f"Processing dataset {i + 1}/{len(datasets)}: {X.shape}, {y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_test, y_train, y_test = preprocess_data(
            X_train,
            y_train,
            X_test,
            y_test,
            onehot=True,
            impute=True,
            standardize=True,
            categorical_features=categorical_indices,
        )

        print(f"Benchmarking dataset {i + 1}/{len(datasets)}")

        result = benchmark_dataset_classification(
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
            csv_path=csv_path,
        )
        results.append(result)

    print("Benchmarking completed.")

    # Print results
    for i, result in enumerate(results):
        print(f"Results for dataset {i + 1}: {result}")
