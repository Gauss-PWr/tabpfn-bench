import os

from sklearn.model_selection import train_test_split
import torch

from tools.benchmark_classification import benchmark_dataset_classification
from tools.dataset import get_openml_ids_clf, load_dataset
from tools.preprocess import preprocess_data
import warnings
warnings.filterwarnings("ignore")


def main_clf():
    assert (
        torch.cuda.is_available()
    ), "CUDA is not available. Please check your PyTorch installation."
    print("CUDA is available. Proceeding with benchmarking...")

    json_path = os.path.join("results", "classification_results.json")
    print("Loading datasets...")

    ids = get_openml_ids_clf()
    datasets = [load_dataset(id) for id in ids][:2] # TODO: change to all datasets
    print(f"Loaded {len(datasets)} datasets for classification benchmarking.")

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

        print(f"Benchmarking dataset {i + 1}/{len(datasets)}")

        result = benchmark_dataset_classification(
            X_train,
            y_train,
            X_test,
            y_test,
            models=[
                "XGBClassifier",
                "LGBMClassifier",
                "CatBoostClassifier",
                "TabPFNClassifier",
            ],
            json_path=json_path,
            tune_time=120,  # 120 seconds, TODO change to 4 hours set tune_time=4 * 60 * 60
            dataset_id=ids[i],
        )
        results.append(result)

    print("Classification benchmarking completed.")


if __name__ == "__main__":
    main_clf()
