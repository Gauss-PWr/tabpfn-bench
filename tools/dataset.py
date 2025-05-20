from sklearn.datasets import fetch_openml
import torch


def load_dataset(id, as_tensor=False):
    """
    Load a dataset from OpenML.
    :param name: Name of the dataset.
    :param as_tensor: If True, return the data as PyTorch tensors.
    :return: Tuple of (X, y).
    """
    dataset = fetch_openml(data_id=id)
    X = dataset.data
    y = dataset.target

    if as_tensor:
        import torch

        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.long)

    return X, y

            
openml_ids_reg = [
    42726,
    44957,
    44958,
    531,
    44994,
    42727,
    44959,
    44978,
    44960,
    44965,
    44973,
    42563,
    44980,
    42570,
    43071,
    41021,
    44981,
    44970,
    550,
    41980,
    546,
    541,
    507,
    44967,
    505,
    422,
    42730,
    416,
]
openml_ids_clf = [
    41156,
    40981,
    1464,
    40975,
    40701,
    23,
    31,
    40670,
    188,
    1475,
    4538,
    41143,
    1067,
    3,
    41144,
    12,
    1487,
    1049,
    41145,
    1489,
    1494,
    40900,
    40984,
    40982,
    41146,
    54,
    40983,
    40498,
    181,
]