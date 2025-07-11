import numpy as np
import openml


def load_dataset(id):
    dataset = openml.datasets.get_dataset(id)
    X, y, categorical_indicator, _ = dataset.get_data(
        target=dataset.default_target_attribute, dataset_format="array"
    )

    X = np.array(X)
    y = np.array(y)
    categorical_indices = [
        i for i, is_cat in enumerate(categorical_indicator) if is_cat
    ]

    return X, y, categorical_indices


def get_openml_ids_reg() -> list[int]:
    return [
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


def get_openml_ids_clf() -> list[int]:
    return [
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
