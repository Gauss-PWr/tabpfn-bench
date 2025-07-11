import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def preprocess_data(
    X_train,
    y_train,
    X_test,
    y_test,
    onehot=True,
    impute=True,
    standardize=True,
    categorical_features=[],
):

    X_train, y_train, X_test, y_test = (
        X_train.cpu().numpy(),
        y_train.cpu().long().numpy(),
        X_test.cpu().numpy(),
        y_test.cpu().long().numpy(),
    )

    if impute:
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

    if onehot:

        def make_pd_from_np(x):
            data = pd.DataFrame(x)
            for c in categorical_features:
                data.iloc[:, c] = data.iloc[:, c].astype("int")
            return data

        X_train, X_test = make_pd_from_np(X_train), make_pd_from_np(X_test)
        transformer = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse=False),
                    categorical_features,
                )
            ],
            remainder="passthrough",
        )

        X_train = transformer.fit_transform(X_train)
        X_test = transformer.transform(X_test)

    if standardize:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test
