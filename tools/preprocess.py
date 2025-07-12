import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


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
    if impute:
        def make_pd_from_np(x):
            data = pd.DataFrame(x)
            for c in categorical_features:
                data.iloc[:, c] = data.iloc[:, c].astype("category")
            return data
        
        X_train_df = make_pd_from_np(X_train)
        X_test_df = make_pd_from_np(X_test)
        
        numerical_features = [i for i in range(X_train_df.shape[1]) if i not in categorical_features]
        
        imputer = ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="mean"), numerical_features),
                ("cat", SimpleImputer(strategy="most_frequent"), categorical_features),
            ],
            remainder="passthrough",
        )
        
        X_train = imputer.fit_transform(X_train_df)
        X_test = imputer.transform(X_test_df)
    
    if onehot:
        if not impute:
            def make_pd_from_np(x):
                data = pd.DataFrame(x)
                for c in categorical_features:
                    data.iloc[:, c] = data.iloc[:, c].astype("category")
                return data
            X_train, X_test = make_pd_from_np(X_train), make_pd_from_np(X_test)
        
        transformer = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
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