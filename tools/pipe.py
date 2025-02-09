import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class DateToNumericTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(pd.to_numeric, errors="coerce")

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class DataProcessingPipeline(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        cat_features=None,
        num_features=None,
        date_features=None,
        for_catboost=False,
    ):
        self.cat_features = cat_features
        self.num_features = num_features
        self.date_features = date_features
        self.for_catboost = for_catboost

        self.num_imputer = KNNImputer(n_neighbors=5, weights="distance")
        self.num_scaler = StandardScaler()
        self.num_pipeline = (
            Pipeline([("imputer", self.num_imputer), ("std_scaler", self.num_scaler)])
            if num_features
            else None
        )

        self.cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
        self.cat_encoder = OneHotEncoder(sparse_output=False)
        self.cat_pipeline = (
            Pipeline([("imputer", self.cat_imputer), ("one_hot", self.cat_encoder)])
            if cat_features and not for_catboost
            else Pipeline([("imputer", self.cat_imputer)]) if cat_features else None
        )

        self.date_to_numeric = DateToNumericTransformer()
        self.date_imputer = KNNImputer(n_neighbors=5, weights="distance")
        self.date_pipeline = (
            Pipeline(
                [
                    ("date_to_numeric", self.date_to_numeric),
                    ("imputer", self.date_imputer),
                ]
            )
            if date_features
            else None
        )

        transformers = []
        if self.num_pipeline:
            transformers.append(("num", self.num_pipeline, self.num_features))
        if self.cat_pipeline:
            transformers.append(("cat", self.cat_pipeline, self.cat_features))
        if self.date_pipeline:
            transformers.append(("date", self.date_pipeline, self.date_features))

        self.full_pipeline = ColumnTransformer(
            transformers, remainder="passthrough", verbose_feature_names_out=False
        )

    def fit(self, X, y=None):
        self.full_pipeline.fit(X)
        return self

    def transform(self, X):
        transformed_data = self.full_pipeline.transform(X)
        feature_names = self.full_pipeline.get_feature_names_out()
        return pd.DataFrame(transformed_data, columns=feature_names)

    def fit_transform(self, X, y=None):
        transformed_data = self.full_pipeline.fit_transform(X)
        feature_names = self.full_pipeline.get_feature_names_out()
        return pd.DataFrame(transformed_data, columns=feature_names)
