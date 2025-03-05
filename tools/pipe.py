import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from missforrest import MissForrest
from imblearn.over_sampling import SMOTE


class DateToNumericTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(pd.to_numeric, errors="coerce")

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)


class DataProcessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        cat_features=[],
        ord_cat_features=[],
        num_features=[],
        date_features=[],
        for_catboost=False,
        imputer="mean",
        use_smote=True,
    ):
        self.cat_features = cat_features
        self.ord_cat_features = ord_cat_features
        self.num_features = num_features
        self.date_features = date_features
        self.for_catboost = for_catboost
        self.use_smote = use_smote
        self.imputer = imputer

        match imputer:
            case "none" | "passthrough":
                self._num_imputer = None
            case "knn":
                self._num_imputer = KNNImputer(n_neighbors=5, weights="distance")
            case "missforrest":
                self._num_imputer = MissForrest()
            case "regression":
                self._num_imputer = IterativeImputer()
            case "mean":
                self._num_imputer = SimpleImputer(strategy="mean")
            case "median":
                self._num_imputer = SimpleImputer(strategy="median")
            case "most_frequent":
                self._num_imputer = SimpleImputer(strategy="most_frequent")
            case _:
                raise ValueError(f"Invalid imputer: {imputer}")
        self._num_scaler = StandardScaler()
        self._num_pipeline = (
            Pipeline([("imputer", self._num_imputer), ("std_scaler", self._num_scaler)])
            if num_features
            else None
        )

        self._cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
        self._cat_encoder = OneHotEncoder(sparse_output=False)
        self._cat_pipeline = (
            Pipeline([("imputer", self._cat_imputer), ("one_hot", self._cat_encoder)])
            if cat_features and not for_catboost
            else Pipeline([("imputer", self._cat_imputer)]) if cat_features else None
        )

        self._ord_cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
        self._ord_cat_encoder = OrdinalEncoder()
        self._ord_cat_pipeline = (
            Pipeline(
                [("imputer", self._ord_cat_imputer), ("ordinal", self._ord_cat_encoder)]
            )
            if ord_cat_features and for_catboost
            else (
                Pipeline([("imputer", self._ord_cat_imputer)])
                if ord_cat_features
                else None
            )
        )

        self._date_to_numeric = DateToNumericTransformer()
        self._date_imputer = KNNImputer(n_neighbors=5, weights="distance")
        self._date_pipeline = (
            Pipeline(
                [
                    ("date_to_numeric", self._date_to_numeric),
                    ("imputer", self._date_imputer),
                ]
            )
            if date_features
            else None
        )

        transformers = []
        if self._num_pipeline:
            transformers.append(("num", self._num_pipeline, self.num_features))
        if self._cat_pipeline:
            transformers.append(("cat", self._cat_pipeline, self.cat_features))
        if self._date_pipeline:
            transformers.append(("date", self._date_pipeline, self.date_features))

        self._full_pipeline = ColumnTransformer(
            transformers, remainder="passthrough", verbose_feature_names_out=True
        )
        
        if self.use_smote:
            self._smote = SMOTE(random_state=42)
        else:
            self._smote = None
        
    def fit(self, X, y=None):
        self._full_pipeline.fit(X)
        return self

    def transform(self, X, y):
        if self.use_smote:
            X, y = self._smote.fit_resample(X.copy(), y.copy())
            
        transformed_data = self._full_pipeline.transform(X)
        feature_names = self._full_pipeline.get_feature_names_out()
        return pd.DataFrame(transformed_data, columns=feature_names), y

    def fit_transform(self, X, y=None):
        transformed_data, y = self._full_pipeline.fit_transform(X, y)
        feature_names = self._full_pipeline.get_feature_names_out()
        return pd.DataFrame(transformed_data, columns=feature_names), y

    def get_feature_names_out(self):
        return self._full_pipeline.get_feature_names_out()
