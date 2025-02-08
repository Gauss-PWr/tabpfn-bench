from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd


class DateToNumericTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.apply(pd.to_numeric, errors='coerce')
    
    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

class DataProcessingPipeline(BaseEstimator, TransformerMixin):
    
    def __init__(self, cat_features, num_features, date_features):
        self.cat_features = cat_features
        self.num_features = num_features
        self.date_features = date_features
        
        self.num_pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5, weights='distance')),
            ('std_scaler', StandardScaler())
        ])

        self.cat_pipeline = Pipeline([
            ('imputer', KNNImputer(n_neighbors=5, weights='distance')),
            ('one_hot', OneHotEncoder())
        ])

        self.date_pipeline = Pipeline([
             ('date_to_numeric', DateToNumericTransformer()),
            ('imputer', KNNImputer(n_neighbors=5, weights='distance')),
        ])

        self.full_pipeline = ColumnTransformer([
            ('num', self.num_pipeline, self.num_features),
            ('cat', self.cat_pipeline, self.cat_features),
            ('date', self.date_pipeline, self.date_features)
        ])

        
    def fit(self, X, y=None):
        return self.full_pipeline.fit(X)
    
    def transform(self, X):
        return self.full_pipeline.transform(X)
    
    def fit_transform(self, X, y=None):
        return self.full_pipeline.fit_transform(X)