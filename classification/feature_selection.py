from sklearn import base
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelection(BaseEstimator, TransformerMixin):
    # @TODO implement

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X
