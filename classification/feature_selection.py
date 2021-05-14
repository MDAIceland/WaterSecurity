from sklearn import base
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureSelectionAndGeneration(BaseEstimator, TransformerMixin):
    # @TODO implement

    def fit(self, x_data, y_data):
        """
        Fits to nxm features x_data and n predictions y_data
        """
        return self

    def transform(self, x_data):
        """
        Transforms x_data from nxm to kxm
        """
        return x_data
