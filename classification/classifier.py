from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np


class Classifier(BaseEstimator, RegressorMixin):
    def fit(self, x_data, y_data):
        return self

    def predict(self, x_data):
        # @TODO implement
        return np.zeros(x_data.shape[0])
