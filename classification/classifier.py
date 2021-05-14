from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np


class Classifier(BaseEstimator, RegressorMixin):
    # @TODO implement
    def fit(self, x_data, y_data):
        """
        x_data: the nxm features
        y_data: the n labels, with values 0,1,2 or 3
        """
        return self

    def predict(self, x_data):
        """
        x_data: the nxm features
        Returns n predictions, which have values 0 to 3, they can be floats
        """
        return np.zeros(x_data.shape[0])
