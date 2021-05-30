from typing import Dict
from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
import xgboost as xgb


class Classifier(BaseEstimator, RegressorMixin):
    def __init__(self, risk):
        self.regressor = None
        self.risk = risk
        self._parameters = None

    @property
    def parameters(self) -> Dict:
        if self._parameters is None:
            from data.model import MODEL_BEST_PARAMS

            self._parameters = MODEL_BEST_PARAMS[self.risk]
        return self._parameters

    def fit(self, x_data: pd.DataFrame, y_data):
        """
        x_data: the nxm features
        y_data: the n labels, with values 0,1,2 or 3
        """
        self.regressor = xgb.XGBRegressor(**self.parameters).fit(x_data, y_data)
        self.feature_importances_ = pd.Series(
            self.regressor.feature_importances_, index=x_data.columns.tolist()
        )
        self.feature_importances_ = self.feature_importances_[
            self.feature_importances_ > 0
        ].sort_values(ascending=False)
        return self

    def predict(self, x_data):
        """
        x_data: the nxm features
        Returns n predictions, which have values 0 to 3, they can be floats
        """
        return self.regressor.predict(x_data)
