import importlib
import os
import pickle
from typing import Generator

import numpy as np
import pandas as pd
import shap
from data.model import MODEL_PATH
from data.model.metrics import TRAINING_METRICS_PATH, VALIDATION_METRICS_PATH
from data.model.predictions import FILLED_DATASET_PATH, PREDICTION_MASK_PATH
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from utils.geo import (
    get_average_1k_population_density,
    get_elevation,
    get_place,
    is_close,
)

from classification import RANDOM_SEED
from classification.classifier import Classifier
from classification.feature_selection import FeatureSelectionAndGeneration


def regression_report(y_true, y_pred):
    """
    Returns a regression report, including Mean Absolute and Squared Errors and Explained Variance
    """
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "Explained Variance": explained_variance_score(y_true, y_pred),
    }


class TrainingRequired(NotFittedError):
    def __init__(self, obj):
        super().__init__(f"{obj} could not be loaded. Training model is required")


class InvalidCoordinates(BaseException):
    pass


class ModelHandler:
    """
    Trains and Tests the model, while also computing metrics.
    During training the model is first fitted, then produces predictions for any unlabled points inside the dataset
    During testing, it receives latitude, longitude, computes the required features for the city, merges with the country features,
    uses the model to predict the output and also output the shap values associated with it.
    """

    def __init__(self):
        self._model = None
        self._explainers = None
        self._dataset = None
        self._valid_metrics = None
        self._train_metrics = None
        self._filled_dataset = None
        self.train_mask = None
        self.feat_names = None
        self.lab_names = None
        # The id columns to remain in the filled dataset
        self.id_columns = [
            "city",
            "country",
            "country_code",
            "c40",
            "latitude",
            "longitude",
            "population_1k_density",
            "elevation",
        ]
        # The id columns to consider also as features
        self.feat_id_columns = [
            "latitude",
            "longitude",
            "population_1k_density",
            "elevation",
        ]

    @property
    def model(self) -> Pipeline:
        """
        If model is not defined, try to loaded from disk
        """
        if self._model is None:
            try:
                from data.model import MODEL, MODEL_PATH

                print(f"Loaded model from {MODEL_PATH}.")
                self._model = MODEL
            except ImportError:
                raise TrainingRequired("Model")
        return self._model

    @model.setter
    def model(self, model: Pipeline):
        self._model = model

    def save_model(self) -> None:
        """
        Saves model to memory
        """
        with open(os.path.join(MODEL_PATH), "wb") as out:
            pickle.dump(self.model, out)
        import data.model

        importlib.reload(data.model)

    @property
    def dataset(self) -> pd.DataFrame:
        """
        The dataset for the training step.
        When it is loaded the first time, several variables are defined:
            - lab_names: the labels names/columns of the dataset
            - unique_labs: the unique labels values
            - feat_names: the features names/columns of the dataset
            - train_mask: the mask that refers to the cities that are labeled at least for one risk
        """
        if self._dataset is None:
            from data.dataset import DATASET as dataset
            from data.labeled.preprocessed import LABELED_CITIES, RISKS_MAPPING

            self.lab_names = sorted(RISKS_MAPPING.keys())
            self.unique_labs = np.unique(dataset[self.lab_names].T.stack().values)
            self.feat_names = [
                x
                for x in dataset.columns
                if x not in self.lab_names
                and (x in self.feat_id_columns or x not in self.id_columns)
            ]
            self.train_mask = dataset[self.lab_names].apply(
                lambda x: not all(pd.isnull(x)), axis=1
            )
            self._dataset = dataset
        return self._dataset

    @property
    def filled_dataset(self) -> pd.DataFrame:
        """
        The dataset that has filled labels, which were produced from the predictions
        """
        if self._filled_dataset is None:
            try:
                self._filled_dataset = pd.read_csv(FILLED_DATASET_PATH)
            except IOError:
                raise TrainingRequired("Filled Dataset")
        return self._filled_dataset

    @filled_dataset.setter
    def filled_dataset(self, dataset: pd.DataFrame):
        self._filled_dataset = dataset

    def compute_metrics(self, y_true, y_pred):
        """
        Compute metrics for regression labels of size nx1
        """
        metrics = {}

        # Interpolate predictions to labels, eg convert 0.2 to 0, 0.7 to 1 etc.
        y_pred_interp = self.unique_labs[
            np.abs(np.reshape(self.unique_labs, (-1, 1)) - y_pred).argmin(axis=0)
        ]
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred_interp)
        metrics["classification_report"] = classification_report(
            y_true, y_pred_interp, output_dict=True
        )
        metrics["regression_report"] = regression_report(y_true, y_pred)
        return metrics

    @property
    def is_fitted(self) -> bool:
        """
        Tries to load model from memory/disk, if it fails, returns False, else returns True
        """
        try:
            self.model
        except TrainingRequired:
            return False
        return True

    def get_total_train_val_set_per_risk(self) -> Generator:
        dataset = self.dataset
        labeled = dataset[self.train_mask]
        for label in self.lab_names:
            train_mask = ~pd.isnull(dataset[label])
            labeled = dataset.loc[train_mask, :]
            train_set, valid_set = train_test_split(
                labeled, test_size=0.3, random_state=RANDOM_SEED
            )
            yield (label, labeled, [train_set, valid_set])

    @property
    def explainers(self):
        """
        The SHAP explainers per model
        """
        if self._explainers is None:
            self._explainers = {
                label: shap.Explainer(
                    self.model[label].named_steps["Classification"].regressor,
                )
                for label in self.model
            }
        return self._explainers

    def train(self) -> None:
        """
        - Trains 7 different models, one per each different water security risk.
        - Applies feature selection and generation per different model.
        - Keeps 0.3 validation size, computes classification metrics, saves them, then fits each model to the whole available dataset for each risk.
        - Creates the filled dataset and saves it to disk
        - Creates the prediction mask (what labels from the filled dataset were predicted) and saves it to memory
        """
        model = {}
        train_metrics = {}
        valid_metrics = {}
        filled_dataset: pd.DataFrame = None

        for (
            label,
            labeled,
            [train_set, valid_set],
        ) in self.get_total_train_val_set_per_risk():

            model[label] = Pipeline(
                [
                    ("FeatureSelection", FeatureSelectionAndGeneration(feats_num=200)),
                    ("Classification", Classifier(label)),
                ]
            )
            model[label].fit(train_set[self.feat_names], train_set[label])
            train_preds = model[label].predict(train_set[self.feat_names])
            valid_preds = model[label].predict(valid_set[self.feat_names])
            train_metrics[label] = self.compute_metrics(train_set[label], train_preds)
            valid_metrics[label] = self.compute_metrics(valid_set[label], valid_preds)
            model[label].fit(labeled[self.feat_names], labeled[label])
            if filled_dataset is None:
                filled_dataset = self.dataset[self.id_columns + self.lab_names].copy()
            filled_dataset.loc[~self.train_mask, label] = model[label].predict(
                self.dataset.loc[~self.train_mask, self.feat_names]
            )
        self.model = model
        self.save_model()
        with open(VALIDATION_METRICS_PATH, "wb") as out:
            pickle.dump(valid_metrics, out)
        with open(TRAINING_METRICS_PATH, "wb") as out:
            pickle.dump(train_metrics, out)
        self.filled_dataset = filled_dataset
        self.filled_dataset.to_csv(FILLED_DATASET_PATH, index=False)
        prediction_mask = self.filled_dataset[self.id_columns + self.lab_names]
        prediction_mask[self.lab_names] = pd.isnull(self.dataset[self.lab_names])
        prediction_mask.to_csv(PREDICTION_MASK_PATH, index=False)
        import data.model.metrics

        importlib.reload(data.model.metrics)
        import data.model.predictions

        importlib.reload(data.model.predictions)

    def test(self, latitude: float, longitude: float):
        """
        Given a specific latitude and longitude value, either returns saved predictions from the filled dataset, if the point is close to the
        ones that have already been predicted, or uses a REST API to load the country to which the latitude and longitude refer, uses the country data
        to create the feature vector and computes the prediction using the trained models.
        Returns the series of the found labels, which also contain city and country,
        and the series of booleans which shows which predictions were predicted and which where not.
        IF it is an online prediction, it also returns the shap values associated with the prediction.
        """
        try:
            from data.model.predictions import FILLED_DATASET, PREDICTION_MASK
        except ImportError:
            raise TrainingRequired("Filled Dataset")
        check_existing = FILLED_DATASET.apply(
            lambda x: is_close((latitude, longitude), (x["latitude"], x["longitude"])),
            axis=1,
        )
        if np.any(check_existing):
            labs = list(sorted(self.model.keys()))
            return (
                FILLED_DATASET.loc[check_existing, labs + ["city", "country"]].iloc[0],
                PREDICTION_MASK.loc[check_existing, labs].iloc[0],
            )
        return self._test_online_prediction(latitude, longitude)

    def _test_online_prediction(self, latitude, longitude):
        try:
            place = get_place(latitude, longitude)
        except AttributeError:
            raise InvalidCoordinates
        population_density = get_average_1k_population_density(latitude, longitude)
        elevation = get_elevation(latitude, longitude)

        from data.unlabeled import COUNTRIES_DATASET

        feats = COUNTRIES_DATASET.loc[place["code"]].copy()
        feats["latitude"] = latitude
        feats["longitude"] = longitude
        feats["population_1k_density"] = population_density
        feats["elevation"] = elevation
        feats["population"] = None
        preds = {}
        mask = {}
        shap_values = {}
        for label in self.model:
            preds[label] = self.model[label].predict(feats)[0]
            mask[label] = True
            transformed = (
                self.model[label].named_steps["FeatureSelection"].transform(feats)
            )
            shap_values[label] = self.explainers[label](transformed)

        preds["city"] = place["city"]
        preds["country"] = place["country"]
        return pd.Series(preds), pd.Series(mask), shap_values
