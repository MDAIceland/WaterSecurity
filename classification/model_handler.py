from sklearn.base import BaseEstimator
from data.labeled.preprocessed import LABELED_CITIES
import os
import pickle
import pandas as pd
import numpy as np

from data.model import MODEL_PATH
from sklearn.pipeline import Pipeline
from classification.classifier import Classifier
from classification.feature_selection import FeatureSelection
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from classification import RANDOM_SEED
from data.model.metrics import VALIDATION_METRICS_PATH, TRAINING_METRICS_PATH
from data.model.predictions import PREDICTION_MASK_PATH, FILLED_DATASET_PATH
from utils.geo import is_close, get_place


class TrainingRequired(NotFittedError):
    def __init__(self, obj):
        super.__init__(f"{obj} could not be loaded. Training setp is required")


class InvalidCoordinates(BaseException):
    pass


class ModelHandler:
    """
    Trains and Tests the model, while also computing metrics.
    During training the model is first fitted, then produces predictions for any unlabled points inside the dataset
    """

    def __init__(self):
        self._model = None
        self._dataset = None
        self._valid_metrics = None
        self._train_metrics = None
        self.id_columns = ["city", "coutnry", "latitude", "longitude"]

    @property
    def model(self):
        if self._model is None:
            try:
                from data.model import MODEL

                self._model = MODEL
            except ImportError:
                raise TrainingRequired("Model")
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        with open(os.path.join(MODEL_PATH), "wb") as out:
            pickle.dump(model, out)

    @property
    def dataset(self):
        if self._dataset is None:
            from data.labeled.preprocessed import LABELED_CITIES, RISKS_MAPPING
            from data.dataset import DATASET as dataset

            self.cities = LABELED_CITIES
            self.lab_names = sorted(RISKS_MAPPING.keys())
            self.feat_names = [
                x
                for x in dataset.columns
                if x not in self.lab_names and x not in LABELED_CITIES.columns
            ]
            self.train_mask = dataset[self.lab_names].apply(
                lambda x: all(pd.isnull(x)), axis=1
            )
        return self._dataset

    @staticmethod
    def compute_metrics(y_true, y_pred):
        metrics = {}
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)
        metrics["classification_report"] = classification_report(y_true, y_pred)
        return metrics

    def train(self):
        dataset = self.dataset
        labeled = dataset[self.train_mask]
        labeled[self.lab_names]

        model = {}
        train_metrics = {}
        valid_metrics = {}
        filled_dataset = dataset[self.lab_names + self.id_columns].copy()
        for label in self.lab_names:
            train_mask = ~pd.isnull(dataset[label])
            labeled = dataset.iloc[train_mask, :]
            train_set, valid_set = train_test_split(
                labeled, test_size=0.3, random_state=RANDOM_SEED
            )

            model[label] = Pipeline(
                [
                    ("FeatureSelection", FeatureSelection()),
                    ("Classification", Classifier()),
                ]
            )
            model[label].fit(train_set[self.feat_names], train_set[label])
            train_preds = model[label].predict(train_set[self.feat_names])
            valid_preds = model[label].predict(valid_set[self.feat_names])
            train_metrics[label] = self.compute_metrics(train_set[label], train_preds)
            valid_metrics[label] = self.compute_metrics(valid_set[label], valid_preds)
            model[label].fit(labeled[self.feat_names], labeled[label])

            filled_dataset.loc[~train_mask, label] = model[label].predict(
                dataset.loc[~train_mask, self.feat_names]
            )
        self.model = model
        with open(VALIDATION_METRICS_PATH, "wb") as out:
            pickle.dump(self.valid_metrics, out)
        with open(TRAINING_METRICS_PATH, "wb") as out:
            pickle.dump(self.train_metrics, out)
        filled_dataset.to_csv(FILLED_DATASET_PATH, index=False)
        pd.isnull(dataset[self.lab_names]).to_csv(PREDICTION_MASK_PATH, index=False)

    def test(self, latitude, longitude):

        try:
            from data.model.predictions import FILLED_DATASET, PREDICTION_MASK
        except ImportError:
            raise TrainingRequired("Filled Dataset")
        check_existing = FILLED_DATASET.apply(
            lambda x: is_close((latitude, longitude), (x["latitude"], x["longitude"])),
            axis=1,
        )
        if np.any(check_existing):

            return (
                FILLED_DATASET.loc[check_existing, self.lab_names].iloc[0, :],
                PREDICTION_MASK.loc[check_existing, :].iloc[0],
            )
        try:
            place = get_place(latitude, longitude)
        except AttributeError:
            raise InvalidCoordinates
        from data.unlabeled.preprocessed import COUNTRIES_DATASET

        feats = COUNTRIES_DATASET[COUNTRIES_DATASET["country"] == place["code"]]
        preds = {}
        mask = {}
        for label in self.lab_names:
            preds[label] = self.model[label].predict(feats)
            mask[label] = True
        preds["city"] = place["city"]
        preds["country"] = place["country"]
        return preds, mask
