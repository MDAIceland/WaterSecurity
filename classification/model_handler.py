from sklearn.base import BaseEstimator
from data.labeled.preprocessed import LABELED_CITIES
import os
import pickle
import pandas as pd
import numpy as np
import importlib
from data.model import MODEL_PATH
from sklearn.pipeline import Pipeline
from classification.classifier import Classifier
from classification.feature_selection import FeatureSelectionAndGeneration
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from classification import RANDOM_SEED
from data.model.metrics import VALIDATION_METRICS_PATH, TRAINING_METRICS_PATH
from data.model.predictions import PREDICTION_MASK_PATH, FILLED_DATASET_PATH
from utils.geo import is_close, get_place, get_average_1k_population_density


class TrainingRequired(NotFittedError):
    def __init__(self, obj):
        super().__init__(f"{obj} could not be loaded. Training model is required")


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
        self._filled_dataset = None
        self.train_mask = None
        self.feat_names = None
        self.lab_names = None
        # The id columns to remain in the filled dataset
        self.id_columns = ["city", "country", "latitude", "longitude"]

    @property
    def model(self) -> Pipeline:
        """
        If model is not defined, try to loaded from disk
        """
        if self._model is None:
            try:
                from data.model import MODEL

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
            from data.labeled.preprocessed import LABELED_CITIES, RISKS_MAPPING
            from data.dataset import DATASET as dataset

            self.lab_names = sorted(RISKS_MAPPING.keys())
            self.unique_labs = np.unique(dataset[self.lab_names].T.stack().values)
            self.feat_names = [
                x
                for x in dataset.columns
                if x not in self.lab_names and x not in LABELED_CITIES.columns
            ]
            self.train_mask = dataset[self.lab_names].apply(
                lambda x: all(pd.isnull(x)), axis=1
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
        metrics["classification_report"] = classification_report(y_true, y_pred)
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

    def train(self) -> None:
        """
        - Trains 7 different models, one per each different water security risk.
        - Applies feature selection and generation per different model.
        - Keeps 0.3 validation size, computes classification metrics, saves them, then fits each model to the whole available dataset for each risk.
        - Creates the filled dataset and saves it to disk
        - Creates the prediction mask (what labels from the filled dataset were predicted) and saves it to memory
        """
        dataset = self.dataset
        labeled = dataset[self.train_mask]
        labeled[self.lab_names]

        model = {}
        train_metrics = {}
        valid_metrics = {}
        filled_dataset = dataset[self.id_columns + self.lab_names].copy()
        for label in self.lab_names:
            train_mask = ~pd.isnull(dataset[label])
            labeled = dataset.loc[train_mask, :]
            train_set, valid_set = train_test_split(
                labeled, test_size=0.3, random_state=RANDOM_SEED
            )

            model[label] = Pipeline(
                [
                    ("FeatureSelection", FeatureSelectionAndGeneration()),
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
        self.save_model()
        with open(VALIDATION_METRICS_PATH, "wb") as out:
            pickle.dump(valid_metrics, out)
        with open(TRAINING_METRICS_PATH, "wb") as out:
            pickle.dump(train_metrics, out)
        self.filled_dataset = filled_dataset
        self.filled_dataset.to_csv(FILLED_DATASET_PATH, index=False)
        prediction_mask = self.filled_dataset[self.id_columns + self.lab_names]
        prediction_mask[self.lab_names] = pd.isnull(dataset[self.lab_names])
        prediction_mask.to_csv(PREDICTION_MASK_PATH, index=False)
        import data.model.metrics

        importlib.reload(data.model.metrics)
        import data.model.predictions

        importlib.reload(data.model.predictions)

    def test(self, latitude, longitude):
        """
        Given a specific latitude and longitude value, either returns saved predictions from the filled dataset, if the point is close to the
        ones that have already been predicted, or uses a REST API to load the country to which the latitude and longitude refer, uses the country data
        to create the feature vector and computes the prediction using the trained models.
        Returns the series of the found labels, which also contain city and country, and the series of booleans which shows which predictions were predicted and which where not.
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
        try:
            place = get_place(latitude, longitude)
        except AttributeError:
            raise InvalidCoordinates
        population_density = get_average_1k_population_density(latitude, longitude)

        from data.unlabeled import COUNTRIES_DATASET

        feats = COUNTRIES_DATASET.loc[place["code"]].copy()
        feats["population_1k_density"] = population_density
        preds = {}
        mask = {}
        for label in self.model:
            preds[label] = self.model[label].predict(feats)[0]
            mask[label] = True
        preds["city"] = place["city"]
        preds["country"] = place["country"]
        return pd.Series(preds), pd.Series(mask)
