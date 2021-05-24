from classification.feature_selection import FeatureSelectionAndGeneration
from classification.model_handler import ModelHandler
from sklearn.pipeline import Pipeline
from classification import RANDOM_SEED
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from data.labeled.preprocessed import RISKS_MAPPING


def test_feature_selection_model():
    handler = ModelHandler()
    dataset = handler.dataset
    labeled = dataset[handler.train_mask]
    model = {}
    train_metrics = {}
    valid_metrics = {}
    filled_dataset = dataset[handler.id_columns + handler.lab_names].copy()
    for label in handler.lab_names:
        print("Risk:", RISKS_MAPPING[label])
        train_mask = ~pd.isnull(dataset[label])
        labeled = dataset.loc[train_mask, :]
        train_set, valid_set = train_test_split(
            labeled, test_size=0.3, random_state=RANDOM_SEED
        )

        model[label] = Pipeline(
            [
                ("FeatureSelection", FeatureSelectionAndGeneration()),
            ]
        )
        print(
            "\n".join(
                model[label]
                .fit_transform(train_set[handler.feat_names], train_set[label])
                .columns.tolist()
            )
        )
