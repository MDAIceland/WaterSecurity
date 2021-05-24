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
    for label in handler.lab_names:
        print(f"\n\n**Risk: {RISKS_MAPPING[label]}**\n\n")
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
        d = model[label].fit_transform(train_set[handler.feat_names], train_set[label])
        assert len(d.shape) == 2


def test_produce_augmented_features():
    handler = ModelHandler()
    dataset = handler.dataset
    train_set = dataset[handler.train_mask]
    model = {}
    label = handler.lab_names[0]
    model[label] = FeatureSelectionAndGeneration(apply_selection=False)

    d = model[label].fit_transform(train_set[handler.feat_names], train_set[label])
    d.to_csv("data/dataset/tmp_augmented_dataset.csv")
