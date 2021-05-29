import sys

sys.path.append("water_security")

from classification.feature_selection import (
    FeatureSelectionAndGeneration,
)
from classification.model_handler import ModelHandler
from sklearn.pipeline import Pipeline
from classification import RANDOM_SEED
from sklearn.model_selection import train_test_split
import pandas as pd
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
                (
                    "FeatureSelection",
                    FeatureSelectionAndGeneration(verbose=True, apply_selection=True),
                ),
            ]
        )
        d = model[label].fit_transform(train_set[handler.feat_names], train_set[label])
        assert len(d.shape) == 2
        v = model[label].transform(
            valid_set[[x for x in handler.feat_names if x != "population"]]
        )
        assert tuple(d.columns.tolist()) == tuple(v.columns.tolist())
