import sys

sys.path.append("water_security")

from classification.feature_selection import (
    FeatureSelectionAndGeneration,
)
from classification.model_handler import ModelHandler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from data.labeled.preprocessed import RISKS_MAPPING

import xgboost as xgb
from sklearn.model_selection import GridSearchCV, KFold


def boosting_reg(model, train, y_train, risk, best_parameters):

    """Cross Validation"""

    kfold = KFold(n_splits=10)
    reg_cv = GridSearchCV(
        model,
        cv=kfold,
        param_grid={
            "regressor__colsample_bytree": [0.1],
            "regressor__min_child_weight": [1.0],
            "regressor__max_depth": [7, 9],
            "regressor__n_estimators": [10],
            "regressor__alpha": [10],
            "regressor__subsample": [0.5],
            "regressor__objective": ["reg:squarederror"],
        },
    )
    # also try "objective": ["multi:softmax", "multi:softprob", "rank:map"], "n_classes": 4'''
    reg_cv.fit(train, y_train)
    best_parameters[risk] = reg_cv.best_params_

    """Training"""

    gbm = xgb.XGBRegressor(**best_parameters[risk])
    gbm.fit(train, y_train)

    """Feature selection
    im=pd.DataFrame({'importance':gbm.feature_importances_,'var':X.columns})
    im=im.sort_values(by='importance',ascending=False)
    fig,ax = plt.subplots(figsize=(8,8))
    plot_importance(xgb,max_num_features=15,ax=ax,importance_type='gain')
    plt.show()"""

    # accuracy_scores[risk]=[gbm.score(train,y_train),0]
    sorted_idx = np.argsort(gbm.feature_importances_)[::-1]
    best_features = list()
    for index in sorted_idx:
        if gbm.feature_importances_[index] > 0:
            best_features.append(train.columns[index])
    return gbm, best_features[:15], best_parameters  # , accuracy_scores


def test_xgboost():
    handler = ModelHandler()
    dataset = handler.dataset
    labeled = dataset[handler.train_mask]
    best_parameters = dict()
    accuracy_scores = dict()
    model = Pipeline(
        [
            ("generation_and_selection", FeatureSelectionAndGeneration(feats_num=10)),
            ("regressor", xgb.XGBRegressor()),
        ],
        memory=".pipeline_cache.tmp",
    )
    for label in ["risk3"]:
        print(f"\n\n**Risk: {RISKS_MAPPING[label]}**\n\n")
        train_mask = ~pd.isnull(dataset[label])
        labeled = dataset.loc[train_mask, :]
        gbm, features, best_parameters = boosting_reg(
            model, labeled[handler.feat_names], labeled[label], label, best_parameters
        )


def test_online_prediction():
    handler = ModelHandler()
    ret = handler.test(0, 30)
    assert len(ret) == 3


def test_online_prediction_for_existing():
    # Tokyo
    handler = ModelHandler()
    ret = handler.test(35.6897, 139.6922)
    assert len(ret) == 2


def test_train():
    handler = ModelHandler()
    handler.train()
