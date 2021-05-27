
import sys
import os
import pandas as pd
sys.path.append('..')
os.chdir("./classification")

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from classification.model_handler import ModelHandler
from classification.feature_selection import FeatureSelectionAndGeneration
import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)


handler = ModelHandler()
dataset = handler.dataset
train_set = dataset[handler.train_mask]
model = {}
label = []
for label in handler.lab_names:
    model[label] = FeatureSelectionAndGeneration(apply_selection=False)

    augmented_features = model[label].fit_transform(train_set[handler.feat_names], train_set[label])

    augmented_features = augmented_features.iloc[augmented_features[augmented_features.columns[2:]].dropna().index]

    train_set[label].reset_index(drop=True, inplace=True)
    notna_mask = train_set[label].notna()


    augmented_features.reset_index(drop=True, inplace=True)
    augmented_features.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in augmented_features.columns.values]


    X_train, X_test, y_train, y_test = train_test_split(
                    augmented_features[notna_mask].fillna(0), train_set[label][notna_mask].values.astype(int), test_size=0.3, random_state=42
                )


    mod = XGBClassifier()
    mod.fit(X_train, y_train)
    # make predictions for test data
    y_pred = mod.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    
    print(label,"Accuracy: %.2f%%" % (accuracy * 100.0))
