
#%%
import sys
import os
import pandas as pd
sys.path.append('..')
os.chdir("./classification")
from sklearn.model_selection import train_test_split
from classification.model_handler import ModelHandler
from classification.feature_selection import FeatureSelectionAndGeneration

import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import kerastuner as kt
#%%
handler = ModelHandler()
dataset = handler.dataset
train_set = dataset[handler.train_mask]
model = {}
label = handler.lab_names[1]
model[label] = FeatureSelectionAndGeneration(apply_selection=False)

augmented_features = model[label].fit_transform(train_set[handler.feat_names], train_set[label])
# %%
augmented_features = augmented_features.iloc[augmented_features[augmented_features.columns[2:]].dropna().index]
#%%
train_set[label].reset_index(drop=True, inplace=True)
notna_mask = train_set[label].notna()

#%%
augmented_features.reset_index(drop=True, inplace=True)
#%%
X_train, X_test, y_train, y_test = train_test_split(
                augmented_features[notna_mask].fillna(0), train_set[label][notna_mask].values.astype(int), test_size=0.2, random_state=42
            )
# %%

#%%
def model_builder(hp):
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model = keras.Sequential(
        [
            layers.Dense(32, activation="relu",input_shape=(X_train.shape[1],)),
            layers.Dense(units=hp_units, activation="relu"),
            layers.Dense(4, activation='softmax'),
        ]
    )

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                metrics=['accuracy'])
    return model

#%%
tuner = kt.Hyperband(
    model_builder,
    objective='val_accuracy',
    max_epochs=500,
    hyperband_iterations=4)

#%%
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

#%%
tuner.search(X_train, y_train, epochs=500, validation_split=0.2, callbacks=[stop_early])
#%%
#history = model.fit()
#%%
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

model = tuner.hypermodel.build(best_hps)
history = model.fit(X_train, y_train, epochs=500, validation_split=0.2)
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1

print('Best epoch: %d' % (best_epoch,))

hypermodel = tuner.hypermodel.build(best_hps)
# Retrain the model
hypermodel.fit(X_train, y_train, epochs=best_epoch, validation_split=0.2, verbose=0)
#%%
eval_result = hypermodel.evaluate(X_test, y_test)
print("[test loss, test accuracy]:", eval_result)
#print("---------------------------------- Evaluation ---------------------------------------")
#test_loss, test_acc = model.evaluate(X_test, y_test)
