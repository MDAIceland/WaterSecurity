from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


# Generation of the Feature Selection Class for the Pipeline
class FeatureSelection(BaseEstimator, TransformerMixin):

    # Initiation of the variables for the feature selection, using sklearn SelectKBest Algorithm
    def __init__(self):
        self.fitted_selector = None
        self.min_num_feats = 10
        self.scores_ = None

    # Process of feature selection is done in this part: Select the K-best features
    # using the f_regression function since labels also have a meaning Risk:0 (means low) and 2(means high)
    def fit(self, x, y):
        # risks.remove(label)
        # print(risks)
        assert all(~np.isnan(y))

        # l = x.columns[x.isna().any()].tolist()
        # print(l)

        var_num = x.shape[0]
        var_num = max(int((var_num * 15) / 100), self.min_num_feats)
        print("Picked variable number:", var_num)

        # Applying select K-best
        bestFeatures = SelectKBest(score_func=f_regression, k=var_num)
        self.fitted_selector = bestFeatures.fit(x, y)
        self.scores_ = self.fitted_selector.scores_
        self.feats_indices = bestFeatures.get_support()

    # Return the vector of best features
    def transform(self, X):
        return self.fitted_selector.transform(X)


# Generation of the transformer class for the return of selected features in the pipeline
class DummyTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feats = None

    def fit(self, X, y):
        self.feats = X.columns
        return self

    def transform(self, X):
        return X

    def get_feature_names(self):
        return self.feats


import re

# Class for generation of cross polynomial features
class ColumnSubstringPolynomial(BaseEstimator, TransformerMixin):
    def __init__(self, element):
        self.element = element
        self.poly = None
        self.pop_feats = []

    # Returns every column name that has a specific string inside
    @staticmethod
    def getArrayOfFeatures(data, name):
        arr = data.columns.values
        return [s for s in arr if (name in s)]

    # Obtaining the list of columns and fitting them to a feature generation model
    def fit(self, X, y=None):
        self.poly = PolynomialFeatures(interaction_only=False, include_bias=False)
        self.pop_feats = self.getArrayOfFeatures(X, self.element)
        self.poly.fit(X[self.pop_feats].values)
        self.mask = np.sum(self.poly.powers_, axis=1) > 1

        # print(crossed_df.shape)
        return self

    # Acquire the selected features and generate a dataframe with only selected feature columns
    def transform(self, data):
        crossed_feats = self.poly.transform(data[self.pop_feats])[:, self.mask]

        # Convert to Pandas DataFrame and merge to original dataset
        crossed_df = pd.DataFrame(crossed_feats)
        return crossed_df

    # Obtain the original feature names after the automatic naming of the cross feature algorithm
    def get_feature_names(self):
        feats = []
        replacement_dict = {cnt: x for cnt, x in enumerate(self.pop_feats)}
        comb_pattern = re.compile(r"(x[0-9]+) (x[0-9]+)")
        single_pattern = re.compile(r"(x[0-9]+)")
        pattern = re.compile(r"(x[0-9]+)")
        for feat in np.array(self.poly.get_feature_names())[self.mask]:
            if re.match(comb_pattern, feat):
                out_feat = re.sub(
                    comb_pattern,
                    r"Feat[\1] * Feat[\2]",
                    feat,
                )
            else:
                out_feat = re.sub(
                    single_pattern,
                    r"Feat[\1]",
                    feat,
                )
            out_feat = re.sub(
                pattern,
                lambda m: replacement_dict[int(m.group()[1:])],
                out_feat,
            )
            feats.append(out_feat)
        return feats


# Generate a PCA class for using the pipeline
class PCAWrapper(BaseEstimator, TransformerMixin):

    # Initializing necessary variables for performing the PCA
    def __init__(self):
        self.num_components = 10
        self.pca = PCA(n_components=self.num_components)
        self.component_cols = ["PC" + str(i + 1) for i in range(self.num_components)]

    # Fit the data to Principle Component Generator, return insight regarding the
    # first 10 principle components of the dataset
    def fit(self, X, y):
        self.pca.fit(X)
        percentage_list = [
            element * 100 for element in self.pca.explained_variance_ratio_
        ]

        print(
            "Explained variation percentage per principal component: {}".format(
                percentage_list
            )
        )
        total_explained_percentage = sum(self.pca.explained_variance_ratio_) * 100
        print(
            "Total percentage of the explained data by",
            self.pca.n_components,
            "components is: %.2f" % total_explained_percentage,
        )
        print(
            "Percentage of the information that is lost for using",
            self.pca.n_components,
            "components is: %.2f" % (100 - total_explained_percentage),
        )
        return self

    # Return the dataframe of generated first 10 Principle Components
    def transform(self, X):
        pca_ret = self.pca.transform(X)
        return pd.DataFrame(data=pca_ret, columns=self.component_cols)

    # Get the name of the Principle Components
    def get_feature_names(self):
        return self.component_cols


# Robust Scaler Class for the generation of pipeline
class RobustScalerWrapper:

    # Initiation of the Robust Scaler Model
    def __init__(self):
        self.robust_scaler = RobustScaler()
        self.columns = None

    # Fit the robust scaler with the given data and returns the scaling version
    def fit(self, X, y):
        self.columns = X.columns
        self.robust_scaler.fit(X, y)
        return self

    # Scaled version of the dataframe is returned for further processing in the pipeline
    def transform(self, X):
        return pd.DataFrame(self.robust_scaler.transform(X), columns=self.columns)


# Main class for the generation of pipeline
class FeatureSelectionAndGeneration(BaseEstimator, TransformerMixin):
    """
        Throughout this process, generation of the new features is done by using
    PCA and Polynomial Cross Features algorithm. Once feature generation is done,
    script uses all of the generated and original features as an input to perform
    a feature selection based on the SelectKBest algorithm of the Sklearn. F_regression
    score is used since numbers in the risk factors are representing a certain value.
    In the end, only selected feature columns, latitude and longitude columns are returned
    back for further prediction of the NaN values for a specific risk factor."""

    # Determine the columns that needs to be substracted before the feature generation
    def __init__(self, apply_selection=True):
        self.id_columns = [
            "latitude",
            "longitude",
        ]

        # Defining the pipeline order given different classes created for the pipeline process
        self.pipeline = Pipeline(
            [
                ("scale", RobustScalerWrapper()),
                (
                    "generation",
                    FeatureUnion(
                        [
                            ("scaled", DummyTransformer()),
                            ("pca", PCAWrapper()),
                            ("pop_poly", ColumnSubstringPolynomial("population")),
                            ("perc_poly", ColumnSubstringPolynomial("%")),
                        ]
                    ),
                ),
                ("selection", FeatureSelection()),
            ]
        )
        self.feat_names = None
        self.apply_selection = apply_selection

        # If feature selection is not applied, you can remove the steps from the pipeline.
        # Flexible solution to remove the unwanted steps
        if not apply_selection:
            self.pipeline.steps.pop(2)

    # Splitting the initial dataset columns into two, for further processing
    def split(self, data):
        return (
            data[self.id_columns],
            data[[col for col in data.columns.values if col not in self.id_columns]],
        )

    # General order of the feature selection and generation process is defined here
    def fit(self, x_data, y_data):
        """
        Fits to nxm features x_data and n predictions y_data
        """
        _, x_data = self.split(x_data)
        self.pipeline.fit(x_data, y_data)
        columns = self.pipeline.named_steps["generation"].get_feature_names()
        dfcolumns = pd.DataFrame(columns)

        # If feature selection process is wanted
        if self.apply_selection:
            dfscores = pd.DataFrame(self.pipeline.named_steps["selection"].scores_)
            feats_indices = self.pipeline.named_steps["selection"].feats_indices
            # print(dfscores)

            # Concat two dataframes for better visualization
            featureScores = pd.concat([dfcolumns, dfscores], axis=1)
            featureScores.columns = ["Specs", "Score"]

            # Print defined amount of features according to the assigned scores with descending order
            print(
                "Features select \n",
                featureScores.iloc[feats_indices]
                .sort_values("Score", ascending=False)
                .to_markdown(),
            )
            self.feat_names = featureScores.iloc[feats_indices].Specs.tolist()
        else:
            self.feat_names = columns
        return self

    # Only return the designated features in addition to the removed features
    # at the beginning of the pipeline process
    def transform(self, x_data):
        """
        Transforms x_data from nxm to kxm
        """
        labs, x_data = self.split(x_data)
        new_x_data = pd.DataFrame(
            self.pipeline.transform(x_data), columns=self.feat_names
        )

        return pd.concat([labs, new_x_data], axis=1)
