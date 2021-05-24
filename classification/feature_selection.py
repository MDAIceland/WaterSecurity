from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class FeatureSelection(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.fitted_selector = None
        self.min_num_feats = 10
        self.scores_ = None

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

    def transform(self, X):
        return self.fitted_selector.transform(X)


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


class ColumnSubstringPolynomial(BaseEstimator, TransformerMixin):
    def __init__(self, element):
        self.element = element
        self.poly = None
        self.pop_feats = []

    @staticmethod
    def getArrayOfFeatures(data, name):
        arr = data.columns.values
        return [s for s in arr if (name in s)]

    def fit(self, X, y=None):
        self.poly = PolynomialFeatures()
        self.pop_feats = self.getArrayOfFeatures(X, self.element)
        self.poly.fit(X[self.pop_feats].values)

        # print(crossed_df.shape)
        return self

    def transform(self, data):
        crossed_feats = self.poly.transform(data[self.pop_feats])

        # Convert to Pandas DataFrame and merge to original dataset
        crossed_df = pd.DataFrame(crossed_feats)
        return crossed_df

    def get_feature_names(self):
        feats = []
        for out_feat in self.poly.get_feature_names():
            for cnt, pop_feat in enumerate(self.pop_feats):
                out_feat = re.sub(r"x" + str(cnt) + r"\b", pop_feat, out_feat)
            feats.append(out_feat)
        return feats


class PCAWrapper(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_components = 10
        self.pca = PCA(n_components=self.num_components)
        self.component_cols = ["PC" + str(i + 1) for i in range(self.num_components)]

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

    def transform(self, X):
        pca_ret = self.pca.transform(X)
        return pd.DataFrame(data=pca_ret, columns=self.component_cols)

    def get_feature_names(self):
        return self.component_cols


class RobustScalerWrapper:
    def __init__(self):
        self.robust_scaler = RobustScaler()
        self.columns = None

    def fit(self, X, y):
        self.columns = X.columns
        self.robust_scaler.fit(X, y)
        return self

    def transform(self, X):
        return pd.DataFrame(self.robust_scaler.transform(X), columns=self.columns)


class FeatureSelectionAndGeneration(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.id_columns = [
            "latitude",
            "longitude",
        ]
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

    def split(self, data):
        return (
            data[self.id_columns],
            data[[col for col in data.columns.values if col not in self.id_columns]],
        )

    def fit(self, x_data, y_data):
        """
        Fits to nxm features x_data and n predictions y_data
        """
        _, x_data = self.split(x_data)
        self.pipeline.fit(x_data, y_data)
        dfscores = pd.DataFrame(self.pipeline.named_steps["selection"].scores_)
        dfcolumns = pd.DataFrame(
            self.pipeline.named_steps["generation"].get_feature_names()
        )
        feats_indices = self.pipeline.named_steps["selection"].feats_indices
        # print(dfscores)

        # Concat two dataframes for better visualization
        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ["Specs", "Score"]
        # print 15% of the total features according to the score features
        print(
            "Features select \n",
            featureScores.iloc[feats_indices].sort_values("Score", ascending=False),
        )
        self.feat_names = featureScores.iloc[feats_indices].Specs.tolist()
        return self

    def transform(self, x_data):
        """
        Transforms x_data from nxm to kxm
        """
        labs, x_data = self.split(x_data)
        new_x_data = pd.DataFrame(
            self.pipeline.transform(x_data), columns=self.feat_names
        )

        return pd.concat([labs, new_x_data], axis=1)
