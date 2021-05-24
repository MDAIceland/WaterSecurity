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

    def fit(self, x, y):
        # risks.remove(label)
        # print(risks)
        x = self.preprocess(x)
        assert all(~np.isnan(y))

        # l = x.columns[x.isna().any()].tolist()
        # print(l)

        var_num = x.shape[1]
        var_num = max(int((var_num * 15) / 100), self.min_num_feats)
        print("Picked variable number:", var_num)

        # Applying select K-best
        bestFeatures = SelectKBest(score_func=f_regression, k=var_num)
        self.fitted_selector = bestFeatures.fit(x, y)
        dfscores = pd.DataFrame(self.fitted_selector.scores_)
        dfcolumns = pd.DataFrame(x.columns)

        # print(dfscores)

        # Concat two dataframes for better visualization
        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ["Specs", "Score"]
        # print 15% of the total features according to the score features
        print("Features select \n", featureScores.nlargest(var_num, "Score"))

    def transform(self, X):
        return self.fitted_selector.transform(self.preprocess(X))


class DummyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(X):
        return X


class PolynomialPopulation(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.poly = None
        self.pop_feats = []

    @staticmethod
    def getArrayOfFeatures(data, name):
        arr = data.columns.values
        return [s for s in arr if (name in s)]

    def fit(self, X, y=None):
        self.poly = PolynomialFeatures()
        self.pop_feats = self.getArrayOfFeatures(X, "population")
        self.poly.fit(X[self.pop_feats].values)

        # print(crossed_df.shape)
        return self

    def transform(self, data):
        crossed_feats = self.poly.transform(data[self.pop_feats])
        # Convert to Pandas DataFrame and merge to original dataset
        crossed_df = pd.DataFrame(crossed_feats)
        return crossed_df


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


class FeatureSelectionAndGeneration(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.id_columns = [
            "country",
            "city",
            "country_code",
            "c40",
            "latitude",
            "longitude",
        ]
        self.pipeline = Pipeline(
            [
                ("scale", RobustScaler()),
                (
                    "generation",
                    FeatureUnion(
                        [DummyTransformer(), PCAWrapper(), PolynomialPopulation()]
                    ),
                ),
                ("selection", FeatureSelection()),
            ]
        )

    def split(self, data):
        data = data.drop(columns=self.id_columns)
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
        return self

    def transform(self, x_data):
        """
        Transforms x_data from nxm to kxm
        """
        labs, x_data = self.split(x_data)
        new_x_data = self.pipeline.transform(x_data)
        return pd.concat([labs, new_x_data], axis=1)
