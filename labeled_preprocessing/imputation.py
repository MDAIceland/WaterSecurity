from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np


class LabeledDatasetImputer:
    """
    Imputes missing data on y. Assumes also missing data on X. Uses two different types of imputation, as it assumes that y is Categorical
    k_features_per_label: the number of features to keep from X for the imputation, defaults to 0 (no features selection)
    verbose: verbosity of the iterative imputers, defaults to 0
    seed: the random seed, defaults to 42
    labels_est: the estimator object to be used during labels imputation
    feats_est: the estimator object to be used during features estimation
    """

    def __init__(
        self,
        k_features_per_label=0,
        verbose=0,
        seed=42,
        labels_est=None,
        feats_est=None,
    ):
        self.x_imputer = None
        self.y_imputer = None
        self.verbose = verbose
        self.selection_mask = None
        self.seed = seed
        self.k_features_per_label = k_features_per_label
        self.labels_est = labels_est
        self.feats_est = feats_est

    def create_selection_mask(self, X, y):
        if not self.k_features_per_label:
            return np.zeros(X.shape[1]) == 0
        selection_mask = None
        for cnt in range(y.shape[1]):
            labeled = ~np.isnan(y[:, cnt])
            _y = y[labeled, cnt]
            _x = X[labeled, :]

            selector = SelectKBest(f_classif, k=self.k_features_per_label).fit(
                np.nan_to_num(_x), _y
            )
            if selection_mask is None:
                selection_mask = selector.get_support()
            else:
                selection_mask = (selection_mask + selector.get_support()) > 0
        return selection_mask

    def fit_transform(self, X, y, ret_imputed_x=False):
        """
        X: nxp matrix
        y: nxv matrix
        Both matrices are allowed to have missing values
        if `ret_imputed_x`, return (imputed_x,imputed_y), otherwise return imputed_y
        """
        print("Applying feature selection..")
        self.selection_mask = self.create_selection_mask(X, y)
        if self.feats_est is None:
            self.feats_est = KNeighborsRegressor(n_neighbors=5)

        print(f"Creating imputed X using {self.feats_est.__class__.__name__}..")
        self.x_imputer = IterativeImputer(
            estimator=self.feats_est,
            initial_strategy="most_frequent",
            verbose=self.verbose,
            n_nearest_features=200,
            random_state=self.seed,
            skip_complete=True,
        )
        imputed_x = self.x_imputer.fit_transform(X[:, self.selection_mask])
        if self.labels_est is None:
            self.labels_est = make_pipeline(
                SelectKBest(
                    f_classif, k=min(int(0.1 * X.shape[0]), imputed_x.shape[1])
                ),
                RandomForestClassifier(n_estimators=50, random_state=self.seed),
            )
        print(f"Creating imputed Y using {self.labels_est.__class__.__name__}..")
        self.y_imputer = IterativeImputer(
            estimator=self.labels_est,
            initial_strategy="most_frequent",
            max_iter=10,
            random_state=self.seed,
            skip_complete=True,
            verbose=self.verbose,
        )
        imputed_y = self.y_imputer.fit_transform(np.hstack([y, imputed_x]))[
            :, : y.shape[1]
        ]
        if ret_imputed_x:
            return imputed_x, imputed_y
        return imputed_y

    def transform(self, X, y, ret_imputed_x=False):
        """
        X: nxp matrix
        y: nxv matrix
        Both matrices are allowed to have missing values
        if `ret_imputed_x`, return (imputed_x,imputed_y), otherwise return imputed_y
        """
        imputed_x = self.x_imputer.transform(X[:, self.selection_mask])
        ret = self.y_imputer.transform(np.hstack([y, imputed_x]))[:, : y.shape[1]]
        if ret_imputed_x:
            return imputed_x, ret
        return ret
