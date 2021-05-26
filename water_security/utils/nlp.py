import numpy as np
import spacy

try:
    nlp = spacy.load("en_core_web_md")
except:
    spacy.cli.download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

from sklearn.base import BaseEstimator, TransformerMixin


class SimilarityAnalysis(BaseEstimator, TransformerMixin):
    """
    Creates similarity matrix to the provided pandas series. Can be fitted to a specific data.
    The computed non empty spacy vectors will then be used as reference to compare with another
    dataset.
    """

    def __init__(self):
        self.similarity_vectors = None

    def fit(self, description):
        """
        Creates an sxs matrix
        """
        ret = description.apply(lambda x: nlp(".".join(x)))
        self.similarity_vectors = [x for x in ret if x]
        return self

    def transform(self, description):
        """
        Produces a nxs matrix
        """
        ret = description.apply(lambda x: nlp(".".join(x)))
        ret = np.vstack(
            ret.apply(
                lambda x: [
                    (x.similarity(y) if x else np.nan) for y in self.similarity_vectors
                ]
            )
        )
        return ret

    def fit_transform(self, description):
        """
        Produces a nxn matrix
        """
        ret = description.apply(lambda x: nlp(".".join(x)))
        self.similarity_vectors = [x for x in ret if x]
        ret = ret.apply(
            lambda x: [
                (x.similarity(y) if x else np.nan) for y in self.similarity_vectors
            ]
        )
        ret = np.vstack(ret)
        return ret


def create_sim_vector(description):
    return SimilarityAnalysis().fit_transform(description)
