import logging

import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class ModelTransformerMixin(TransformerMixin):
    """This mixin adds a transform method to a model which calls `predict()`, effectively
    making it possible to use the model as an intermediate step in a pipeline."""

    def transform(self, X):
        return self.predict(X)


class SampleWeightMixin:
    """Adds functionality to a model which reads sample weights from the input data,
    drops it from the data and trains the model, passing the weights into its `fit()`
    method.
    """

    def _drop(self, X):
        if not isinstance(X, pd.DataFrame):
            logger.warning(
                "No column names present since X is not a dataframe; the sample_weights column, if present, has not "
                "been dropped."
            )
            return X
        if "sample_weights" in X.columns:
            X = X.drop(columns="sample_weights")
        return X

    def fit(self, X, y, sample_weight=None):
        if "sample_weights" in X:
            return super().fit(self._drop(X), y, sample_weight=X.sample_weights)
        else:
            return super().fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        return super().predict(self._drop(X))

    def predict_proba(self, X):
        return super().predict_proba(self._drop(X))

    def decision_function(self, X):
        return super().decision_function(self._drop(X))


class RandomForestTransformer(
    ModelTransformerMixin, SampleWeightMixin, RandomForestClassifier
):
    pass


class EBMTransformer(
    ModelTransformerMixin, SampleWeightMixin, ExplainableBoostingClassifier
):
    pass


class XGBoostTransformer(XGBClassifier):
    """Because of how the XGBClassifier is implemented, the mixins don't work. So
    we just copypasta them here."""

    def _drop(self, X):
        if isinstance(X, pd.DataFrame) and "sample_weights" in X.columns:
            X = X.drop(columns="sample_weights")
        return X

    def fit(self, X, y, sample_weight=None):
        if "sample_weights" in X:
            return super().fit(self._drop(X), y, X.sample_weights)
        else:
            return super().fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        return super().predict(self._drop(X))

    def predict_proba(self, X):
        return super().predict_proba(self._drop(X))
