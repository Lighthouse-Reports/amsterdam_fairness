import functools

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Adapted from: https://github.com/MaxHalford/xam/blob/master/xam/feature_extraction/encoding/bayesian_target.py


class BayesianTargetEncoder(BaseEstimator, TransformerMixin):

    """
    Reference: https://www.wikiwand.com/en/Bayes_estimator#/Practical_example_of_Bayes_estimators
    Args:
        columns (list of strs): Columns to encode.
        prior_weight (int): Value used to weight the prior.
        minb_obs (int): Minimum nr of observations needed to deviate from prior.
        suffix (str): Suffix used for naming the newly created variables.
    """

    def __init__(self, columns=None, prior_weight=100, min_obs=30, suffix="_mean_enc"):
        self.columns = columns
        self.prior_weight = prior_weight
        self.min_obs = min_obs
        self.suffix = suffix
        self.prior_ = None
        self.posteriors_ = None

        self._is_fit = False

    def fit(self, X, y=None, **fit_params):

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X has to be a pandas.DataFrame")

        if not isinstance(y, pd.Series):
            # return self
            raise ValueError("y has to be a pandas.Series")

        X = X.copy()

        # Default to using all the categorical columns
        columns = (
            X.select_dtypes(["object", "category"]).columns
            if self.columns is None
            else self.columns
        )

        names = []
        for cols in columns:
            if isinstance(cols, list):
                name = "_".join(cols)
                names.append("_".join(cols))
                X[name] = functools.reduce(
                    lambda a, b: a.astype(str) + "_" + b.astype(str),
                    [X[col] for col in cols],
                )
            else:
                names.append(cols)

        # Compute prior and posterior probabilities for each feature
        X = pd.concat((X[names], y.rename("y")), axis="columns")
        self.prior_ = y.mean()
        self.posteriors_ = {}

        for name in names:
            agg = X.groupby(name)["y"].agg(["size", "count", "mean"])
            sizes = agg["size"]
            counts = agg["count"]
            means = agg["mean"]
            pw = self.prior_weight
            posterior_probs = (pw * self.prior_ + counts * means) / (pw + counts)
            posterior_probs = posterior_probs[sizes > self.min_obs]
            self.posteriors_[name] = posterior_probs.to_dict()

        self._is_fit = True
        return self

    def transform(self, X, y=None):
        self._raise_if_not_fitted()

        if not isinstance(X, pd.DataFrame):
            raise ValueError("X has to be a pandas.DataFrame")

        transformed_columns = []

        for cols in self.columns:
            if isinstance(cols, list):
                name = "_".join(cols)
                x = functools.reduce(
                    lambda a, b: a.astype(str) + "_" + b.astype(str),
                    [X[col] for col in cols],
                )
            else:
                name = cols
                x = X[name]

            transf_col_name = name + self.suffix
            transformed_columns.append(transf_col_name)

            X[transf_col_name] = (
                x.map(self.posteriors_[name]).fillna(self.prior_).astype(float)
            )

        return X[transformed_columns]

    def _raise_if_not_fitted(self):
        if not self._is_fit:
            raise ValueError("Transformer has not been fitted")
