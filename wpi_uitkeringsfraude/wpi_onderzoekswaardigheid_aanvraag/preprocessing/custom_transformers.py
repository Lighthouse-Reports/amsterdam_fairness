import logging

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer

logger = logging.getLogger(__name__)


def bool_to_num(X):
    """Replace all True and False values in X by 1 and 0 respectively."""
    # Call replace twice instead of with a dict to prevent it from failing if
    # there are no Trues or no Falses.
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == "boolean":
            X.loc[:, c] = X[c].replace(True, 1).replace(False, 0)
        else:
            X.loc[:, c] = X[c].replace("True", 1).replace("False", 0)
    return X


class BoolToIntTransformer(FunctionTransformer):
    """Transformer that replaces all boolean values by integers, so that they can
    be processed as numericals.
    """

    def __init__(self):
        super().__init__(bool_to_num)


def to_float(X):
    """Cast to float"""
    return X.astype(float)


class FloatTransformer(FunctionTransformer):
    """Transformer that casts all input to floats."""

    def __init__(self):
        super().__init__(to_float)


class SimpleImputerWithRenaming(SimpleImputer):
    """Transformer that expands sklearn's `SimpleImputer` with meaningful feature
    names if a missingness indicator is added and the input data was a `pd.DataFrame`.
    """

    def __init__(
        self,
        *,
        missing_values=np.nan,
        strategy="mean",
        fill_value=None,
        verbose=0,
        copy=True,
        add_indicator=False,
    ):
        super().__init__(
            missing_values=missing_values,
            strategy=strategy,
            fill_value=fill_value,
            verbose=verbose,
            copy=copy,
            add_indicator=add_indicator,
        )
        self.input_columns = None
        self.output_col_suffixes = None

    def fit(self, X, y=None):
        super().fit(X, y)
        try:
            self.input_columns = X.columns.tolist()
        except AttributeError:
            logger.warning("To rename output columns, please input a dataframe")
        return self

    def transform(self, X):
        X_t = super().transform(X)
        if (self.input_columns is not None) and self.add_indicator:
            self._create_output_column_names(X_t)
        return X_t

    def _create_output_column_names(self, X):
        if X.shape[1] != 2:
            logger.warning(
                "Skipping renaming of output columns; the transformed data did not have the expected number of columns (2)"
            )
            return

        if isinstance(self.input_columns, list) & (len(self.input_columns) != 1):
            logger.warning(
                "Skipping renaming of output columns; this only works with exactly one input column"
            )
            return
        elif isinstance(self.input_columns, list) & (len(self.input_columns) == 1):
            self.output_col_suffixes = ["value", f"was_{self.strategy}_imputed"]
        else:
            raise ValueError(
                f"`input_columns` must be a list, type given: {type(self.input_columns)}"
            )

    def get_feature_names(self):
        return self.output_col_suffixes
