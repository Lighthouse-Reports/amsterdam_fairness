import datetime as dt
import logging
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from fraude_preventie.clean import CleanTransformer

logger = logging.getLogger(__name__)


class WPICleanTransformer(CleanTransformer):
    """Class for performing cleaning steps on dataframes within the sklearn pipeline.

    Parameters
    ----------
    fix_no_end_date:
        list of columns for which to replace no end date with NaN

    Notes
    =====
    * the current implementation of this transformer does not need fitting, and the
      calling code hence does not call fit on the transformer. Unless you refactor
      all usages of this transformer, do not change this!

    """

    def __init__(
        self,
        id_column=None,
        drop_duplicates: bool = True,
        fix_date_columns: List[str] = None,
        clean_string_columns: List[str] = None,
        missing_values_mapping: dict = None,
        col_type_mapping: Sequence[Union[str, Tuple[str, Union[str, Any]]]] = None,
        fix_bool_columns: List[str] = None,
        remove_invalidated_data: bool = False,
        fix_no_end_date: List[str] = None,
        do_dtype_optimization: Union[bool, List[str]] = True,
    ):
        super().__init__(
            self,
            drop_duplicates=drop_duplicates,
            fix_date_columns=fix_date_columns,
            clean_string_columns=clean_string_columns,
            missing_values_mapping=missing_values_mapping,
            col_type_mapping=col_type_mapping,
            fix_bool_columns=fix_bool_columns,
            do_dtype_optimization=do_dtype_optimization,
        )
        self.id_column = id_column
        self.remove_invalidated_data = remove_invalidated_data
        self.fix_no_end_date = fix_no_end_date
        self.is_fit = True

    def fit(self, X, y=None):
        pass  # no fitting necessary

    def transform(self, X):
        logger.debug("Transforming...")
        logger.debug(f"Transform input df: {X}")
        if self.remove_invalidated_data:
            X = self._remove_socrates_invalidated_data(X)
        if self.fix_no_end_date:
            X = self._fix_no_end_date(X, self.fix_no_end_date)
        X = super().transform(X)
        logger.debug("Transform done.")
        return X

    def _raise_if_not_fitted(self):
        if not self.is_fit:
            raise ValueError("Transformer has not been fitted")

    @staticmethod
    def _fix_no_end_date(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Rows without an end date have end date 31 Dec 2382 in Socrates (mostly)
        or another far away date (sometimes). Therefore, transform values more than
        10 years away to NaN.

        Parameters
        ----------
        df:
            dataframe to transform
        cols:
            list of columns to apply the transformation to

        Returns
        -------
        df:
            copy of input dataframe with end dates fixed in specified columns.
        """
        if len(cols) > 0:
            logging.debug(
                f"Setting end date to missing for rows without end date on {cols}..."
            )
            df = df.copy()
            for c in cols:
                cutoff_date = dt.datetime.now() + dt.timedelta(days=10 * 365)
                try:
                    relevant_rows = df[c] >= cutoff_date.date()
                except TypeError:
                    relevant_rows = df[c] >= f"{cutoff_date:%Y-%m-%d}"
                df.loc[relevant_rows, c] = np.nan
        return df

    @staticmethod
    def _remove_socrates_invalidated_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean a Socrates table from invalid data (not 'geldig').

        Parameters
        ----------
        df:
            Socrates df to clean

        Returns
        -------
        df:
            cleaned df
        """
        return df[df["geldig"] == 1]
