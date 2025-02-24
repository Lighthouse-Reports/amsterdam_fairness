import datetime as dt
import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class SocratesRelevantDateFilter(BaseEstimator, TransformerMixin):
    """Class for filtering Socrates dataframe on rows relevant at a specific
    reference date.

    Parameters
    ----------
    ref_date_col:
        name of the column containing the reference date for each row
    begindatum_col:
        name of the column containing the begindatum
    einddatum_col:
        name of the column containing the einddatum
    opvoer_col:
        name of the column containing the opvoerdatum
    afvoer_col:
        name of the column containing the afvoerdatum

    Notes
    =====
    * Except for the reference date, all other column names have default values.
      These can be overwritten, for example when the dataframe has been joined
      with another, resulting in column names with suffixes. Also, some tables
      use variations on dtbegin/dteinde.
    """

    def __init__(
        self,
        ref_date_col: str,
        begindatum_col: str = None,
        einddatum_col: str = None,
        opvoer_col: str = "dtopvoer",
        afvoer_col: str = None,
    ):
        self.ref_date_col = ref_date_col
        self.begindatum_col = begindatum_col
        self.einddatum_col = einddatum_col
        self.opvoer_col = opvoer_col
        self.afvoer_col = afvoer_col
        self.is_fitted = True

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Filter dataframe on rows that were relevant at a reference date.

        A row is relevant if on the reference date it was:
        (1) applicable (reference date in between start and end date of the row)
        (2) known (entered into the system before reference date)
        (3) not invalidated (not 'afgevoerd' before reference date)

        Parameters
        ----------
        X:
            df to filter

        Returns
        -------
        X:
            df filtered
        """
        result = X[
            self._known_at_reference_date(X) & ~self._afgevoerd_at_reference_date(X)
        ]

        if self.begindatum_col is None or self.einddatum_col is None:
            logger.warning(
                f"{self.__class__.__name__}:Not checking begindatum and einddatum, "
                f"if required please specify both `begindatum_col` and `einddatum_col`"
            )
            return result

        return result.loc[self._applicable_at_reference_time(result)]

    def _raise_if_not_fitted(self):
        if not self.is_fitted:
            raise ValueError("Transformer has not been fitted")

    def _applicable_at_reference_time(self, df):
        return self._begindatum_matching_reference_date(
            df
        ) & self._einddatum_matching_reference_date(df)

    @staticmethod
    def fix_time_delay(value_to_fix: np.timedelta64) -> np.timedelta64:
        """
        There could be some minutes delay inputting the data into the database.
        This results in some data being excluded by the filters because of seconds/minutes
        even if they should be included.
        For this reason we add one minute to our dates to make the filter less strict.
        """
        return value_to_fix + np.timedelta64(1, "m")

    def _begindatum_matching_reference_date(self, df):
        return self.fix_time_delay(df[self.ref_date_col]) >= df[self.begindatum_col]

    def _einddatum_matching_reference_date(self, df):
        return df[self.einddatum_col].isna() | (
            df[self.ref_date_col] <= self.fix_time_delay(df[self.einddatum_col])
        )

    def _known_at_reference_date(self, df):
        return self.fix_time_delay(df[self.ref_date_col]) >= df[self.opvoer_col]

    def _afgevoerd_at_reference_date(self, df):
        if self.afvoer_col:
            return self._afgevoerd(df) & (
                df[self.ref_date_col] >= self.fix_time_delay(df[self.afvoer_col])
            )

        logger.warning(
            f"{self.__class__.__name__}:Not checking afvoer date, "
            f"if required please specify `afvoer_col`"
        )
        return np.repeat(False, len=len(df))

    def _afgevoerd(self, df):
        return ~df[self.afvoer_col].isna()


class SocratesRelevantPeriodFilter(SocratesRelevantDateFilter):
    """Class for filtering Socrates dataframe on rows relevant in a period
    before a specific reference date.

    Parameters
    ----------
    ref_date_col:
        name of the column containing the reference date for each row
    begindatum_col:
        name of the column containing the begindatum
    einddatum_col:
        name of the column containing the einddatum
    opvoer_col:
        name of the column containing the opvoerdatum
    afvoer_col:
        name of the column containing the afvoerdatum
    period:
        how far to look back. Data relevant between `ref_date_col - period` and
        `ref_date_col` will be selected
    """

    def __init__(
        self,
        period: dt.timedelta,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.period = period
        self.is_fitted = True

    def fit(self, X, y=None):
        return self

    def _applicable_at_reference_time(self, df):
        # If at least one of the begindatum or the einddatum lies inside the
        # reference period, then the periods overlap, so the row is applicable.
        ref_period_start = df[self.ref_date_col] - self.period
        return (
            df[self.einddatum_col].isna()
            | (ref_period_start <= self.fix_time_delay(df[self.einddatum_col]))
        ) & (self.fix_time_delay(df[self.ref_date_col]) >= df[self.begindatum_col])


def date_in_ref_period(
    df: pd.DataFrame,
    ref_col: str,
    check_col: str,
    period_days: int,
    nan_value: bool = False,
) -> pd.Series:
    """Check for which rows of the df `check_col` lies in the reference period.
    Returns False if `check_col` is NaT.

    Parameters
    ----------
    df: applications dataframe with afspraken data merged to it
    ref_col: reference column
    check_col: date column to check against reference column
    period_days: how far back from `ref_col` the reference period goes in days
    nan_value: what to return for rows where `check_col` is NaT (defaults to False)

    Returns
    -------
    :
        boolean series
    """
    ref_period_start = df[ref_col] - dt.timedelta(days=period_days)
    ref_period_end = df[ref_col]
    condition = (
        SocratesRelevantDateFilter.fix_time_delay(df[check_col]) >= ref_period_start
    ) & (df[check_col] <= SocratesRelevantDateFilter.fix_time_delay(ref_period_end))
    if nan_value is True:
        condition[df[check_col].isna()] = True
    return condition
