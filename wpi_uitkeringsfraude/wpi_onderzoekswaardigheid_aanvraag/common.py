import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def at_least_one_true(
    df: pd.DataFrame, col_to_group_by: str, col_to_sum: str, new_column_name: str
) -> pd.DataFrame:
    """
    Groups the df by col_to_group_by and checks if the sum of the col_to_sum values is > 0.
    This check is stored in new_column_name.

    It returns a dataframe containing new_column_name.

    Parameters
    ----------
    df:
        Dataframe to group
    col_to_group_by:
        Column used to group by
    col_to_sum:
        Column that is going to be summed after grouping
    new_column_name:
        Name of the column where the result is stored
    """
    feature = (df.groupby(col_to_group_by)[col_to_sum].sum() > 0).astype(bool)
    feature.name = new_column_name
    return pd.DataFrame(feature)


def replace_nan_with_zero(df: pd.DataFrame, columns: List) -> pd.DataFrame:
    """Replace NaNs with zero in specified columns.

    Parameters
    ----------
    df:
        df containing NaNs
    columns:
        columns for which NaNs should be replaced

    Returns
    -------
    df:
        df with filled NaNs
    """
    df[columns] = df[columns].fillna(0)
    return df
