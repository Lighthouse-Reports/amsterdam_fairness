import numpy as np
import pandas as pd


def add_month_feature(df: pd.DataFrame, source_col_name: str, target_col_name: str):
    """Extract month from a date column (1-12) circular encoded,
    creating sin and cosine columns."""
    s, c = circular_encode(df[source_col_name].dt.month, 12)
    df[f"{target_col_name}_sin"] = s
    df[f"{target_col_name}_cos"] = c
    return df


def add_year_feature(df: pd.DataFrame, source_col_name: str, target_col_name: str):
    """Extract day of the year from a date column (from 1 to 365 or 366, depending on leap years)
    and circular encodes it, creating sin and cosine columns."""
    yday = df[source_col_name].apply(extract_yday_from_date)
    s, c = circular_encode(yday, 366)
    df[f"{target_col_name}_sin"] = s
    df[f"{target_col_name}_cos"] = c
    return df


def circular_encode(data_col: "pd.Series", max_val: int):
    """
    Circular encodes calculating sin and cos the values from 1 to max_val.

    Parameters
    ----------
    data_col
        Series containing the data to encode

    max_val
        Maximum possible value of the data series

    Returns
    -------
        sin of the encoded data
        cos of the encoded data

    """
    sin = np.sin(2 * np.pi * data_col / max_val)
    cos = np.cos(2 * np.pi * data_col / max_val)
    return sin, cos


def extract_yday_from_date(x):
    try:
        return x.timetuple().tm_yday
    except ValueError:
        return None
