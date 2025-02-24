import logging
from collections import Counter
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper, gen_features

from wpi_onderzoekswaardigheid_aanvraag.model.auxiliary.embeddings_encoder import (
    EmbeddingsEncoder,
)
from wpi_onderzoekswaardigheid_aanvraag.model.auxiliary.sample_weights_transformer import (
    EBMTransformer,
    RandomForestTransformer,
    XGBoostTransformer,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.custom_transformers import (
    BoolToIntTransformer,
    FloatTransformer,
    SimpleImputerWithRenaming,
)

logger = logging.getLogger(__name__)


def split_data_train_dev_test(df):
    """
    Creating sets for model building and testing. Steps:
    1. Training set (70%) - for building the model
    2. Development set a.k.a. hold-out set (15%) - for optimizing model parameters
    3. Test set (15%) - For testing the performance of the tuned model
    """

    # Split data into features (X) and labels (y).
    X = df.drop("onderzoekswaardig", axis=1)
    y = df.onderzoekswaardig
    logger.debug("Original dataset shape %s" % Counter(df.onderzoekswaardig))

    # Split the dataset.
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, train_size=0.7, stratify=y
    )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_rest, y_rest, train_size=0.5, stratify=y_rest
    )

    logger.debug("Training set shape %s" % Counter(y_train))
    logger.debug("Development set shape %s" % Counter(y_dev))
    logger.debug("Testing set shape %s" % Counter(y_test))

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def split_data_train_test(
    df: pd.DataFrame, shuffle: bool, label: str = "onderzoekswaardig", random_state=None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Creating sets for model building and testing. Steps:

    1. Training set (85%) - for use with cross-validations
    2. Test set (15%) - For possibly testing the performance of any tuned models

    Parameters
    ----------
    df:
        Dataframe that will be split
    shuffle:
        whether to shuffle the data
    label:
        column that contains the label
    random_state:
        when shuffling use this random seed

    Returns
    -------
    X_train:
        features for training
    X_test:
        features for testing
    y_train:
        label for training
    y_test:
        label for testing
    """

    # Split data into features (X) and labels (y).
    X = df.drop(label, axis=1)
    y = df.onderzoekswaardig
    logger.debug("Original dataset shape %s" % Counter(df[label]))

    if shuffle:
        stratify = y
    else:
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=0.85,
        stratify=stratify,
        shuffle=shuffle,
        random_state=random_state,
    )

    logger.debug("Training set shape %s" % Counter(y_train))
    logger.debug("Testing set shape %s" % Counter(y_test))

    return X_train, X_test, y_train, y_test


def make_dataframe_mapper(cat_cols, num_cols):
    """Assemble feature transformations."""
    cat_features = gen_features(
        columns=cat_cols, classes=[{"class": EmbeddingsEncoder}]
    )

    cols_to_mean_impute = {
        "sum_inkomen_netto",
        "sum_inkomen_bruto",
    }.intersection(set(num_cols))

    mean_imputed_features = gen_features(
        columns=list(cols_to_mean_impute),
        classes=[
            {
                "class": SimpleImputerWithRenaming,
                "strategy": "mean",
                "add_indicator": True,
                "verbose": 2,
            }
        ],
    )

    other_num_cols = gen_features(
        columns=list(set(num_cols) - cols_to_mean_impute),
        classes=[{"class": BoolToIntTransformer}, {"class": FloatTransformer}],
    )

    mean_imputed_features = [
        ([col], kls, args) for col, kls, args in mean_imputed_features
    ]
    other_num_cols = [([col], kls, args) for col, kls, args in other_num_cols]

    mapper = DataFrameMapper(
        other_num_cols + mean_imputed_features + cat_features,
        input_df=True,
        df_out=True,
        default=False,  # Drop unspecified columns
    )
    return mapper


def make_model(algorithm_type, **params):
    if algorithm_type == "RF":
        model = RandomForestTransformer(
            random_state=42,
            n_jobs=-1,
            **params,
        )
    elif algorithm_type == "XGB":
        model = XGBoostTransformer(
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric="auc",
            random_state=42,
            **params,
        )
    elif algorithm_type == "EBM":
        model = EBMTransformer(
            random_state=42,
            n_jobs=-1,
            **params,
        )
    else:
        raise ValueError(
            f"Argument algorithm_type should be one of ['RF', 'XGB', 'EBM'], got input: '{algorithm_type}'"
        )

    logger.info(f'Using model "{algorithm_type}".')
    return model


def filter_application_handling(
    df, include_handling_types: List[str] = None
) -> pd.DataFrame:
    """Filter applications based on who handled them:
    - screening by inkomensconsulent
    - screening by handhaving
    - onderzoek by handhaving

    Parameters
    ----------
    df:
        dataset
    include_handling_types:
        list of handling types to include; each string should match a boolean
        column in the dataframe indicating if the application was of that type

    Returns
    -------
    result:
        filtered df
    """
    if include_handling_types:
        return df[df[include_handling_types].sum(axis=1) > 0]
    return df
