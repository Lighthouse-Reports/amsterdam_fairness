import copy
import importlib.resources
import logging

import great_expectations as ge
import pandas as pd
from great_expectations.core.expectation_suite import (
    ExpectationSuite,
    expectationSuiteSchema,
)

logger = logging.getLogger(__name__)


def check_expectations(
    df: pd.DataFrame,
    expectations_file: str,
    include_train_expectations: bool = False,
    raise_upon_fail: bool = False,
) -> dict:
    """Validate that df conforms to the given expectations.

    Parameters
    ----------
    df: Dataframe to validate
    expectations_file: Name of JSON file with expectations (located in `/resources`)
    include_train_expectations: Whether or not to include expectations that are
        only applicable for training data
    raise_upon_fail: Whether or not to raise an error when an expectation fails
        (default: False)

    Returns
    -------
    :
        JSON-formatted dictionary of validation results
    """
    dataset = ge.from_pandas(df)
    es = load_expectations(expectations_file)

    if include_train_expectations:
        es = filter_expectations_no_train(es)
    result = dataset.validate(es)

    if result["success"]:
        logger.info("Expectations check succeeded")
    else:
        result = filter_failed_validations(result)
        if raise_upon_fail:
            raise Exception(f"Some expectations failed:\n{result['results']}")
        else:
            logger.warning(f"Failed expectations:\n{result['results']}")

    return result


def load_expectations(file_name: str) -> ExpectationSuite:
    expectations = importlib.resources.read_text(
        "wpi_onderzoekswaardigheid_aanvraag.resources", file_name
    )
    es = expectationSuiteSchema.loads(expectations)
    logger.info(f"Loaded expectations from file: {file_name}")
    return es


def filter_expectations_no_train(
    expectation_suite: ExpectationSuite,
) -> ExpectationSuite:
    result = copy.deepcopy(expectation_suite)
    result["expectations"] = [
        exp
        for exp in expectation_suite["expectations"]
        if not exp["meta"]["train_only"]
    ]
    return result


def filter_failed_validations(val_result: dict) -> dict:
    """Filter validation results on failed expectations."""
    result = copy.deepcopy(val_result)
    result["expectations"] = [
        res for res in val_result["results"] if not res["success"]
    ]
    return result
