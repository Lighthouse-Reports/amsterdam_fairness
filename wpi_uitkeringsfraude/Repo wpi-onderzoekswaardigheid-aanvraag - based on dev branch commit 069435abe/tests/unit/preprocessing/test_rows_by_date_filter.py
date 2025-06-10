import datetime as dt

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from wpi_onderzoekswaardigheid_aanvraag.preprocessing.rows_by_date_filter import (
    SocratesRelevantDateFilter,
    SocratesRelevantPeriodFilter,
)


@pytest.fixture()
def socrates_ref_date_filter():
    yield SocratesRelevantDateFilter(
        ref_date_col="dtaanvraag",
        begindatum_col="dtbegin",
        einddatum_col="dteinde",
        opvoer_col="dtopvoer",
        afvoer_col="dtafvoer",
    )


@pytest.fixture()
def socrates_ref_period_filter_365d():
    yield SocratesRelevantPeriodFilter(
        period=dt.timedelta(days=365),
        ref_date_col="dtaanvraag",
        begindatum_col="dtbegin",
        einddatum_col="dteinde",
        opvoer_col="dtopvoer",
        afvoer_col="dtafvoer",
    )


class TestSocratesRefDateFilter:
    def test_transform(
        self,
        socrates_ref_date_filter,
        mocker,
    ):
        mock_input = pd.DataFrame(
            np.array(
                [
                    [True, False, True, True],
                    [False, True, False, False],
                    [True, True, True, False],
                    [False, False, True, False],
                    [True, False, False, False],
                ]
            ),
            columns=["known", "afgevoerd", "applicable", "expected_to_be_included"],
        )

        mocker.patch(
            "wpi_onderzoekswaardigheid_aanvraag.preprocessing.rows_by_date_filter.SocratesRelevantDateFilter._known_at_reference_date",
            return_value=mock_input["known"],
        )
        mocker.patch(
            "wpi_onderzoekswaardigheid_aanvraag.preprocessing.rows_by_date_filter.SocratesRelevantDateFilter._afgevoerd_at_reference_date",
            return_value=mock_input["afgevoerd"],
        )
        mocker.patch(
            "wpi_onderzoekswaardigheid_aanvraag.preprocessing.rows_by_date_filter.SocratesRelevantDateFilter._applicable_at_reference_time",
            return_value=mock_input["applicable"],
        )

        output = socrates_ref_date_filter.transform(mock_input)
        expected = mock_input.loc[mock_input["expected_to_be_included"]]
        assert_frame_equal(output, expected)

    def test_applicable_at_reference_time(self, socrates_ref_date_filter):
        input_df = pd.DataFrame(
            np.array(
                [
                    ["2015-01-01", "2015-01-05", "2014-01-01"],
                    ["2015-01-01", "2015-01-05", "2016-01-01"],
                    ["2015-01-05", "2015-01-01", "2015-01-03"],
                    ["2015-01-01", "2015-01-05", "2015-01-03"],
                    ["2015-01-01", "2015-01-05", "2015-01-05"],
                    ["2015-01-01", "2015-01-05", "2015-01-01"],
                    ["2015-01-01", np.nan, "2015-01-03"],
                ]
            ),
            columns=["dtbegin", "dteinde", "dtaanvraag"],
        ).astype("datetime64")

        output = socrates_ref_date_filter._applicable_at_reference_time(input_df)
        expected = pd.Series(
            [
                False,
                False,
                False,
                True,
                True,
                True,
                True,
            ]
        )
        assert (output == expected).all()

    def test_known_at_reference_date(self, socrates_ref_date_filter):
        input_df = pd.DataFrame(
            np.array(
                [
                    ["2015-01-05", "2014-01-01"],
                    ["2015-01-05", "2016-01-01"],
                    ["2015-01-05", "2015-01-05"],
                ]
            ),
            columns=["dtopvoer", "dtaanvraag"],
        ).astype("datetime64")

        output = socrates_ref_date_filter._known_at_reference_date(input_df)
        expected = pd.Series(
            [
                False,
                True,
                True,
            ]
        )
        assert (output == expected).all()

    def test_afgevoerd_at_reference_date(self, socrates_ref_date_filter):
        input_df = pd.DataFrame(
            np.array(
                [
                    ["2015-01-01", "2015-01-05"],
                    ["2015-01-05", "2015-01-01"],
                    [np.nan, "2015-01-01"],
                ]
            ),
            columns=["dtafvoer", "dtaanvraag"],
        ).astype("datetime64")

        output = socrates_ref_date_filter._afgevoerd_at_reference_date(input_df)
        expected = pd.Series(
            [
                True,
                False,
                False,
            ]
        )
        assert (output == expected).all()


class TestSocratesRefPeriodFilter:
    def test_applicable_at_reference_time(self, socrates_ref_period_filter_365d):
        input_df = pd.DataFrame(
            np.array(
                [
                    ["2015-01-01", "2015-01-05", "2014-01-01"],  # Both after ref period
                    [
                        "2015-01-01",
                        "2015-01-05",
                        "2016-01-04",
                    ],  # Only dteinde in ref period
                    ["2015-01-01", "2015-01-05", "2015-01-10"],  # Both in ref period
                    [
                        "2015-01-01",
                        "2015-01-05",
                        "2015-01-03",
                    ],  # Only dtbegin in ref period
                    [
                        "2015-01-01",
                        "2015-01-05",
                        "2016-01-10",
                    ],  # Both before ref period
                    ["2015-01-01", np.nan, "2015-01-03"],  # No dteinde
                ]
            ),
            columns=["dtbegin", "dteinde", "dtaanvraag"],
        ).astype("datetime64")

        output = socrates_ref_period_filter_365d._applicable_at_reference_time(input_df)
        expected = pd.Series(
            [
                False,
                True,
                True,
                True,
                False,
                True,
            ]
        )
        assert (output == expected).all()
