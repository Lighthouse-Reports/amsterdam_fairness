import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from wpi_onderzoekswaardigheid_aanvraag.components.proces_type_info import (
    ProcesTypeInfo,
)


def screening_indicator_test_inputs():
    return [
        {
            "pro_id": np.nan,
            "pro_startdatum": pd.to_datetime(np.nan),
            "sre_id": np.nan,
            "expected_outcome": pd.Series(
                [1, 0, 0],
                index=["is_screening_ic", "is_screening_hh", "is_onderzoek_hh"],
            ),
        },
        {
            "pro_id": 10,
            "pro_startdatum": pd.to_datetime("2020-03-23"),
            "sre_id": np.nan,
            "expected_outcome": pd.Series(
                [0, 1, 0],
                index=["is_screening_ic", "is_screening_hh", "is_onderzoek_hh"],
            ),
        },
        {
            "pro_id": 11,
            "pro_startdatum": pd.to_datetime("2020-07-16"),
            "sre_id": np.nan,
            "expected_outcome": pd.Series(
                [0, 1, 0],
                index=["is_screening_ic", "is_screening_hh", "is_onderzoek_hh"],
            ),
        },
        {
            "pro_id": 12,
            "pro_startdatum": pd.to_datetime("2020-03-22"),
            "sre_id": np.nan,
            "expected_outcome": pd.Series(
                [0, 0, 1],
                index=["is_screening_ic", "is_screening_hh", "is_onderzoek_hh"],
            ),
        },
        {
            "pro_id": 13,
            "pro_startdatum": pd.to_datetime("2020-01-01"),
            "sre_id": 744,
            "expected_outcome": pd.Series(
                [0, 1, 0],
                index=["is_screening_ic", "is_screening_hh", "is_onderzoek_hh"],
            ),
        },
    ]


class TestProcesTypeInfo:
    @pytest.mark.parametrize("input_row", screening_indicator_test_inputs())
    def test_add_indicator_screening_exactly_one_hot(self, input_row):
        input_df = pd.DataFrame.from_records(input_row, index=[0])
        output = ProcesTypeInfo()._add_indicator_screening(input_df)
        assert (
            output.loc[
                0, ["is_screening_ic", "is_screening_hh", "is_onderzoek_hh"]
            ].sum()
            == 1
        )

    def test_add_indicator_vpo_planned(self):
        input_df = pd.DataFrame(
            np.array(
                [
                    [1, pd.to_datetime("2015-01-01"), np.nan],
                    [1, pd.to_datetime("2015-01-02"), pd.to_datetime("2016-01-01")],
                    [2, pd.to_datetime("2015-01-01"), np.nan],
                    [2, pd.to_datetime("2015-01-02"), np.nan],
                    [3, pd.to_datetime("2015-01-01"), pd.to_datetime("2016-01-01")],
                    [4, pd.to_datetime("2015-01-01"), pd.to_datetime("2014-01-01")],
                    [5, np.nan, pd.to_datetime("2014-01-01")],
                ]
            ),
            columns=["pro_id", "pro_einddatum", "pon_hercontroledatum"],
        )

        output = ProcesTypeInfo()._get_indicator_vpo_planned(input_df)

        expected = pd.Series(
            [
                True,
                True,
            ],
            index=[1, 3],
            name="vpo_planned",
        )
        expected.index.name = "pro_id"
        assert_series_equal(output, expected)
