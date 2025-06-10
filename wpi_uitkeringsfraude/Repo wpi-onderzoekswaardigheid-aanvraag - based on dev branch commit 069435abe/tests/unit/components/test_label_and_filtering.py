import numpy as np
import pandas as pd
import pytest

from wpi_onderzoekswaardigheid_aanvraag.components.label_and_filtering import (
    LabelAndFiltering,
)


@pytest.fixture()
def label_test_transformer():
    transformer = LabelAndFiltering()
    # Mock the class attributes for consistent testing.
    transformer.srp_id_onderzoekswaardig = [1, 2]
    transformer.srp_id_uitfilteren = [3, 4]
    transformer.dienstreden_afwijzing_onderzoekswaardig = [5, 6]
    return transformer


def label_test_inputs():
    return [
        {
            # Onderzoek HH with SRP ID in list
            "is_onderzoek_hh": True,
            "is_screening_hh": False,
            "is_screening_ic": False,
            "srp_id": 1,
            "vpo_planned": np.nan,
            "besluit": np.nan,
            "reden_dienstreden": np.nan,
            "expected_outcome": 1,
        },
        {
            # Onderzoek HH with SRP ID not in list
            "is_onderzoek_hh": True,
            "is_screening_hh": False,
            "is_screening_ic": False,
            "srp_id": 0,
            "vpo_planned": np.nan,
            "besluit": np.nan,
            "reden_dienstreden": np.nan,
            "expected_outcome": 0,
        },
        {
            # Screening HH with VPO planned
            "is_onderzoek_hh": False,
            "is_screening_hh": True,
            "is_screening_ic": False,
            "srp_id": np.nan,
            "vpo_planned": 1,
            "besluit": np.nan,
            "reden_dienstreden": np.nan,
            "expected_outcome": 1,
        },
        {
            # Screening HH without VPO planned
            "is_onderzoek_hh": False,
            "is_screening_hh": True,
            "is_screening_ic": False,
            "srp_id": np.nan,
            "vpo_planned": 0,
            "besluit": np.nan,
            "reden_dienstreden": np.nan,
            "expected_outcome": 0,
        },
        {
            # Screening IC not rejected
            "is_onderzoek_hh": False,
            "is_screening_hh": False,
            "is_screening_ic": True,
            "srp_id": np.nan,
            "vpo_planned": np.nan,
            "besluit": 1,
            "reden_dienstreden": np.nan,
            "expected_outcome": 0,
        },
        {
            # Screening IC rejected for reason not in list
            "is_onderzoek_hh": False,
            "is_screening_hh": False,
            "is_screening_ic": True,
            "srp_id": np.nan,
            "vpo_planned": 0,
            "besluit": 3,
            "reden_dienstreden": 0,
            "expected_outcome": 0,
        },
        {
            # Screening IC rejected for reason in list
            "is_onderzoek_hh": False,
            "is_screening_hh": False,
            "is_screening_ic": True,
            "srp_id": np.nan,
            "vpo_planned": 0,
            "besluit": 3,
            "reden_dienstreden": 6,
            "expected_outcome": 1,
        },
    ]


class TestLabelAndFiltering:
    @pytest.mark.parametrize("input_row", label_test_inputs())
    def test_add_label(self, input_row, label_test_transformer):
        input_df = pd.DataFrame.from_records(input_row, index=[0])
        output = label_test_transformer.add_label(input_df)
        assert output.at[0, "onderzoekswaardig"] == output.at[0, "expected_outcome"]
