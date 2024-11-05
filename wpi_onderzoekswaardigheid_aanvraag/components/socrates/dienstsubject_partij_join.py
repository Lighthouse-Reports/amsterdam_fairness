import logging

import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.components import Component
from wpi_onderzoekswaardigheid_aanvraag.components.component import (
    NoFittingRequiredMixin,
)

logger = logging.getLogger(__name__)


class SocratesDienstsubjectPartijJoin(NoFittingRequiredMixin, Component):
    def _transform(  # type: ignore
        self,
        scoring: bool,
        dienstsubject: pd.DataFrame,
        partij: pd.DataFrame,
        *args,
        **kwargs,
    ):
        return SocratesDienstsubjectPartijJoin.join_dienstsubject_partij(
            dienstsubject, partij
        )

    @classmethod
    def join_dienstsubject_partij(
        cls, dienstsubject: pd.DataFrame, partij: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge dienstsubject data with partij data to get information about
        persoon in the dienstsubject table.

        Parameters
        ----------
        dienstsubject: dienstsubject dataframe
        partij: partij dataframe

        Returns
        -------
        :
            dienstsubject joined with partij
        """
        # Note: the subjectnr from Dienstsubject refers to the partijnr from
        # Partij, unlike the subjectnr from Dienst, which refers to the
        # persoonnr from Partij.
        result = dienstsubject.merge(
            partij, left_on="subjectnr", right_on="partijnr", how="left"
        )

        result = result[
            [
                "dienstnr",
                "persoonnr",
                "dtbegin",
                "dteinde",
                "dtopvoer",
                "dtafvoer",
                "geldig",
            ]
        ].rename(  # Rename columns to match the Dienst table.
            columns={
                "persoonnr": "subjectnr",
                "dtbegin": "dtbegindienst",
                "dteinde": "dteindedienst",
            }
        )
        result.name = "dienstsubject_partij"
        return result
