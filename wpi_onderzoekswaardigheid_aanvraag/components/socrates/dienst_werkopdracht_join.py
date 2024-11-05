import logging

import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.components import Component
from wpi_onderzoekswaardigheid_aanvraag.components.component import (
    NoFittingRequiredMixin,
)

logger = logging.getLogger(__name__)


class SocratesDienstWerkopdrachtJoin(NoFittingRequiredMixin, Component):
    def _transform(  # type: ignore
        self,
        scoring: bool,
        dienst: pd.DataFrame,
        werkopdracht: pd.DataFrame,
        *args,
        **kwargs,
    ):
        return self.join_dienst_werkopdracht(dienst, werkopdracht)

    @classmethod
    def join_dienst_werkopdracht(
        cls, dienst: pd.DataFrame, werkopdracht: pd.DataFrame
    ) -> pd.DataFrame:
        """Add the first `dtopvoer` to a dienst by checking the generated
        werkopdracht for processing a new application.

        The `dtopvoer` in Socrates Dienst gets overwritten when a decision
        gets made about the application, but we need the very first `dtopvoer`
        for linking applications to a proces for generating the label.
        Socrates Werkopdracht contains this original `dtopvoer`.

        Parameters
        ----------
        dienst: dienst dataframe
        werkopdracht: werkopdracht dataframe

        Returns
        -------
        :
            dienst with new column containing first dtopvoer
        """
        # Dataset only contains werkopdracht soorten related to a new application,
        # so no need to filter on the 'soort'.
        result = dienst.merge(
            werkopdracht[["dienstnr", "dtopvoer"]].add_suffix("_werkopdracht"),
            left_on="application_dienstnr",
            right_on="dienstnr_werkopdracht",
            how="inner",
        )
        # A very small number of diensten have two aanvraag werkopdrachten.
        # Keep the first one because it's the earliest moment an investigation
        # could be ordered.
        result = result.sort_values(
            "dtopvoer_werkopdracht", ascending=True
        ).drop_duplicates(subset=["application_dienstnr"], keep="first")
        result.name = "dienst_werkopdracht"
        result = result.rename(columns={"dtopvoer_werkopdracht": "first_dtopvoer"})
        return result
