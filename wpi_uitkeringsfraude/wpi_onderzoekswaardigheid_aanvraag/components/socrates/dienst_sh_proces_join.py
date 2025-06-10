import datetime as dt
import logging

import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.components import Component
from wpi_onderzoekswaardigheid_aanvraag.components.component import (
    NoFittingRequiredMixin,
)

logger = logging.getLogger(__name__)


class SocratesDienstSherlockProcesJoin(NoFittingRequiredMixin, Component):
    def _transform(  # type: ignore
        self,
        scoring: bool,
        dienst_werkopdracht: pd.DataFrame,
        proces_klant: pd.DataFrame,
        *args,
        **kwargs,
    ):
        """Add proces information to diensten.

        Parameters
        ----------
        scoring: if False, join data about the proces that was carried out because of the application;
        else, return `dienst_werkopdracht` unchanged.
        dienst_werkopdracht: dienst dataframe with werkopdracht joined to it
        proces_klant: proces dataframe with klant data joined to it

        Returns
        -------
        :
            dataframe
        """
        result = dienst_werkopdracht

        if not scoring:
            dienst_proces_raw = self.join_dienst_proces(
                dienst_werkopdracht, proces_klant
            )
            result = self.add_relevant_proces(result, dienst_proces_raw)

        return result

    @classmethod
    def join_dienst_proces(
        cls,
        dienst: pd.DataFrame,
        proces_klant: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge proces data with dienst data.

        Parameters
        ----------
        dienst: dienst dataframe
        proces_klant: proces dataframe with kln_adminnummer joined to it

        Returns
        -------
        :
            dienst and proces dataframe joined
        """
        dienst_proces_raw = dienst.merge(
            proces_klant,
            how="left",
            left_on="subjectnr",
            right_on="kln_adminnummer",
        )
        return dienst_proces_raw

    @classmethod
    def add_relevant_proces(cls, dienst, dienst_proces):
        diensten_with_proces = cls.find_relevant_proces(dienst_proces)
        result = dienst.merge(
            diensten_with_proces, how="left", on="application_dienstnr"
        )
        return result

    @classmethod
    def find_relevant_proces(cls, dienst_proces):
        """For every application, find the proces that was carried out because
        of the application, if there is one.

        No direct key between dienst and proces exists, so this function uses a
        rule of thumb: If a proces started within 28 days after the opvoer of a
        dienst, then this application probably triggered the proces.

        Parameters
        ----------
        dienst_proces: dienst dataframe with proces data merged to it

        Returns
        -------
        :
            diensten with information about the relevant application proces
        """
        n_before = dienst_proces[
            "application_dienstnr"
        ].nunique()  # For logging purposes

        # Make sure we only consider processes for which the procescode indicates
        # that they were related to an application.
        dienst_proces = cls.filter_application_related_proces(dienst_proces)

        dienst_opvoer = dienst_proces["first_dtopvoer"]
        dienst_opvoer_plus28d = dienst_proces["first_dtopvoer"] + dt.timedelta(days=28)

        rel_rows = dienst_proces[
            dienst_proces["pro_startdatum"].between(
                dienst_opvoer, dienst_opvoer_plus28d
            )
        ]

        # In case an application matches with multiple processen, take the proces with
        # the closest start date.
        diensten_with_proces = rel_rows.sort_values(
            "pro_startdatum", ascending=True
        ).drop_duplicates(["application_dienstnr"], keep="first")

        logger.debug(
            f"Unable to make match with a proces for {n_before - len(diensten_with_proces)} out of {n_before} rows"
        )
        return diensten_with_proces[
            [
                "application_dienstnr",
                "pro_id",
                "srp_id",
                "spr_id",
                "pro_startdatum",
                "pro_einddatum",
                "sre_id",
                "pro_teamactueelid",
            ]
        ]

    @classmethod
    def filter_application_related_proces(cls, proces: pd.DataFrame) -> pd.DataFrame:
        result = proces.dropna(subset=["spr_id"])
        result = result[result["spr_id"] == 146]
        return result
