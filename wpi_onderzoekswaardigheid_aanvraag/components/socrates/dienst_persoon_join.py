import logging

import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.components import Component
from wpi_onderzoekswaardigheid_aanvraag.components.component import (
    NoFittingRequiredMixin,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.rows_by_date_filter import (
    SocratesRelevantDateFilter,
)

logger = logging.getLogger(__name__)


class SocratesDienstPersoonJoin(NoFittingRequiredMixin, Component):
    def _transform(  # type: ignore
        self,
        scoring: bool,
        dienst: pd.DataFrame,
        persoon: pd.DataFrame,
        *args,
        **kwargs,
    ):
        result = self.join_dienst_persoon(dienst, persoon)
        return self.rename_cols(result)

    @classmethod
    def join_dienst_persoon(
        cls, dienst: pd.DataFrame, persoon: pd.DataFrame
    ) -> pd.DataFrame:
        """Join dienst data with persoon data.

        Parameters
        ----------
        dienst: dienst dataframe
        persoon: persoon dataframe

        Returns
        -------
        :
            joined data
        """
        dienst_persoon_raw = pd.merge(
            dienst,
            persoon[pd.notnull(persoon["subjectnr"])].add_suffix("_persoon"),
            how="left",
            left_on="subjectnr",
            right_on="subjectnr_persoon",
        )
        persoon_info = cls.filter_persoon_relevant_to_application(dienst_persoon_raw)
        persoon_info = persoon_info.sort_values(
            "dtopvoer", ascending=True
        ).drop_duplicates(subset=["application_dienstnr"], keep="last")

        applications_before = set(dienst["application_dienstnr"].unique())
        applications_after = set(persoon_info["application_dienstnr"].unique())

        applications_disappeared = applications_before.difference(applications_after)
        if len(applications_disappeared) != 0:
            logger.warning(
                f"People from the following applications could not be matched to information from Socrates "
                f"Persoon, dienstnr: {applications_disappeared}"
            )

        result = dienst.merge(
            persoon_info[
                [c for c in persoon_info.columns if "_persoon" in c]
                + ["application_dienstnr"]
            ],
            on="application_dienstnr",
            how="left",
        )

        result.name = "persoon"
        return result

    @classmethod
    def filter_persoon_relevant_to_application(cls, dienst_persoon):
        """Filter rows in persoon that are relevant to an application.

        Parameters
        ----------
        dienst_persoon: dienst dataframe with persoon data merged to it

        Returns
        -------
        :
            filtered dienst dataframe
        """
        filterer = SocratesRelevantDateFilter(
            ref_date_col="first_dtopvoer",
            opvoer_col="dtopvoer_persoon",
            afvoer_col="dtafvoer_persoon",
        )
        return filterer.fit_transform(dienst_persoon)

    @classmethod
    def rename_cols(cls, dienst_persoon):
        return dienst_persoon.rename(
            columns={
                "nationaliteit1_persoon": "nationaliteit",
            }
        )
