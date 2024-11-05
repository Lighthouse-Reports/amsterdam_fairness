import logging

import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.components import Component
from wpi_onderzoekswaardigheid_aanvraag.components.component import (
    NoFittingRequiredMixin,
)

logger = logging.getLogger(__name__)


class ApplicationFeatures(NoFittingRequiredMixin, Component):
    def _transform(  # type: ignore
        self,
        scoring: bool,
        applications: pd.DataFrame,
        *args,
        **kwargs,
    ):
        result = self.add_features(applications)
        if not scoring:
            result = self.drop_long_aanvraag_opvoer_gap(result)
        return result

    @classmethod
    def add_features(cls, applications: pd.DataFrame) -> pd.DataFrame:
        """Create features about the application.

        Parameters
        ----------
        applications: dienst dataframe processed to contain applications

        Returns
        -------
        :
            applications with new features
        """
        result = applications.copy()
        result["aanvraag_opvoer_gap_in_days"] = cls.calc_aanvraag_opvoer_gap(result)
        return result

    @classmethod
    def calc_aanvraag_opvoer_gap(cls, applications: pd.DataFrame) -> pd.Series:
        """Calculate difference in days between the aanvraag date and opvoer date.

        Parameters
        ----------
        applications: dienst dataframe processed to contain applications

        Returns
        -------
        :
            series of new feature
        """
        return (applications["first_dtopvoer"] - applications["dtaanvraag"]).dt.days

    @classmethod
    def drop_long_aanvraag_opvoer_gap(cls, applications: pd.DataFrame) -> pd.DataFrame:
        """Drop applications with more than 180 days between the aanvraag date and opvoer date,
        because they're assumed to be invalid.

        Parameters
        ----------
        applications: dienst dataframe processed to contain applications

        Returns
        -------
        :
            filtered applications dataframe
        """
        return applications[applications["aanvraag_opvoer_gap_in_days"] <= 180]
