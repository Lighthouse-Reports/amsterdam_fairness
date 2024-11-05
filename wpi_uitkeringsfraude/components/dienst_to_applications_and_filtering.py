import logging
from typing import List

import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.components import Component
from wpi_onderzoekswaardigheid_aanvraag.components.component import (
    NoFittingRequiredMixin,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.function_decorators import (
    log_filtering_step,
)

logger = logging.getLogger(__name__)


class DienstToApplicationsAndFiltering(NoFittingRequiredMixin, Component):
    """Create base dataset of applications by filtering dienst dataframe."""

    def __init__(self, core_productnr: List[int], *args, **kwargs):
        """Class to create a dataset of applications from a set of diensten doing some basic cleaning and filtering.

        Parameters
        ----------
        core_productnr: list of the product numbers the model will be used for, dataset is filtered on these.
        """
        super().__init__(*args, **kwargs)
        self.core_productnr = core_productnr

    def _transform(  # type: ignore
        self,
        scoring: bool,
        dienst: pd.DataFrame,
    ) -> pd.DataFrame:
        """Filter the data for relevant rows.

        Parameters
        ----------
        scoring: if True, do not apply filters for training dataset
        dienst: dienst dataframe

        Returns
        -------
        :
            filtered data
        """
        applications = dienst.pipe(self.rename_columns)

        if not scoring:
            applications = (
                applications.pipe(self.filter_dates)
                .pipe(self.filter_productnr)
                .pipe(self.drop_missing_application_date)
                .pipe(self.drop_duplicate_diensten)
            )
        return applications

    @classmethod
    def rename_columns(cls, dienst: pd.DataFrame):
        """Rename columns:
        - `subjectnrklant` to `subjectnr` because that's what it's called everywhere else
        - `dienstnr` to `application_dienstnr` to distinguish from dienstnr of related diensten

        Parameters
        ----------
        dienst: dienst dataframe

        Returns
        -------
        :
            renamed dataframe
        """
        return dienst.rename(
            columns={
                "subjectnrklant": "subjectnr",
                "dienstnr": "application_dienstnr",
                "productnr": "application_productnr",
            }
        )

    @classmethod
    @log_filtering_step
    def filter_dates(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out diensten with an application date far in the past.

        Parameters
        ----------
        df: dienst dataframe

        Returns
        -------
        :
            filtered data
        """
        return df[df["dtaanvraag"] >= "2015-01-01"]

    @log_filtering_step
    def filter_productnr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out diensten with an out-of-scope producnr.

        Parameters
        ----------
        df: dienst dataframe

        Returns
        -------
        :
            filtered data
        """
        return df[df["application_productnr"].isin(self.core_productnr)]

    @classmethod
    @log_filtering_step
    def drop_missing_application_date(cls, df):
        """Filter out diensten without an application date.

        Parameters
        ----------
        df: dienst dataframe

        Returns
        -------
        :
            filtered data
        """
        return df.dropna(subset=["dtaanvraag"])

    @classmethod
    @log_filtering_step
    def drop_duplicate_diensten(cls, df):
        """For every dienst, keep only the first ever row in Socrates, because
        that represents the information at the time of application (and thus at
        the time of prediction).

        Note that for the dienst features multiple rows may be relevant, but
        not for building our dataset of applications (which we're doing here).

        Parameters
        ----------
        df: dienst dataframe

        Returns
        -------
        :
            filtered data
        """
        return df.sort_values("dtopvoer", ascending=True).drop_duplicates(
            ["application_dienstnr"], keep="first"
        )
