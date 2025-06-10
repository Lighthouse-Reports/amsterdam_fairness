import datetime as dt

import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.common import replace_nan_with_zero
from wpi_onderzoekswaardigheid_aanvraag.components import Component
from wpi_onderzoekswaardigheid_aanvraag.components.component import (
    NoFittingRequiredMixin,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.rows_by_date_filter import (
    SocratesRelevantPeriodFilter,
)


class SocratesStopzettingFeatures(NoFittingRequiredMixin, Component):
    def _transform(  # type: ignore
        self,
        scoring: bool,
        applications: pd.DataFrame,
        dienst_history: pd.DataFrame,
        stopzetting: pd.DataFrame,
        *args,
        **kwargs,
    ):
        applications_stopzetting_raw = self.join_applications_stopzetting(
            dienst_history, stopzetting
        )
        result = self.add_features(applications, applications_stopzetting_raw)
        return result

    @classmethod
    def join_applications_stopzetting(
        cls,
        dienst_history: pd.DataFrame,
        stopzetting: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge dienst data with stopzetting data.

        Parameters
        ----------
        dienst_history: dataframe with history of relevant diensten per application
        stopzetting: stopzetting dataframe

        Returns
        -------
        :
            dienst_history and stopzetting dataframe joined
        """
        applications_stopzetting_raw = pd.merge(
            dienst_history,
            stopzetting.add_suffix("_stopzetting"),
            how="inner",
            left_on="dienstnr_dienst",
            right_on="dienstnr_stopzetting",
        )
        return applications_stopzetting_raw

    @classmethod
    def add_features(
        cls, applications: pd.DataFrame, applications_stopzetting: pd.DataFrame
    ):
        """Add stopzetting features to applications.

        Parameters
        ----------
        applications: applications dataframe
        applications_stopzetting: applications dataframe with stopzetting data merged to it

        Returns
        -------
        :
            dienst dataframe with stopzetting features
        """
        applications_stopzetting = cls.filter_stopzetting_relevant_to_application(
            applications_stopzetting
        )

        features = cls.calc_features(applications_stopzetting)
        result = applications.merge(features, on="application_dienstnr", how="left")

        # People with no stopzetting at all will have NaN features, but counts
        # should then actually be zero.
        result = replace_nan_with_zero(
            result,
            columns=[
                "stopzetting_count_last_year",
            ],
        )

        result.name = "applications_stopzetting"
        return result

    @classmethod
    def filter_stopzetting_relevant_to_application(
        cls, applications_stopzetting: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter stopzettingen that are relevant to an application.

        Parameters
        ----------
        applications_stopzetting: applications dataframe with stopzetting data merged to it

        Returns
        -------
        :
            filtered applications_stopzetting dataframe
        """
        filterer = SocratesRelevantPeriodFilter(
            ref_date_col="first_dtopvoer",
            begindatum_col="dtbeginstopzetting_stopzetting",
            einddatum_col="dtafvoer_stopzetting",
            opvoer_col="dtopvoer_stopzetting",
            afvoer_col="dtafvoer_stopzetting",
            period=dt.timedelta(days=365),
        )
        return filterer.fit_transform(applications_stopzetting)

    @classmethod
    def calc_features(cls, applications_stopzetting: pd.DataFrame) -> pd.DataFrame:
        """Calculate stopzetting features:
        - stopzetting_count_last_year: Number of voorwaarden

        Parameters
        ----------
        applications_stopzetting: dienst dataframe with stopzetting data merged to it

        Returns
        -------
        :
            dataframe of features
        """
        features = applications_stopzetting.groupby("application_dienstnr").agg(
            stopzetting_count_last_year=pd.NamedAgg(
                column="reden_stopzetting", aggfunc="size"
            ),
        )
        return features
