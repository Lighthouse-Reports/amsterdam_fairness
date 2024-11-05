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


class SocratesVoorwaardeFeatures(NoFittingRequiredMixin, Component):

    relevant_voorwaarde_for_onderzoekswaardigheid = [1, 40, 41, 68, 309]

    def _transform(  # type: ignore
        self,
        scoring: bool,
        applications: pd.DataFrame,
        voorwaarde: pd.DataFrame,
        *args,
        **kwargs,
    ):
        applications_voorwaarde_raw = self.join_applications_voorwaarde(
            applications, voorwaarde
        )
        result = self.add_features(applications, applications_voorwaarde_raw)
        return result

    @classmethod
    def join_applications_voorwaarde(
        cls, applications: pd.DataFrame, voorwaarde: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge dienst data with voorwaarde data.

        Parameters
        ----------
        applications: applications dataframe
        voorwaarde: voorwaarde dataframe

        Returns
        -------
        :
            dienst and voorwaarde dataframe joined
        """
        applications_voorwaarde_raw = pd.merge(
            applications[pd.notnull(applications["subjectnr"])],
            voorwaarde.add_suffix("_voorwaarde"),
            how="inner",
            left_on="subjectnr",
            right_on="subjectnr_voorwaarde",
        )
        return applications_voorwaarde_raw

    @classmethod
    def add_features(
        cls, applications: pd.DataFrame, applications_voorwaarde: pd.DataFrame
    ):
        """Add voorwaarde features to applications.

        Parameters
        ----------
        applications: applications dataframe
        applications_voorwaarde: applications dataframe with voorwaarde data merged to it

        Returns
        -------
        :
            applications dataframe with voorwaarde features
        """
        applications_voorwaarde = cls.filter_voorwaarde_relevant_to_application(
            applications_voorwaarde
        )
        features = cls.calc_features(applications_voorwaarde)
        result = applications.merge(features, how="left", on="application_dienstnr")

        # People with no voorwaarden at all were not in applications_voorwaarde_raw, so
        # will have NaN features, but some features should then actually be zero.
        result = replace_nan_with_zero(
            result,
            columns=[
                "voorwaarde_count_last_year",
                "unique_voorwaarde_count_last_year",
            ],
        )
        result.name = "applications_voorwaarde"
        return result

    @classmethod
    def filter_voorwaarde_relevant_to_application(
        cls, applications_voorwaarde: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter voorwaarden that are relevant to an application.

        Parameters
        ----------
        applications_voorwaarde: applications dataframe with voorwaarde data merged to it

        Returns
        -------
        :
            filtered applications_voorwaarde dataframe
        """
        filterer = SocratesRelevantPeriodFilter(
            ref_date_col="first_dtopvoer",
            begindatum_col="dtbegin_voorwaarde",
            einddatum_col="dteinde_voorwaarde",
            opvoer_col="dtopvoer_voorwaarde",
            afvoer_col="dtafvoer_voorwaarde",
            period=dt.timedelta(days=365),  # Look at voorwaarden up to 1 year ago
        )
        return filterer.fit_transform(applications_voorwaarde)

    @classmethod
    def calc_features(cls, applications_voorwaarde: pd.DataFrame) -> pd.DataFrame:
        """Calculate voorwaarde features:
        - unique_voorwaarde_count_last_year: Number of unique types of voorwaarde

        Parameters
        ----------
        applications_voorwaarde: applications dataframe with voorwaarde data merged to it

        Returns
        -------
        :
            dataframe of features
        """
        relevant_voorwaarden = applications_voorwaarde[
            applications_voorwaarde["soort_voorwaarde"].isin(
                cls.relevant_voorwaarde_for_onderzoekswaardigheid
            )
        ]

        features = relevant_voorwaarden.groupby("application_dienstnr").agg(
            voorwaarde_count_last_year=pd.NamedAgg(
                column="soort_voorwaarde", aggfunc="size"
            ),
            unique_voorwaarde_count_last_year=pd.NamedAgg(
                column="soort_voorwaarde", aggfunc=pd.Series.nunique
            ),
        )

        return features
