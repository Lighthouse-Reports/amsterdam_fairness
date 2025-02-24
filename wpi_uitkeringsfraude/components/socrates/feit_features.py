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


class SocratesFeitFeatures(NoFittingRequiredMixin, Component):
    def _transform(  # type: ignore
        self,
        scoring: bool,
        applications: pd.DataFrame,
        feit: pd.DataFrame,
        *args,
        **kwargs,
    ):
        applications_feit_raw = self.join_applications_feit(applications, feit)
        result = self.add_features(applications, applications_feit_raw)
        return result

    @classmethod
    def join_applications_feit(
        cls, applications: pd.DataFrame, feit: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge dienst data with feit data.

        Parameters
        ----------
        applications: applications dataframe
        feit: feit dataframe

        Returns
        -------
        :
            dienst and feit dataframe joined
        """
        applications_feit_raw = pd.merge(
            applications[pd.notnull(applications["subjectnr"])],
            feit.add_suffix("_feit"),
            how="inner",
            left_on="subjectnr",
            right_on="subjectnr_feit",
        )
        return applications_feit_raw

    @classmethod
    def add_features(
        cls, applications: pd.DataFrame, applications_feit: pd.DataFrame
    ) -> pd.DataFrame:
        """Add feit features to applications.

        Parameters
        ----------
        applications: applications dataframe
        applications_feit: applications dataframe with feit data merged to it

        Returns
        -------
        :
            applications dataframe with feit features
        """
        applications_feit = cls.filter_feit_relevant_to_application(applications_feit)
        features = cls.calc_features(applications_feit)
        result = applications.merge(features, how="left", on="application_dienstnr")

        # People with no feiten at all were not in applications_feit, so
        # will have NaN features, but some features should then actually be zero.
        result = replace_nan_with_zero(
            result,
            columns=[
                "feit_count_last_year",
                "unique_feit_count_last_year",
                "avg_percentage_maatregel",
                "no_maatregel_count",
            ],
        )
        result.name = "applications_feit"

        return result

    @classmethod
    def filter_feit_relevant_to_application(cls, applications_feit):
        """Filter feit that is relevant to an application.

        Parameters
        ----------
        applications_feit: applications dataframe with with feit data merged to it

        Returns
        -------
        :
            filtered applications_feit dataframe
        """
        filterer = SocratesRelevantPeriodFilter(
            ref_date_col="first_dtopvoer",
            begindatum_col="dtconstatering_feit",
            einddatum_col="dtconstatering_feit",
            opvoer_col="dtopvoer_feit",
            afvoer_col="dtafvoer_feit",
            period=dt.timedelta(days=365),  # Look at feiten up to 1 year ago
        )
        return filterer.fit_transform(applications_feit)

    @classmethod
    def calc_features(cls, applications_feit):
        """Calculate feit features:
        - feit_count_last_year: Number of feiten
        - unique_feit_count_last_year: Number of unique types of feit
        - avg_percentage_maatregel: Average reduction related to the feiten (proxy for seriousness of the feiten)
        - no_maatregel_count: Number of times a feit had no consequences

        Parameters
        ----------
        applications_feit: applications dataframe with with feit data merged to it

        Returns
        -------
        :
            dataframe of features
        """
        features = applications_feit.groupby("application_dienstnr").agg(
            feit_count_last_year=pd.NamedAgg(column="soort_feit", aggfunc="size"),
            unique_feit_count_last_year=pd.NamedAgg(
                column="soort_feit", aggfunc=pd.Series.nunique
            ),
            avg_percentage_maatregel=pd.NamedAgg(
                column="percentage_feit", aggfunc="mean"
            ),
        )
        features = features.join(count_no_maatregel(applications_feit))
        return features


def count_no_maatregel(df: pd.DataFrame):
    """Count how often a feit did not result in a maatregel.

    The translation of codes is:
    1: ander feit met maatregel over dezelfde periode
    2: dringende reden
    3: maatregel via handmatige component

    Only 2 truly means no maatregel, hence only 2s are counted.
    """
    df["rdgeenmaatregel_feit_equals_2"] = df["rdgeenmaatregel_feit"] == 2
    feature = df.groupby("application_dienstnr")["rdgeenmaatregel_feit_equals_2"].sum()
    feature.name = "no_maatregel_count"
    return pd.DataFrame(feature)
