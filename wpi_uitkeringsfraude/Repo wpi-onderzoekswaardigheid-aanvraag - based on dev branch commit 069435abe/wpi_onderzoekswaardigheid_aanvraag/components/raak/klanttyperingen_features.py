import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.common import replace_nan_with_zero
from wpi_onderzoekswaardigheid_aanvraag.components import Component
from wpi_onderzoekswaardigheid_aanvraag.components.component import (
    NoFittingRequiredMixin,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.rows_by_date_filter import (
    date_in_ref_period,
)


class RaakKlanttyperingenFeatures(NoFittingRequiredMixin, Component):
    def _transform(  # type: ignore
        self,
        scoring: bool,
        applications: pd.DataFrame,
        kltyp: pd.DataFrame,
        *args,
        **kwargs,
    ):
        applications_kltyp_raw = self.join_applications_kltyp(applications, kltyp)
        result = self.add_features(applications, applications_kltyp_raw)
        return result

    @classmethod
    def join_applications_kltyp(
        cls, applications: pd.DataFrame, kltyp: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge dienst data with klanttyperingen data.

        Parameters
        ----------
        applications: applications dataframe
        kltyp: klanttyperingen dataframe

        Returns
        -------
        :
            applications and kltyp dataframe joined
        """
        applications_kltyp_raw = pd.merge(
            applications[pd.notnull(applications["subjectnr"])],
            kltyp.add_suffix("_kltyp"),
            how="inner",
            left_on="subjectnr",
            right_on="administratienummer_kltyp",
        )

        return applications_kltyp_raw

    def add_features(
        self,
        applications: pd.DataFrame,
        applications_kltyp: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add klanttyperingen features to applications.

        Parameters
        ----------
        applications: applications dataframe
        applications_kltyp: applications dataframe with klanttyperingen merged to it

        Returns
        -------
        :
            applications dataframe with klanttyperingen features
        """
        applications_kltyp = self.filter_kltyp_relevant_to_application(
            applications_kltyp
        )
        features = self.calc_features(applications_kltyp)
        result = applications.merge(features, how="left", on="application_dienstnr")
        result = replace_nan_with_zero(
            result,
            columns=[
                "klanttypering_schuldhulpverlening_last_year",
            ],
        )
        result.name = "applications_kltyp"
        return result

    @classmethod
    def filter_kltyp_relevant_to_application(cls, applications_kltyp):
        """Filter klanttyperingen that are relevant to an application.

        Parameters
        ----------
        applications_kltyp: applications dataframe with klanttyperingen merged to it

        Returns
        -------
        :
            filtered applications_kltyp dataframe
        """
        result = applications_kltyp.copy()
        ref_period_days = 365

        cond1 = date_in_ref_period(
            result,
            ref_col="first_dtopvoer",
            check_col="aanvang_klanttypering_kltyp",
            period_days=ref_period_days,
        )
        cond2 = date_in_ref_period(
            result,
            ref_col="first_dtopvoer",
            check_col="einde_klanttypering_kltyp",
            period_days=ref_period_days,
        )
        return result[cond1 | cond2]

    @classmethod
    def calc_features(cls, applications_kltyp):
        """

        Parameters
        ----------
        applications_kltyp: applications dataframe with klanttyperingen data merged to it

        Returns
        -------
        :
            dataframe of features
        """
        df = applications_kltyp.copy()
        features = cls._has_had_schuldhulpverlening(df)
        return features

    @staticmethod
    def _has_had_schuldhulpverlening(df: pd.DataFrame) -> pd.Series:
        df["kltyp_contains_SHV"] = df["klanttypering_kltyp"].str.contains("SHV")
        feature = (
            df.groupby("application_dienstnr")["kltyp_contains_SHV"].sum() > 0
        ).astype(int)
        feature.name = "klanttypering_schuldhulpverlening_last_year"
        return feature
