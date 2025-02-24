import numpy as np
import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.common import replace_nan_with_zero
from wpi_onderzoekswaardigheid_aanvraag.components import Component
from wpi_onderzoekswaardigheid_aanvraag.components.component import (
    NoFittingRequiredMixin,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.rows_by_date_filter import (
    date_in_ref_period,
)


class RaakDeelnamesFeatures(NoFittingRequiredMixin, Component):
    def _transform(  # type: ignore
        self,
        scoring: bool,
        applications: pd.DataFrame,
        deelnames: pd.DataFrame,
        *args,
        **kwargs,
    ):
        applications_deelnames_raw = self.join_applications_deelnames(
            applications, deelnames
        )
        result = self.add_features(applications, applications_deelnames_raw)
        return result

    @classmethod
    def join_applications_deelnames(
        cls, applications: pd.DataFrame, deelnames: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge dienst data with deelnames data.

        Parameters
        ----------
        applications: applications dataframe
        deelnames: deelnames dataframe

        Returns
        -------
        :
            dienst and deelnames dataframe joined
        """
        applications_deelnames_raw = pd.merge(
            applications[pd.notnull(applications["subjectnr"])],
            deelnames.add_suffix("_deelnames"),
            how="inner",
            left_on="subjectnr",
            right_on="administratienummer_deelnames",
        )

        return applications_deelnames_raw

    def add_features(
        self,
        applications: pd.DataFrame,
        applications_deelnames: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add deelnames features to applications.

        Parameters
        ----------
        applications: applications dataframe
        applications_deelnames: applications dataframe with deelnames data merged to it

        Returns
        -------
        :
            applications dataframe with deelname features
        """
        applications_deelnames = self.filter_deelnames_relevant_to_application(
            applications_deelnames
        )
        features = self.calc_features(applications_deelnames)

        result = applications.merge(features, how="left", on="application_dienstnr")
        result = replace_nan_with_zero(
            result,
            columns=[
                "deelnames_count_last_year",
                "deelnames_started_count_last_year",
                "deelnames_started_percentage_last_year",
                "deelnames_ended_early_count_last_year",
                "deelnames_not_finished_count_last_year",
                "deelnames_ended_because_behavior_count_last_year",
                "deelnames_ended_because_work_or_education_count_last_year",
                "deelnames_ended_early_percentage_last_year",
                "deelnames_not_finished_percentage_last_year",
            ],
        )

        result.name = "applications_deelnames"
        return result

    @classmethod
    def filter_deelnames_relevant_to_application(cls, applications_deelnames):
        """Filter deelnames that are relevant to an application.

        Parameters
        ----------
        applications_deelnames: applications dataframe with deelnames data merged to it

        Returns
        -------
        :
            filtered applications_deelnames dataframe
        """
        result = applications_deelnames.copy()
        ref_period_days = 365

        cond1 = date_in_ref_period(
            result,
            ref_col="first_dtopvoer",
            check_col="datum_aanmelding_deelnames",
            period_days=ref_period_days,
        )
        cond2 = date_in_ref_period(
            result,
            ref_col="first_dtopvoer",
            check_col="start_deelname_deelnames",
            period_days=ref_period_days,
        )
        cond3 = date_in_ref_period(
            result,
            ref_col="first_dtopvoer",
            check_col="einde_deelname_deelnames",
            period_days=ref_period_days,
        )
        return result[cond1 | cond2 | cond3]

    @classmethod
    def calc_features(cls, applications_deelnames):
        """Calculate deelnames features.

        Parameters
        ----------
        applications_deelnames: applications dataframe with deelnames data merged to it

        Returns
        -------
        :
            dataframe of features
        """
        df = applications_deelnames.copy()

        df["deelname_started"] = cls._deelname_started(df)
        df["deelname_ended_early"] = cls._deelname_ended_early(df)
        df["deelname_not_finished"] = cls._deelname_not_finished(df)
        df["ended_because_behavior"] = cls._ended_because_behavior(df)
        df["ended_because_work_or_education"] = cls._ended_because_work_or_education(df)

        features = df.groupby("application_dienstnr").agg(
            deelnames_count_last_year=pd.NamedAgg(
                column="administratienummer_deelnames", aggfunc="size"
            ),
            deelnames_started_count_last_year=pd.NamedAgg(
                column="deelname_started", aggfunc="sum"
            ),
            deelnames_started_percentage_last_year=pd.NamedAgg(
                column="deelname_started", aggfunc="mean"
            ),
            deelnames_ended_early_count_last_year=pd.NamedAgg(
                column="deelname_ended_early", aggfunc="sum"
            ),
            deelnames_not_finished_count_last_year=pd.NamedAgg(
                column="deelname_not_finished", aggfunc="sum"
            ),
            deelnames_ended_because_behavior_count_last_year=pd.NamedAgg(
                column="ended_because_behavior", aggfunc="sum"
            ),
            deelnames_ended_because_work_or_education_count_last_year=pd.NamedAgg(
                column="ended_because_work_or_education", aggfunc="sum"
            ),
        )

        # Calculate those outside of NamedAgg, because it makes more sense to
        # have the nr of started deelnames as the numerator than the total nr of
        # deelnames.
        features["deelnames_ended_early_percentage_last_year"] = (
            features["deelnames_ended_early_count_last_year"]
            / features["deelnames_started_count_last_year"]
        )
        features["deelnames_not_finished_percentage_last_year"] = (
            features["deelnames_not_finished_count_last_year"]
            / features["deelnames_started_count_last_year"]
        )

        return features.replace(np.inf, np.nan)

    @staticmethod
    def _deelname_started(df: pd.DataFrame):
        return (
            ~df["start_deelname_deelnames"].isna()
            | ~df["einde_deelname_deelnames"].isna()
        )

    @staticmethod
    def _deelname_ended_early(df: pd.DataFrame):
        return df["reden_voortijdig_afgebroken_deelnames"] != ""

    @staticmethod
    def _deelname_not_finished(df: pd.DataFrame):
        return (
            (df["reden_niet_geaccepteerd_deelnames"] != "")
            | (df["reden_niet_gestart_deelnames"] != "")
            | (df["reden_voortijdig_afgebroken_deelnames"] != "")
        )

    @classmethod
    def _ended_because_behavior(cls, df: pd.DataFrame):
        return (
            cls._niet_geaccepteerd_behavior(df["reden_niet_geaccepteerd_deelnames"])
            | cls._niet_gestart_behavior(df["reden_niet_gestart_deelnames"])
            | cls._voortijdig_afgebroken_behavior(
                df["reden_voortijdig_afgebroken_deelnames"]
            )
        )

    @staticmethod
    def _niet_geaccepteerd_behavior(s: pd.Series) -> pd.Series:
        return s.isin(["weggebleven zonder opgave", "niet gemotiveerd"])

    @staticmethod
    def _niet_gestart_behavior(s: pd.Series) -> pd.Series:
        return s.isin(
            [
                "deelnemer niet verschenen",
                "motivatie deelnemer ontoereikend",
                "meerdere keren niet verschenen",
                "werknemer weigert",
                "negatief ontslag-gedrag",
            ]
        )

    @staticmethod
    def _voortijdig_afgebroken_behavior(s: pd.Series) -> pd.Series:
        return s.isin(
            [
                "onvoldoende gemotiveerd",
                "veelvuldige absentie",
                "onvoldoende vorderingen",
                "uit dienst (ontslag)",
                "ontslag op staande voet (activering)",
                "ontslag in proeftijd (activering)",
            ]
        )

    @staticmethod
    def _ended_because_work_or_education(df: pd.DataFrame):
        return (
            df["reden_niet_geaccepteerd_deelnames"].isin(["betaald werk"])
            | df["reden_niet_gestart_deelnames"].isin(
                [
                    "aanvaarding reguliere baan",
                    "aanvaarding reguliere scholing",
                ]
            )
            | df["reden_voortijdig_afgebroken_deelnames"].isin(
                ["regulier werk gevonden", "andere opleiding", "uit dienst (baan)"]
            )
        )
