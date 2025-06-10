import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.common import replace_nan_with_zero
from wpi_onderzoekswaardigheid_aanvraag.components import Component
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.rows_by_date_filter import (
    date_in_ref_period,
)


class RaakAfsprakenFeatures(Component):
    def _fit(  # type: ignore
        self,
        applications: pd.DataFrame,
        afspraken: pd.DataFrame,
        *args,
        **kwargs,
    ):
        applications_afspr_raw = self.join_applications_afspr(applications, afspraken)
        fit_attributes = {
            "common_afspraaktype": self._list_common_values(
                applications_afspr_raw, "afspraaktype", lower_lim=300
            ),
            "common_afspraakresultaat": self._list_common_values(
                applications_afspr_raw, "afspraakresultaat", lower_lim=150
            ),
        }
        return fit_attributes

    def _transform(  # type: ignore
        self,
        scoring: bool,
        applications: pd.DataFrame,
        afspraken: pd.DataFrame,
        *args,
        **kwargs,
    ):
        applications_afspr_raw = self.join_applications_afspr(applications, afspraken)
        result = self.add_features(applications, applications_afspr_raw, scoring)
        return result

    @classmethod
    def join_applications_afspr(
        cls, applications: pd.DataFrame, afspraken: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge dienst data with afspraken data.

        Parameters
        ----------
        applications: applications dataframe
        afspraken: afspraken dataframe

        Returns
        -------
        :
            dienst and afspraken dataframe joined
        """
        afspraken["date"] = afspraken["datum_invoer"].combine_first(
            afspraken["datum_afhandeling_gepland"]
        )

        applications_afspr_raw = pd.merge(
            applications[pd.notnull(applications["subjectnr"])],
            afspraken.add_suffix("_afspr"),
            how="inner",
            left_on="subjectnr",
            right_on="administratienummer_afspr",
        )

        return applications_afspr_raw

    @classmethod
    def _list_common_values(
        cls, application_afspr: pd.DataFrame, col: str, lower_lim: int
    ):
        """List all values in column `col` that are common by calculating for
        each how often it occurs on average per year and selecting those that
        occur at least `lower_lim` times.

        Parameters
        ----------
        applications_afspr: applications dataframe with afspraken data merged to it
        col: column to list common values for
        lower_lim: mininum number of times a value has to be seen per year on average to be 'common'

        Returns
        -------
        :
            list of values
        """
        df = application_afspr.copy()
        df = cls.filter_afspraken_relevant_to_application(df)
        df["year"] = df["date_afspr"].dt.year

        yearly_counts = (
            df.dropna(subset=["year"]).groupby("year")[f"{col}_afspr"].value_counts()
        )
        avg_per_year = yearly_counts.groupby(level=1).mean()
        common_values = avg_per_year[avg_per_year > lower_lim].index.tolist()

        # Remove empty afspraaktype/-resultaat.
        if " " in common_values:
            common_values.remove(" ")
        if "" in common_values:
            common_values.remove("")

        return common_values

    def add_features(
        self,
        applications: pd.DataFrame,
        applications_afspr: pd.DataFrame,
        scoring: bool,
    ) -> pd.DataFrame:
        """Add afspraken features to applications.

        Parameters
        ----------
        applications: applications dataframe
        applications_afspr: applications dataframe with afspraken data merged to it
        scoring: whether in scoring mode or not

        Returns
        -------
        :
            applications dataframe with feit features
        """
        applications_afspr = self.filter_afspraken_relevant_to_application(
            applications_afspr
        )
        features = self.calc_features(applications_afspr)

        result = applications.merge(features, how="left", on="application_dienstnr")
        result = replace_nan_with_zero(
            result,
            columns=[
                "afspraken_no_show_count_last_year",
                "afspraken_no_contact_count_last_year",
                "afspraken_geen_recht_houding_last_year",
            ],
        )

        result.name = "applications_afspr"
        return result

    @classmethod
    def filter_afspraken_relevant_to_application(cls, applications_afspr):
        """Filter afspraken that are relevant to an application.

        Parameters
        ----------
        applications_afspr: applications dataframe with afspraken data merged to it

        Returns
        -------
        :
            filtered applications_afspr dataframe
        """
        result = applications_afspr.copy()
        ref_period_days = 365

        cond1 = date_in_ref_period(
            result,
            ref_col="first_dtopvoer",
            check_col="datum_invoer_afspr",
            period_days=ref_period_days,
        )
        cond2 = date_in_ref_period(
            result,
            ref_col="first_dtopvoer",
            check_col="datum_afhandeling_gepland_afspr",
            period_days=ref_period_days,
        )
        cond3 = (
            result["datum_invoer_afspr"].isna()
            & result["datum_afhandeling_gepland_afspr"].isna()
        )
        return result[cond1 | cond2 | cond3]

    @classmethod
    def calc_features(cls, applications_afspr):
        """Calculate afspraken features:
        - afspraken_count_last_year: Number of afspraken (any kind)
        - afpsraken_no_show_count_last_year: Number of times applicant didn't show for an afspraak
        - afpsraken_no_contact_count_last_year: Number of times no contact could be established

        Parameters
        ----------
        applications_afspr: applications dataframe with afspraken data merged to it

        Returns
        -------
        :
            dataframe of features
        """
        applications_afspr["no_show_bool"] = cls._afspraakresultaat_no_show(
            applications_afspr["afspraakresultaat_afspr"]
        )
        applications_afspr["no_contact_bool"] = cls._afspraakresultaat_no_contact(
            applications_afspr["afspraakresultaat_afspr"]
        )
        applications_afspr["attitude_bool"] = cls._afspraakresultaat_attitude(
            applications_afspr["afspraakresultaat_afspr"]
        )

        features = applications_afspr.groupby("application_dienstnr").agg(
            afspraken_no_show_count_last_year=pd.NamedAgg(
                column="no_show_bool", aggfunc="sum"
            ),
            afspraken_no_contact_count_last_year=pd.NamedAgg(
                column="no_contact_bool", aggfunc="sum"
            ),
            afspraken_geen_recht_houding_count_last_year=pd.NamedAgg(
                column="attitude_bool", aggfunc="sum"
            ),
        )

        features["afspraken_geen_recht_houding_last_year"] = (
            features["afspraken_geen_recht_houding_count_last_year"] > 0
        )
        return features

    @staticmethod
    def _afspraakresultaat_no_show(s: pd.Series):
        # Filter out no shows with notification ("met bericht"), because that's
        # very different from no shows without notifying.
        return (
            s.str.contains("no show", regex=False)
            & ~s.str.contains("met bericht", regex=False)
        ) | s.str.contains("niet verschenen", regex=False)

    @staticmethod
    def _afspraakresultaat_no_contact(s: pd.Series):
        return (
            s.str.contains("geen contact", regex=False)
            | s.str.contains("niet bereikt", regex=False)
            | s.str.contains("geen gegevens geleverd", regex=False)
        )

    @staticmethod
    def _afspraakresultaat_attitude(s: pd.Series):
        return s.str.contains(
            "geen recht op bijstand ivm houding & gedrag", regex=False
        )
