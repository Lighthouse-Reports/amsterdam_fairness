import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.common import (
    at_least_one_true,
    replace_nan_with_zero,
)
from wpi_onderzoekswaardigheid_aanvraag.components import Component
from wpi_onderzoekswaardigheid_aanvraag.components.component import (
    NoFittingRequiredMixin,
)


class RaakWerkervaringFeatures(NoFittingRequiredMixin, Component):
    def _transform(  # type: ignore
        self,
        scoring: bool,
        applications: pd.DataFrame,
        werkervaring: pd.DataFrame,
        *args,
        **kwargs,
    ):
        applications_werk_raw = self.join_applications_werk(applications, werkervaring)
        result = self.add_features(applications, applications_werk_raw)
        return result

    @classmethod
    def join_applications_werk(
        cls, applications: pd.DataFrame, werkervaring: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge dienst data with werkervaring data.

        Parameters
        ----------
        applications: applications dataframe
        werkervaring: werkervaring dataframe

        Returns
        -------
        :
            dienst and werkervaring dataframe joined
        """
        applications_werk_raw = pd.merge(
            applications[pd.notnull(applications["subjectnr"])],
            werkervaring.add_suffix("_werk"),
            how="inner",
            left_on="subjectnr",
            right_on="administratienummer_werk",
        )

        return applications_werk_raw

    def add_features(
        self,
        applications: pd.DataFrame,
        applications_werk: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add werkervaring features to applications.

        Parameters
        ----------
        applications: applications dataframe
        applications_werk: applications dataframe with werkervaring data merged to it

        Returns
        -------
        :
            applications dataframe with werkervaring features
        """
        applications_werk = self.filter_werk_relevant_to_application(applications_werk)
        features = self.calc_features(applications_werk)

        result = applications.merge(features, how="left", on="application_dienstnr")
        result = replace_nan_with_zero(
            result,
            columns=["occupation_risk_sector"],
        )

        result.name = "applications_werk"
        return result

    @classmethod
    def filter_werk_relevant_to_application(cls, applications_werk):
        """Filter werkervaring that are relevant to an application.

        Parameters
        ----------
        applications_werk: applications dataframe with werkervaring data merged to it

        Returns
        -------
        :
            filtered applications_werk dataframe
        """
        # TODO: Figure out if/how we can filter this dataframe on recency.
        return applications_werk

    @classmethod
    def calc_features(cls, applications_werk):
        """Calculate werkvaring features.

        Parameters
        ----------
        applications_werk: applications dataframe with werkervaring data merged to it

        Returns
        -------
        :
            dataframe of features
        """
        df = applications_werk.copy()

        df["risk_sector"] = cls._beroep_risicobranche(
            df["belangrijkste_werkervaring_werk"]
        )

        features = at_least_one_true(
            df=df,
            col_to_group_by="application_dienstnr",
            col_to_sum="risk_sector",
            new_column_name="occupation_risk_sector",
        )

        return features

    @staticmethod
    def _beroep_risicobranche(s: pd.Series) -> pd.Series:
        return (
            _occupation_bouw(s)
            | _occupation_horeca_kitchen(s)
            | _occupation_horeca_cook(s)
            | _occupation_other_risk_sector(s)
        )


def _occupation_bouw(series):
    exclude = series.str.contains(_lists_to_regex(_bouw_substrings_exclusions()))
    return series.str.contains("bouw") & ~exclude


def _bouw_substrings_exclusions():
    return [
        "boer",
        "landbouw",
        "gebouw",
        "bouwmarkt",
        "opbouw",
        "website",
        "bosbouw",
        "teken",
    ]


def _occupation_horeca_kitchen(series):
    exclude = series.str.contains(_lists_to_regex(_horeca_kitchen_exclusions()))
    return series.str.contains("keuken") & ~exclude


def _horeca_kitchen_exclusions():
    return ["verkoop", "monteur"]


def _occupation_horeca_bar_staff(series):
    return series.str.startswith("bar")


def _occupation_horeca_cook(series):
    return series.str.contains(r".*kok .*|\-kok.*|.*koken.*") | series.str.startswith(
        "kok"
    )


def _occupation_other_risk_sector(series):
    return series.str.contains(
        _lists_to_regex(
            _cleaning_substrings(),
            _horeca_substrings(),
            _construction_substrings(),
            _other_occupation_substrings(),
        )
    )


def _lists_to_regex(*lists):
    return ".*" + ".*|.*".join([item for sublist in lists for item in sublist]) + ".*"


def _cleaning_substrings():
    return [
        "schoonmaker",
        "schoonmaak",
        "kamermeisje",
        "huishoudelijke hulp",
        "huishouding",
        "huishouden",
    ]


def _horeca_substrings():
    return [
        "horeca",
        "chefkok",
        "restaurant",
        "serveer",
        "pizz",
        "cateraar",
        "catering",
        "ober",
        "kelner",
        "cafe",
        "café",
        "snackbar",
        "afwas",
    ]


def _construction_substrings():
    return [
        "klusjesman",
        "schilder",
        "behanger",
        "stukadoor",
        "metselaar",
        "stratenmaker",
        "straatmaker",
        "elektricien",
        "tegelzetter",
        "dakdekker",
        "glaszetter",
        "loodgieter",
        "timmerman",
    ]


def _other_occupation_substrings():
    return [
        "glazenwasser",
        "hovenier",
        "tuinman",
        "autohandelaar",
        "kapper",
        "haarstylist",
        "taxi",
        "particulier chauffeur",
    ]
