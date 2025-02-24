import datetime as dt
import logging

import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.common import replace_nan_with_zero
from wpi_onderzoekswaardigheid_aanvraag.components import Component
from wpi_onderzoekswaardigheid_aanvraag.components.component import (
    NoFittingRequiredMixin,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.rows_by_date_filter import (
    SocratesRelevantDateFilter,
    SocratesRelevantPeriodFilter,
)

logger = logging.getLogger(__name__)


class SocratesAdresFeatures(NoFittingRequiredMixin, Component):
    # Adres soorten that will be counted; listed here so that missing
    # columns can be added if a certain soort is not in the df at all.
    adres_soort_tuple = (1, 2, 3, 9)

    def _transform(  # type: ignore
        self,
        scoring: bool,
        applications: pd.DataFrame,
        adres: pd.DataFrame,
        *args,
        **kwargs,
    ):
        applications_adres_raw = self.join_applications_adres(applications, adres)
        result = self.add_features(applications, applications_adres_raw)
        return result

    @classmethod
    def join_applications_adres(
        cls, applications: pd.DataFrame, adres: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge applications data with adres data.

        Parameters
        ----------
        applications: dienst dataframe processed to contain applications
        adres: adres dataframe

        Returns
        -------
        :
            applications and adres dataframe joined
        """
        applications_adres_raw = pd.merge(
            applications,
            adres.add_suffix("_adres"),
            how="inner",
            left_on="subjectnr",
            right_on="subjectnr_adres",
        )
        return applications_adres_raw

    @classmethod
    def add_features(
        cls, applications: pd.DataFrame, applications_adres: pd.DataFrame
    ) -> pd.DataFrame:
        """Add adres features to applications.

        Parameters
        ----------
        applications: dienst dataframe processed to contain applications
        applications_adres: applications dataframe with adres data merged to it

        Returns
        -------
        :
            applications dataframe with adres features
        """
        recent_relocations = cls.recent_relocated_to_amsterdam(applications_adres)
        moves_per_subject = cls.calculate_last_year_relocations_per_subject(
            applications_adres
        )
        applications_adres = cls.filter_adres_relevant_to_application(
            applications_adres
        )

        applications["active_address_unknown"] = ~applications[
            "application_dienstnr"
        ].isin(applications_adres["application_dienstnr"])

        adres_features = cls.calculate_adres_features(applications_adres)

        result = (
            applications.merge(
                recent_relocations,
                how="left",
                left_on="application_dienstnr",
                right_index=True,
            )
            .merge(moves_per_subject, how="left", on="application_dienstnr")
            .merge(adres_features, how="left", on="application_dienstnr")
        )

        result = replace_nan_with_zero(
            result,
            columns=[
                "at_least_one_address_in_amsterdam",
                "active_address_count",
                "bijzondere_doelgroep_address",
                "less_than_30_days_since_last_relocation",
                "relocation_count_last_year",
                "relocated_to_ams_last_90d",
                "soort_adres_1",
                "soort_adres_2",
                "soort_adres_3",
                "soort_adres_9",
            ],
        )

        # If this feature is missing, then people either had only a post address
        # or none at all. Impute with a high value so that they get treated like
        # people who moved a very long time ago.
        result = result.fillna({"days_since_last_relocation": 99999})

        result.name = "applications_adres"
        return result

    @classmethod
    def calculate_last_year_relocations_per_subject(
        cls, applications_adres: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate how many times a person has relocated in the last year.

        Parameters
        ----------
        applications_adres: applications dataframe with adres data merged to it

        Returns
        -------
        :
            filtered applications_adres dataframe
        """
        filtered = SocratesRelevantPeriodFilter(
            ref_date_col="first_dtopvoer",
            begindatum_col="dtbegin_adres",
            einddatum_col="dteinde_adres",
            opvoer_col="dtopvoer_adres",
            afvoer_col="dtafvoer_adres",
            period=dt.timedelta(days=365),  # Look at relocations up to 1 year ago
        ).fit_transform(applications_adres)
        last_year_relocations = (
            filtered[["application_dienstnr"]]
            .value_counts()
            .rename_axis("application_dienstnr")
            .reset_index(name="relocation_count_last_year")
        )
        last_year_relocations["relocation_count_last_year"] -= 1
        return last_year_relocations

    @classmethod
    def recent_relocated_to_amsterdam(
        cls, applications_adres: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate if a person has recently relocated from outside amsterdam.

        Parameters
        ----------
        applications_adres: applications dataframe with adres data merged to it

        Returns
        -------
        :
            filtered applications_adres dataframe
        """
        filtered = applications_adres[
            (applications_adres["soort_adres"] == 1)
            | (applications_adres["soort_adres"] == 9)
        ]
        filtered = SocratesRelevantPeriodFilter(
            ref_date_col="first_dtopvoer",
            begindatum_col="dtbegin_adres",
            einddatum_col="dteinde_adres",
            opvoer_col="dtopvoer_adres",
            afvoer_col="dtafvoer_adres",
            period=dt.timedelta(days=90),  # Look at addresses active up to 3 months
        ).fit_transform(filtered)

        if not filtered.empty:
            feature = filtered.groupby(["application_dienstnr"])[
                [
                    "first_dtopvoer",
                    "plaats_adres",
                    "dtbegin_adres",
                    "dteinde_adres",
                    "dtopvoer_adres",
                    "dtafvoer_adres",
                ]
            ].apply(cls._relocated_to_amsterdam)
        else:
            feature = pd.Series(dtype="bool")
        feature.name = "relocated_to_ams_last_90d"
        return feature

    @classmethod
    def _relocated_to_amsterdam(cls, filtered: pd.DataFrame) -> bool:
        if (
            len(filtered) < 2
            and "amsterdam" in filtered.iloc[0]["plaats_adres"]
            and filtered.iloc[0]["dtbegin_adres"]
            >= filtered.iloc[0]["first_dtopvoer"] - pd.DateOffset(days=90)
        ):
            return True
        if len(filtered) < 2:
            return False
        filtered = filtered.sort_values(by="dtbegin_adres", ascending=False)
        if (
            "amsterdam" in filtered.iloc[0]["plaats_adres"]
            and "amsterdam" not in filtered.iloc[1]["plaats_adres"]
        ):
            return True
        return False

    @classmethod
    def filter_adres_relevant_to_application(
        cls, applications_adres: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter adres that are relevant to an application.

        Parameters
        ----------
        applications_adres: applications dataframe with adres data merged to it

        Returns
        -------
        :
            filtered applications_adres dataframe
        """
        filterer = SocratesRelevantDateFilter(
            ref_date_col="first_dtopvoer",
            begindatum_col="dtbegin_adres",
            einddatum_col="dteinde_adres",
            opvoer_col="dtopvoer_adres",
            afvoer_col="dtafvoer_adres",
        )
        return filterer.fit_transform(applications_adres)

    @classmethod
    def calculate_adres_features(cls, applications_adres: pd.DataFrame) -> pd.DataFrame:
        """Calculate adres features

        Parameters
        ----------
        applications_adres: applications dataframe with adres data merged to it

        Returns
        -------
        :
            applications dataframe with new features
        """
        soort_adres_features = (
            pd.crosstab(
                applications_adres["application_dienstnr"],
                applications_adres["soort_adres"].apply(int),
            )
            .add_prefix("soort_adres_")
            .reset_index()
        )
        soort_adres_features = cls._add_missing_soort_counts(soort_adres_features)

        applications_adres["unique_adres"] = (
            applications_adres["plaats_adres"]
            + applications_adres["postcodenum_adres"].astype(str)
            + applications_adres["straat_adres"]
            + applications_adres["huisnr_adres"].astype(str)
            + applications_adres["huisletter_adres"].astype(str)
            + applications_adres["huisnrtoev_adres"].astype(str)
        )

        features = applications_adres.groupby("application_dienstnr").agg(
            active_address_count=pd.NamedAgg(column="unique_adres", aggfunc="count"),
        )
        features = pd.merge(
            features,
            soort_adres_features,
            how="outer",
            left_index=True,
            right_on="application_dienstnr",
        )

        days_since_last_relocation = cls._calculate_days_since_last_relocation(
            applications_adres
        )
        features = features.reset_index(drop=True)
        features = features.merge(
            days_since_last_relocation,
            how="left",
            left_on="application_dienstnr",
            right_index=True,
        )

        features["less_than_30_days_since_last_relocation"] = (
            features["days_since_last_relocation"] < 30
        )

        at_least_one_address_in_amsterdam = cls._at_least_one_address_in_amsterdam(
            applications_adres
        )
        last_address_BD = cls._last_address_is_bijzondere_doelgroep(applications_adres)

        features = features.merge(
            at_least_one_address_in_amsterdam,
            how="left",
            left_on="application_dienstnr",
            right_index=True,
        ).merge(
            last_address_BD,
            how="left",
            on="application_dienstnr",
        )
        features["at_least_one_address_in_amsterdam"] = features[
            "at_least_one_address_in_amsterdam"
        ].astype(int)
        return features

    @staticmethod
    def _last_address_is_bijzondere_doelgroep(
        applications_adres: pd.DataFrame,
    ) -> pd.DataFrame:
        """Determine if the last address is an address where a bijzondere doelgroep' (BD) gets registered:
        a special target group such as homeless or addicted persons. These people get registered to a set of
        municipality or other help organizations' addresses if they don't have their own address.
        They are not in the scope of the model, as they go through a different process.

        Parameters
        ----------
        applications_adres: applications dataframe with adres data merged to it

        Returns
        -------
        :
            dataframe containing a boolean that maps if the last address is a municipality address
        """
        applications_adres = applications_adres.copy()
        features = applications_adres.loc[
            :,
            [
                "application_dienstnr",
                "dtbegin_adres",
                "straat_adres",
                "huisnr_adres",
            ],
        ]

        full_address = features.loc[:, "straat_adres"].astype(str) + features.loc[
            :, "huisnr_adres"
        ].astype(str).str.replace(".0", "", regex=True)

        addresses_to_check = [
            "490cc9ae805caf8ec8ceec9d214412a2f10ad6d99c8a0c136393840e01df01f2"
            + "323",  # Technically it's Jan van Galenstraat 323B, but since it's an office building,
            # no one should be living at the other letters either.
            "78466abd93a8fbb9649ebdd2d0d6b5d15dbaad789e3555f629ca8251968e6a2f"
            + "5",  # Ringdijkstraat 5
            "33b27f8438d4b7a23ddfe83ffac1acbac084e0acad9bf8ac931e97bab4afbb3d"
            + "328",  # Jacob van Arteveldestraat 328
            "7964cf41452a9029e29216e8894de0166e65bca642bc348019fe277876647e44"
            + "2",  # Oranje - Vrijstaatplein 2
            "d42549b6f54b3b4fdcc44fac833e07a3001234ee118603b0752dbc0ca78852f2"
            + "8",  # Roggeveenstraat 8
            "82912692a6c0e259f9d49b3d73a5829afbc618568cf9c1266a1a65ee2bc3c690"
            + "215",  # Zeeburgerdijk 215
            "6235f390c8b809a46462ec6f925b91b43b9d307e0f4deb37f4981c2a07cd6d14"
            + "227",  # Postbus 227
            "5dfd36ee59f387106cc3874db453b2f4b1d696f28ab8c0e050dfb8ebf7999905"
            + "2",  # Valckenierstraat 2
            "db1cff30ebf0a8f7cbfa10e0f9a22523d10c8e1522e7195308ef596465fdfd5d"
            + "23",  # Flierbosdreef 23
            "db1cff30ebf0a8f7cbfa10e0f9a22523d10c8e1522e7195308ef596465fdfd5d"
            + "2",  # Flierbosdreef 2
            "4985aa3437f7ecdf150dec7ff297ec55d1a4a66b39afaeb5f8efe40bc9c507cb"
            + "2",  # Elisabeth Wolffstraat 2
            "8d34273f33efd0cbeda111a9b102f70eb3e108e13466d22fb59f5e5871985d1b"
            + "64",  # Alexanderkade 64
            "b2a5711b42e5d4062424d3bde079f66b76944f5cd6439838293ce3a0dd78863b"
            + "59",  # Tollensstraat 59
        ]

        features.loc[:, "bijzondere_doelgroep_address"] = full_address.isin(
            addresses_to_check
        ).astype(int)
        features = features.sort_values("dtbegin_adres").drop_duplicates(
            "application_dienstnr", keep="last"
        )

        return features[["application_dienstnr", "bijzondere_doelgroep_address"]]

    @staticmethod
    def _calculate_days_since_last_relocation(
        applications_adres: pd.DataFrame,
    ) -> pd.DataFrame:
        filtered = applications_adres.loc[
            applications_adres["soort_adres"].isin([1, 3, 9]), :
        ].copy()
        filtered["days_since_last_relocation"] = (
            filtered["first_dtopvoer"] - filtered["dtbegin_adres"]
        ).dt.days
        filtered_2 = filtered.groupby("application_dienstnr").agg(
            days_since_last_relocation=pd.NamedAgg(
                column="days_since_last_relocation", aggfunc="min"
            )
        )
        return filtered_2

    @classmethod
    def _add_missing_soort_counts(cls, counts: pd.DataFrame) -> pd.DataFrame:
        """Make sure that all soort count columns are in the df."""
        cols_required = [f"soort_adres_{int(x)}" for x in cls.adres_soort_tuple]
        cols_missing = set(cols_required) - set(counts.columns)
        counts = counts.assign(**dict.fromkeys(cols_missing, 0))
        return counts[cols_required + ["application_dienstnr"]]

    @staticmethod
    def _at_least_one_address_in_amsterdam(applications_adres):
        applications_adres["in_amsterdam"] = applications_adres[
            "plaats_adres"
        ].str.contains("amsterdam")
        return applications_adres.groupby("application_dienstnr").agg(
            at_least_one_address_in_amsterdam=pd.NamedAgg(
                column="in_amsterdam", aggfunc="any"
            )
        )
