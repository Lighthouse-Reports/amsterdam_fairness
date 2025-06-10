import datetime as dt

import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.common import (
    at_least_one_true,
    replace_nan_with_zero,
)
from wpi_onderzoekswaardigheid_aanvraag.components import Component
from wpi_onderzoekswaardigheid_aanvraag.components.component import (
    NoFittingRequiredMixin,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.rows_by_date_filter import (
    SocratesRelevantPeriodFilter,
)


class SherlockProcesFeatures(NoFittingRequiredMixin, Component):
    def _transform(  # type: ignore
        self,
        scoring: bool,
        applications: pd.DataFrame,
        proces_klant: pd.DataFrame,
        *args,
        **kwargs,
    ):
        applications_proces = self.join_proces_history(applications, proces_klant)
        result = self.add_features(applications, applications_proces)
        return result

    @classmethod
    def join_proces_history(
        cls,
        applications: pd.DataFrame,
        proces_klant: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge applications data with all process data.

        Parameters
        ----------
        applications: applications dataframe
        proces_klant: proces dataframe with sherlock klant data joined to it

        Returns
        -------
        :
            applications and proces_klant dataframe joined
        """
        applications_proces = applications.merge(
            proces_klant.add_suffix("_hist"),
            how="inner",
            left_on="subjectnr",
            right_on="kln_adminnummer_hist",
        )
        return applications_proces

    @classmethod
    def add_features(
        cls, applications: pd.DataFrame, applications_proces: pd.DataFrame
    ):
        """Add past process features to applications.

        Parameters
        ----------
        applications: applications dataframe
        applications_proces: proces base dataset with all past processen merged to it

        Returns
        -------
        result:
            applications dataframe with past process features
        """
        proces_hist_filtered = cls.filter_past_proces_relevant_to_application(
            applications_proces
        )
        past_proces_features = cls.calc_past_proces_features(proces_hist_filtered)
        days_since_last_proces = cls.calc_days_since_last_proces(proces_hist_filtered)

        result = applications.merge(
            past_proces_features, how="left", on="application_dienstnr"
        ).merge(days_since_last_proces, how="left", on="application_dienstnr")

        result = replace_nan_with_zero(
            result,
            columns=[
                "proces_count_last_year",
                "had_beheer_proces_last_year",
                "had_proces_last_year",
            ],
        )
        # A missing value for `days_since_last_proces` implies not having a proces
        # within the reference period, which in our application equals having a proces
        # a long time ago. Therefore, impute with a high value.
        result = result.fillna({"days_since_last_proces": 99999})
        result.name = "applications"
        return result

    @classmethod
    def filter_past_proces_relevant_to_application(cls, applications_proces):
        """Filter past processes that are relevant to an application.

        Parameters
        ----------
        applications_proces: applications dataframe with all past processen merged to it

        Returns
        -------
        :
            filtered applications_proces dataframe
        """
        result = applications_proces[
            applications_proces["spr_id_hist"].isin(
                [
                    146,  # C Aanvraag
                    147,  # C RMO
                    149,  # C RMO (CONA)
                    170,  # O Fraudeonderzoek
                    213,  # C Pro-actief
                    231,  # C VPO
                ]
            )
        ]

        filterer = SocratesRelevantPeriodFilter(
            ref_date_col="first_dtopvoer",
            opvoer_col="pro_startdatum_hist",
            afvoer_col="pro_einddatum_hist",
            period=dt.timedelta(days=365),
        )
        return filterer.fit_transform(result)

    @classmethod
    def calc_past_proces_features(
        cls, applications_proces: pd.DataFrame
    ) -> pd.DataFrame:
        applications_proces["beheerproces_bool"] = (
            applications_proces["spr_id_hist"] != 146
        )
        features = applications_proces.groupby("application_dienstnr").agg(
            proces_count_last_year=pd.NamedAgg(column="pro_id_hist", aggfunc="size"),
        )
        features = features.join(
            at_least_one_true(
                df=applications_proces,
                col_to_group_by="application_dienstnr",
                col_to_sum="beheerproces_bool",
                new_column_name="had_beheer_proces_last_year",
            )
        )
        features["had_proces_last_year"] = features["proces_count_last_year"] > 0
        features["had_proces_last_year"] = features["had_proces_last_year"].astype(int)
        features["had_beheer_proces_last_year"] = features[
            "had_beheer_proces_last_year"
        ].astype(int)
        return features

    @classmethod
    def calc_days_since_last_proces(
        cls, applications_proces: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate the time in days between the application and when the last
        dienst ended. If klant still has an active dienst, return zero.

        Parameters
        ----------
        applications_proces: applications dataframe with all past processen merged to it

        Returns
        -------
        :
            feature value per application
        """
        last_proces = applications_proces.sort_values(
            "pro_einddatum_hist", ascending=True
        ).drop_duplicates("application_dienstnr", keep="last")
        last_proces["days_since_last_proces"] = (
            last_proces["first_dtopvoer"] - last_proces["pro_einddatum_hist"]
        ).dt.days

        # If first_dtopvoer < pro_einddatum_hist, then they still have an
        # active proces at the time of scoring. In this case, replace by
        # zero. When pro_einddatum_hist is NaT that also means no end date, so
        # correct those also.
        last_proces.loc[
            (last_proces["days_since_last_proces"] < 0)
            | last_proces["pro_einddatum_hist"].isna(),
            "days_since_last_proces",
        ] = 0
        result = last_proces[["application_dienstnr", "days_since_last_proces"]]
        return result
