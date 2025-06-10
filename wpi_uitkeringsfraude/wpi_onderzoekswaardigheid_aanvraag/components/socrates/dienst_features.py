import datetime as dt
import logging
from typing import List, Tuple

import numpy as np
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
    SocratesRelevantDateFilter,
    SocratesRelevantPeriodFilter,
)
from wpi_onderzoekswaardigheid_aanvraag.settings.settings import WPISettings

logger = logging.getLogger(__name__)


class SocratesDienstFeatures(NoFittingRequiredMixin, Component):

    # These productnrs are all related somehow to Tozo, a specific type of welfare
    # given out only during the corona period.
    tozo_productnrs = (
        94,
        98,
        99,
        100,
        236,
        237,
        259,
        260,
        276,
        490,
        491,
        492,
        497,
        1045,
    )

    # Product numbers that have 'vordering' in the description, are not Tozo-related,
    # and are because of fraud, i.e. 'verwijtbaar'.
    fraud_vordering_productnrs = (
        2431,
        2432,
        2433,
        2434,
        2454,
        2540,
    )

    # Product numbers that have 'boete' in the description.
    boete_productnrs = (553, 555)

    def _transform(  # type: ignore
        self,
        scoring: bool,
        applications: pd.DataFrame,
        dienst: pd.DataFrame,
        dienstsubject_partij: pd.DataFrame,
        *args,
        **kwargs,
    ):
        applications_diensten_raw = self.join_applications_with_diensten(
            applications, dienst, dienstsubject_partij
        )
        result, dienst_history = self.add_features(
            applications, applications_diensten_raw
        )
        return result, dienst_history

    @classmethod
    def join_applications_with_diensten(
        cls,
        applications: pd.DataFrame,
        dienst: pd.DataFrame,
        dienstsubject_partij: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge proces data with dienst data.

        Parameters
        ----------
        applications: dienst dataframe processed to contain applications
        dienst: raw dienst dataframe
        dienstsubject_partij: dienstsubject dataframe with partij data merged to it

        Returns
        -------
        :
            applications and diensten joined
        """
        dienst = dienst.rename(columns={"subjectnrklant": "subjectnr"})
        all_diensten = cls._get_all_diensten(dienst, dienstsubject_partij)

        # Merge all diensten to the applications.
        applications_diensten_raw = applications.merge(
            all_diensten.add_suffix("_dienst"),
            how="inner",
            left_on="subjectnr",
            right_on="subjectnr_dienst",
        )

        return applications_diensten_raw

    @classmethod
    def _get_all_diensten(cls, dienst, dienstsubject_partij):
        """Combine diensten from tables Dienst and Dienstsubject to get all
        diensten belonging to a subject or their partij.

        Parameters
        ----------
        dienst: dienst dataframe
        dienstsubject_partij: dienstsubject dataframe with partij data merged to it

        Returns
        -------
        result:
            dataframe of diensten by subject
        """
        cols_needed = [
            "subjectnr",
            "dienstnr",
            "srtdienst",
            "besluit",
            "dtaanvraag",
            "dtbegindienst",
            "dteindedienst",
            "dtopvoer",
            "dtafvoer",
            "productnr",
            "geldig",
        ]

        # Table Dienst contains only hoofdklanten. If the subject was the
        # hoofdklant, we should take the dienst information from Dienst. If the
        # subject was not the hoofdklant, take it from Dienstsubject.
        dienstsubject_rel = cls._filter_dienstsubject_not_hoofdklant(
            dienstsubject_partij, dienst
        )

        # dtaanvraag and productnr is only in Dienst, so we have to add it
        # manually to Dienstsubject.
        dienst_info = cls._get_dienst_info_for_dienstsubject(dienstsubject_rel, dienst)
        dienstsubject_rel = dienstsubject_rel.merge(
            dienst_info, how="left", on="dienstnr"
        )

        all_diensten = (
            pd.concat(
                [
                    dienst[cols_needed],
                    dienstsubject_rel[cols_needed],
                ],
                # Add a boolean indicator whether subject is the hoofdklant.
                keys=[True, False],
            )
            .reset_index(level=[0])
            .rename(columns={"level_0": "subject_is_hoofdklant"})
        )

        return all_diensten

    @classmethod
    def _filter_dienstsubject_not_hoofdklant(cls, dienstsubject_partij, dienst):
        """Filter dienstsubject dataframe on rows where the subject is not the
        hoofdklant of the dienst. This is the case if the combination of dienstnr
        and subjectnr also appears in the dienst dataframe, because that contains
        only hoofdklanten with their diensten.

        Parameters
        ----------
        dienstsubject_partij: dienstsubject dataframe with partij data merged to it
        dienst: dienst dataframe

        Returns
        -------
        :
            filtered dienstsubject_partij
        """
        dienst["subject-dienst"] = cls._combine_subject_dienstnr(dienst)
        dienstsubject_partij["subject-dienst"] = cls._combine_subject_dienstnr(
            dienstsubject_partij
        )

        # If the subject-dienst combination is in table Dienst, then subject
        # is the hoofdklant for this dienst.
        dienstsubject_partij["subject_is_hoofdklant"] = dienstsubject_partij[
            "subject-dienst"
        ].isin(dienst["subject-dienst"])

        return dienstsubject_partij[~dienstsubject_partij["subject_is_hoofdklant"]]

    @classmethod
    def _combine_subject_dienstnr(cls, df):
        return df["subjectnr"].astype(str) + "-" + df["dienstnr"].astype(str)

    @classmethod
    def _get_dienst_info_for_dienstsubject(cls, dienstsubject, dienst):
        """For all dienstnr in dienstsubject, return the most recent information
        from dienst.

        Parameters
        ----------
        dienstsubject: dienstsubject dataframe
        dienst: dienst dataframe

        Returns
        -------
        :
            information per dienstnr
        """
        dienstsubject_present_in_dienst = set(
            dienstsubject["dienstnr"].unique()
        ).intersection(dienst["dienstnr"].unique())
        if len(dienstsubject_present_in_dienst) != dienstsubject["dienstnr"].nunique():
            logger.warning(
                f"Not all required dienstnr from dienstsubject could be found in the table dienst, "
                f"the following dienstnr are missing:"
                f" {set(dienstsubject['dienstnr'].unique()).difference(dienstsubject_present_in_dienst)}"
            )

        result = (
            dienst.set_index("dienstnr")
            .loc[list(dienstsubject_present_in_dienst)]
            .sort_values("dtopvoer", ascending=True)
            .reset_index()
            .drop_duplicates(
                "dienstnr",
                keep="last",
            )
        )
        return result[
            [
                "dienstnr",
                "productnr",
                "srtdienst",
                "besluit",
                "dtaanvraag",
            ]
        ]

    @classmethod
    def _filter_out_tozo_productnr(cls, diensten: pd.DataFrame) -> pd.DataFrame:
        """Filter out productnr that are related to Tozo, a type of welfare only
        given out during the corona period. Since Tozo will not return in the future
        (at least not permanently), it doesn't make sense to create features on it.
        """
        return diensten[~diensten["productnr_dienst"].isin(cls.tozo_productnrs)]

    def add_features(
        self,
        applications: pd.DataFrame,
        applications_diensten: pd.DataFrame,
    ):
        """Add dienst features to applications.

        Parameters
        ----------
        applications: dienst dataframe processed to contain applications
        applications_diensten: applications and diensten joined

        Returns
        -------
        result:
            applications dataframe with dienst features
        dienst_history:
            dataframe with history of relevant diensten per application
        """
        applications_diensten[
            "productnr_equals_application"
        ] = self.productnr_equals_application(applications_diensten)

        core_productnr = WPISettings.get_settings()["model"]["core_product_numbers"]
        current_dienst_features = self.calc_current_dienst_features(
            applications_diensten=applications_diensten,
            core_productnr=core_productnr,
        )
        past_dienst_features, dienst_history = self.calc_past_dienst_features(
            applications_diensten=applications_diensten,
            core_productnr=core_productnr,
        )
        past_application_features = self.calc_past_application_features(
            applications_diensten=applications_diensten,
            core_productnr=core_productnr,
        )

        result = (
            applications.merge(
                current_dienst_features, how="left", on="application_dienstnr"
            )
            .merge(past_dienst_features, how="left", on="application_dienstnr")
            .merge(past_application_features, how="left", on="application_dienstnr")
        )

        result = replace_nan_with_zero(
            result,
            columns=[
                x for x in result.columns if (("productnr_" in x) and ("_count" in x))
            ]
            + [
                "has_active_dienst",
                "received_same_product_last_year",
                "application_count_last_year",
                "application_rejected_count_last_year",
                "application_accepted_count_last_year",
                "applied_for_same_product_last_year",
                "rejected_for_same_product_last_year",
                "accepted_for_same_product_last_year",
                "had_boete_last_year",
                "had_vordering_last_year",
            ],
        )
        # A missing value for `days_since_last_dienst_end` implies not having a dienst
        # within the reference period, which for this feature equals having a dienst
        # a long time ago. Therefore, impute with a high value. Same logic for
        # `days_since_last_rejected_application` and `days_since_last_accepted_application`.
        result = result.fillna(
            {
                "days_since_last_dienst_end": 99999,
                "days_since_last_rejected_application": 99999,
                "days_since_last_accepted_application": 99999,
            }
        )
        result.name = "applications_diensten"
        return result, dienst_history

    @classmethod
    def productnr_equals_application(
        cls, applications_diensten: pd.DataFrame
    ) -> np.ndarray:
        """For every combination of application and related dienst check if the
        product numbers are equal.

        Parameters
        ----------
        applications_diensten: applications and diensten joined

        Returns
        -------
        :
            boolean series
        """
        return (
            applications_diensten["application_productnr"]
            == applications_diensten["productnr_dienst"]
        )

    @classmethod
    def calc_current_dienst_features(
        cls, applications_diensten: pd.DataFrame, core_productnr: List[int]
    ) -> np.ndarray:
        """Calculate features on the current diensten, i.e. the set of
        diensten the klant had at the time of the current application.

        Parameters
        ----------
        applications_diensten: applications and diensten joined
        core_productnr: list of the product numbers the model will be used for,
            will be used to create matching features

        Returns
        -------
        :
            dataframe with dienst features per application
        """
        applications_diensten = cls.filter_current_diensten_relevant_to_application(
            applications_diensten
        )

        applications_diensten = applications_diensten.loc[
            applications_diensten["productnr_dienst"].isin(core_productnr)
        ]

        # After the filtering on current diensten all the dienstnr left in the
        # df have at least one active dienst.
        applications_diensten["has_active_dienst"] = 1
        result = applications_diensten.drop_duplicates(
            "application_dienstnr", keep="last"
        )
        return result[["application_dienstnr", "has_active_dienst"]]

    @classmethod
    def filter_current_diensten_relevant_to_application(
        cls, applications_diensten: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter current diensten, i.e. the set of diensten the klant had at
        the time of the application.

        Parameters
        ----------
        applications_diensten: applications and diensten joined

        Returns
        -------
        :
            filtered applications_diensten dataframe
        """
        filterer = SocratesRelevantDateFilter(
            ref_date_col="first_dtopvoer",
            begindatum_col="dtbegindienst_dienst",
            einddatum_col="dteindedienst_dienst",
            opvoer_col="dtopvoer_dienst",
            afvoer_col="dtafvoer_dienst",
        )
        return filterer.fit_transform(applications_diensten)

    def calc_past_dienst_features(
        self, applications_diensten: pd.DataFrame, core_productnr: List[int]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate features on the set of diensten the klant has had in the
        past.

        Parameters
        ----------
        applications_diensten: applications and diensten joined
        core_productnr: list of the product numbers the model will be used for,
            will be used to create matching features

        Returns
        -------
        features:
            dataframe with dienst features per application
        dienst_history:
            dataframe with history of relevant diensten per application
        """
        dienst_history = applications_diensten.pipe(
            self.filter_past_diensten_relevant_to_application
        ).pipe(
            self._filter_accepted_diensten,
        )

        core_dienst_history = dienst_history.loc[
            dienst_history["productnr_dienst"].isin(core_productnr)
        ]

        productnr_counts_last_year = self.count_productnr(
            core_dienst_history, core_productnr
        )
        boete_vordering_features = self.create_boete_vordering_features(dienst_history)
        received_same_product_last_year = self.received_same_product_last_year(
            dienst_history
        )
        days_since_last_dienst_end = self.calc_days_since_last_dienst_end(
            core_dienst_history
        )
        features = (
            productnr_counts_last_year.merge(
                boete_vordering_features, how="outer", on="application_dienstnr"
            )
            .merge(
                received_same_product_last_year,
                how="outer",
                on="application_dienstnr",
            )
            .merge(
                days_since_last_dienst_end,
                how="outer",
                on="application_dienstnr",
            )
        )
        features["application_dienstnr"] = features["application_dienstnr"].astype(str)
        return features, dienst_history

    @classmethod
    def filter_past_diensten_relevant_to_application(
        cls, applications_diensten: pd.DataFrame
    ) -> pd.DataFrame:
        """Filter on the diensten the klant had in the past, before the proces.

        Parameters
        ----------
        applications_diensten: applications and diensten joined

        Returns
        -------
        :
            filtered applications_diensten dataframe
        """
        filterer = SocratesRelevantPeriodFilter(
            ref_date_col="first_dtopvoer",
            begindatum_col="dtbegindienst_dienst",
            einddatum_col="dteindedienst_dienst",
            opvoer_col="dtopvoer_dienst",
            afvoer_col="dtafvoer_dienst",
            period=dt.timedelta(days=365),
        )
        return filterer.fit_transform(applications_diensten)

    def count_productnr(
        self, applications_diensten: pd.DataFrame, core_productnr: List[int]
    ) -> pd.DataFrame:
        # Drop duplicates (per application) to prevent double counts
        unique_diensten = applications_diensten.sort_values(
            "dtopvoer_dienst"
        ).drop_duplicates(["application_dienstnr", "dienstnr_dienst"], keep="last")

        features = (
            pd.crosstab(
                unique_diensten["application_dienstnr"],
                unique_diensten["productnr_dienst"].apply(int),
            )
            .add_prefix("productnr_")
            .add_suffix("_count")
        )
        features = self._add_missing_productnr_counts(features, core_productnr)
        features = features.reset_index()
        features["application_dienstnr"] = features["application_dienstnr"].astype(str)
        return features

    @staticmethod
    def _add_missing_productnr_counts(
        productnr_counts: pd.DataFrame, core_productnr: List[int]
    ) -> pd.DataFrame:
        """Make sure that all productnr count columns are in the df."""
        cols_required = set([f"productnr_{int(x)}_count" for x in core_productnr])
        cols_missing = cols_required - set(productnr_counts.columns)
        productnr_counts = productnr_counts.assign(**dict.fromkeys(cols_missing, 0))
        return productnr_counts[list(cols_required)]

    @classmethod
    def create_boete_vordering_features(
        cls, applications_diensten: pd.DataFrame
    ) -> pd.DataFrame:
        """Create boolean indicators if someone had a vordering or boete.

        Parameters
        ----------
        applications_diensten: applications and diensten joined

        Returns
        -------
        :
            dataframe with two new features
        """
        applications_diensten["is_fraud_vordering"] = applications_diensten[
            "productnr_dienst"
        ].isin(cls.fraud_vordering_productnrs)
        applications_diensten["is_boete"] = applications_diensten[
            "productnr_dienst"
        ].isin(cls.boete_productnrs)

        result = applications_diensten.groupby("application_dienstnr").agg(
            had_vordering_last_year=pd.NamedAgg(
                column="is_fraud_vordering", aggfunc=any
            ),
            had_boete_last_year=pd.NamedAgg(column="is_boete", aggfunc=any),
        )
        return result

    @classmethod
    def received_same_product_last_year(
        cls, applications_diensten: pd.DataFrame
    ) -> pd.DataFrame:
        """Check if, in the year prior to their application, klant received
        the product they're now applying for.

        Parameters
        ----------
        applications_diensten: applications and diensten joined

        Returns
        -------
        :
            boolean indicator per application
        """
        feature = at_least_one_true(
            df=applications_diensten,
            col_to_group_by="application_dienstnr",
            col_to_sum="productnr_equals_application",
            new_column_name="received_same_product_last_year",
        )
        feature = feature.reset_index()
        feature["received_same_product_last_year"] = feature[
            "received_same_product_last_year"
        ].astype(int)
        feature["application_dienstnr"] = feature["application_dienstnr"].astype(str)
        return feature

    @classmethod
    def calc_days_since_last_dienst_end(
        cls, applications_diensten: pd.DataFrame
    ) -> pd.DataFrame:
        """Calculate the time in days between the application and when the last
        dienst ended. If klant still has an active dienst, return zero.

        Parameters
        ----------
        applications_diensten: applications and diensten joined

        Returns
        -------
        :
            feature value per application
        """

        relevant_dienst = cls._filter_accepted_diensten(applications_diensten)

        last_dienst = relevant_dienst.sort_values(
            "dteindedienst_dienst", ascending=True
        ).drop_duplicates("application_dienstnr", keep="last")
        last_dienst["days_since_last_dienst_end"] = (
            last_dienst["first_dtopvoer"] - last_dienst["dteindedienst_dienst"]
        ).dt.days

        # If first_dtopvoer (of the dienst they're applying for) < dteindedienst
        # (of the dienst that last ended), then they still have an active
        # dienst at the time of scoring. In this case, replace by zero.
        # When dteindedienst is NaT that also means no end date, so correct
        # those also.
        last_dienst.loc[
            (last_dienst["days_since_last_dienst_end"] < 0)
            | last_dienst["dteindedienst_dienst"].isna(),
            "days_since_last_dienst_end",
        ] = 0
        result = last_dienst[["application_dienstnr", "days_since_last_dienst_end"]]
        return result

    @classmethod
    def calc_past_application_features(
        cls, applications_diensten: pd.DataFrame, core_productnr: List[int]
    ) -> pd.DataFrame:
        """Calculate features on past applications done by klant.

        Parameters
        ----------
        applications_diensten: applications and diensten joined
        core_productnr: list of the product numbers the model will be used for,
            will be used to create matching features

        Returns
        -------
        :
            dataframe with past application features per application
        """
        applications = cls.filter_past_application_relevant_to_application(
            applications_diensten
        )
        applications = applications.loc[
            applications["productnr_dienst"].isin(core_productnr)
        ]
        application_counts_last_year = cls.count_applications(applications)
        days_since_last_application = cls.calc_days_since_last_application(applications)

        features = application_counts_last_year.merge(
            days_since_last_application, how="outer", on="application_dienstnr"
        )
        return features

    @classmethod
    def filter_past_application_relevant_to_application(cls, applications_diensten):
        """Filter on the applications the klant made before this one.

        Parameters
        ----------
        applications_diensten: applications and diensten joined

        Returns
        -------
        :
            filtered dataframe of relevant applications
        """
        filterer = SocratesRelevantPeriodFilter(
            ref_date_col="first_dtopvoer",
            begindatum_col="dtaanvraag_dienst",
            einddatum_col="dtaanvraag_dienst",
            opvoer_col="dtopvoer_dienst",
            afvoer_col="dtafvoer_dienst",
            period=dt.timedelta(days=365),
        )
        applications = filterer.fit_transform(applications_diensten)

        # Assume that if dtaanvraag is NaN, it was not an application.
        applications = applications.dropna(subset=["dtaanvraag_dienst"])

        return applications

    @classmethod
    def count_applications(cls, applications: pd.DataFrame) -> pd.DataFrame:
        """Per application, determine the following features based on applications
        made by klant in the past year:
        - the number of applications
        - the number of rejected applications
        - the number of accepted applications
        - boolean indicator whether klant applied for the same product
            they're now applying for
        - boolean indicator whether klant was accepted for the same product
            they're now applying for
        - boolean indicator whether klant was rejected for the same product
            they're now applying for

        Parameters
        ----------
        applications: recent applications of klant with application data merged to it

        Returns
        -------
        :
            feature values per application
        """
        overall_features = cls._calc_overall_application_features(applications)

        rej_applications = cls._filter_rejected_diensten(applications)
        rejected_features = cls._calc_rejected_application_features(rej_applications)

        acc_applications = cls._filter_accepted_diensten(applications)
        accepted_features = cls._calc_accepted_application_features(acc_applications)

        features = overall_features.merge(
            rejected_features, how="left", on="application_dienstnr"
        ).merge(accepted_features, how="left", on="application_dienstnr")

        return features

    @classmethod
    def _calc_overall_application_features(cls, all_applications):
        features = all_applications.groupby("application_dienstnr").agg(
            application_count_last_year=pd.NamedAgg(
                column="dienstnr_dienst", aggfunc=pd.Series.nunique
            ),
        )
        features = features.join(
            at_least_one_true(
                df=all_applications,
                col_to_group_by="application_dienstnr",
                col_to_sum="productnr_equals_application",
                new_column_name="applied_for_same_product_last_year",
            )
        )
        features["applied_for_same_product_last_year"] = features[
            "applied_for_same_product_last_year"
        ].astype(int)
        return features

    @classmethod
    def _calc_rejected_application_features(cls, rej_applications):
        features = rej_applications.groupby("application_dienstnr").agg(
            application_rejected_count_last_year=pd.NamedAgg(
                column="dienstnr_dienst", aggfunc=pd.Series.nunique
            ),
        )
        features = features.join(
            at_least_one_true(
                df=rej_applications,
                col_to_group_by="application_dienstnr",
                col_to_sum="productnr_equals_application",
                new_column_name="rejected_for_same_product_last_year",
            )
        )
        features[
            [
                "rejected_for_same_product_last_year",
                "application_rejected_count_last_year",
            ]
        ] = features[
            [
                "rejected_for_same_product_last_year",
                "application_rejected_count_last_year",
            ]
        ].astype(
            int
        )
        return features

    @classmethod
    def _calc_accepted_application_features(cls, acc_applications):
        features = acc_applications.groupby("application_dienstnr").agg(
            application_accepted_count_last_year=pd.NamedAgg(
                column="dienstnr_dienst", aggfunc=pd.Series.nunique
            ),
        )
        features = features.join(
            at_least_one_true(
                df=acc_applications,
                col_to_group_by="application_dienstnr",
                col_to_sum="productnr_equals_application",
                new_column_name="accepted_for_same_product_last_year",
            )
        )
        features[
            [
                "accepted_for_same_product_last_year",
                "application_accepted_count_last_year",
            ]
        ] = features[
            [
                "accepted_for_same_product_last_year",
                "application_accepted_count_last_year",
            ]
        ].astype(
            int
        )
        return features

    @classmethod
    def calc_days_since_last_application(
        cls, applications: pd.DataFrame
    ) -> pd.DataFrame:
        """Count the number of days since the previous rejected and accepted
         application from klant.

        Parameters
        ----------
        applications: recent applications of klant with application data merged to it

        Returns
        -------
        :
            feature values per application
        """
        last_rejected_application = (
            cls._filter_rejected_diensten(applications)
            .sort_values("dtaanvraag_dienst", ascending=True)
            .drop_duplicates("application_dienstnr", keep="last")
        )

        last_accepted_application = (
            cls._filter_accepted_diensten(applications)
            .sort_values("dtaanvraag_dienst", ascending=True)
            .drop_duplicates("application_dienstnr", keep="last")
        )

        last_rejected_application["days_since_last_rejected_application"] = (
            last_rejected_application["first_dtopvoer"]
            - last_rejected_application["dtaanvraag_dienst"]
        ).dt.days
        last_accepted_application["days_since_last_accepted_application"] = (
            last_accepted_application["first_dtopvoer"]
            - last_accepted_application["dtaanvraag_dienst"]
        ).dt.days

        result = last_rejected_application[
            ["days_since_last_rejected_application", "application_dienstnr"]
        ].merge(
            last_accepted_application[
                ["days_since_last_accepted_application", "application_dienstnr"]
            ],
            how="outer",
            on="application_dienstnr",
        )

        return result

    @classmethod
    def _filter_accepted_diensten(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dienst dataframe on diensten that were accepted.

        The meaning of the codes for 'besluit' are:
        1 = Toekennen
        2 = Principe toekenning
        3 = Afwijzen
        13 = Intrekken
        """
        df = cls._get_last_entry_per_dienst(df)
        result = df[
            df["besluit_dienst"].isin([1, 2])
            & ~cls._rejected_begin_date_equals_end_date(df)
        ]
        return result

    @classmethod
    def _filter_rejected_diensten(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dienst dataframe on diensten that were rejected.

        The meaning of the codes for 'besluit' are:
        1 = Toekennen
        2 = Principe toekenning
        3 = Afwijzen
        13 = Intrekken
        """
        df = cls._get_last_entry_per_dienst(df)
        return df[
            df["besluit_dienst"] == 3 | cls._rejected_begin_date_equals_end_date(df)
        ]

    @classmethod
    def _get_last_entry_per_dienst(cls, application_diensten):
        """For every combination of an application and a related dienst, keep only
        the last record of the related dienst so that we can use that to determine
        if it was rejected or accepted.
        """
        return application_diensten.sort_values(
            "dtopvoer_dienst", ascending=True
        ).drop_duplicates(
            subset=["application_dienstnr", "dienstnr_dienst"], keep="last"
        )

    @classmethod
    def _rejected_begin_date_equals_end_date(cls, df):
        """Filter out diensten that were rejected with additional logic besides
        'besluit'. Required because of flaky data.

        Some diensten look like they were accepted, because they have besluit = 1,
        but their begin and end date are the same, indicating that in fact, no
        payments were made.

        This only holds for product numbers that are paid out over a period, so
        *not* for bijzondere bijstand, which are one-off payments. These will always
        have the same begin and end date.
        """
        # Note that these are likely not all for which this is relevant, but
        # they're the main ones.
        relevant_productnr = [
            131,  # WWb/LO
            135,  # WWb/EV
            227,  # Krediethypotheek
        ]
        return (df["dtbegindienst_dienst"] == df["dteindedienst_dienst"]) & df[
            "productnr_dienst"
        ].isin(relevant_productnr)
