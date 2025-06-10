import datetime as dt
from functools import reduce

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


class SocratesRelatieFeatures(NoFittingRequiredMixin, Component):
    def _transform(  # type: ignore
        self,
        scoring: bool,
        applications: pd.DataFrame,
        relatie: pd.DataFrame,
        *args,
        **kwargs,
    ):
        applications_relatie_raw = self.join_applications_relatie(applications, relatie)
        result = self.add_features(applications, applications_relatie_raw)
        return result

    @classmethod
    def join_applications_relatie(
        cls, applications: pd.DataFrame, relatie: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge dienst data with relatie data.

        Parameters
        ----------
        applications: applications dataframe
        relatie: relatie dataframe

        Returns
        -------
        :
            dienst and relatie dataframe joined
        """
        applications_relatie_raw = pd.merge(
            applications[pd.notnull(applications["subjectnr"])],
            relatie.add_suffix("_relatie"),
            how="inner",
            left_on="subjectnr",
            right_on="subjectnr_relatie",
        )
        return applications_relatie_raw

    @classmethod
    def add_features(
        cls, applications: pd.DataFrame, applications_relatie: pd.DataFrame
    ) -> pd.DataFrame:
        """Add relatie features to applications.

        Parameters
        ----------
        applications: applications dataframe
        applications_relatie: applications dataframe with relatie data merged to it

        Returns
        -------
        :
            applications dataframe with relatie features
        """
        current_features = cls.calc_current_features(applications_relatie)
        last_90d_features = cls._calculate_partner_changes_last_90d(
            applications_relatie
        )

        result = applications.merge(
            current_features,
            how="left",
            left_on="application_dienstnr",
            right_index=True,
        ).merge(
            last_90d_features,
            how="left",
            left_on="application_dienstnr",
            right_index=True,
        )

        # People with no relaties at all were not in applications_relatie, so
        # will have NaN features, but these should then actually be zero.
        result = replace_nan_with_zero(
            result,
            columns=[
                "has_partner",
                "has_medebewoner",
                "has_algemene_relatie",
                "medebewoner_count",
                "algemene_relatie_count",
                "kostendeler_count",
                "lives_with_partner",
                "is_parttime_parent",
                "is_fulltime_parent",
                "changed_partners_last_90d",
                "separated_from_partner_last_90d",
            ],
        )
        result["lives_with_partner"] = result["lives_with_partner"].astype(int)
        result["is_parttime_parent"] = result["is_parttime_parent"].astype(int)
        result["is_fulltime_parent"] = result["is_fulltime_parent"].astype(int)
        result["changed_partners_last_90d"] = result[
            "changed_partners_last_90d"
        ].astype(int)
        result["separated_from_partner_last_90d"] = result[
            "separated_from_partner_last_90d"
        ].astype(int)
        result.name = "applications_relatie"
        return result

    @classmethod
    def filter_current_relatie_relevant_to_application(cls, applications_relatie):
        """Filter current relaties that are relevant to an application.

        Parameters
        ----------
        applications_relatie: applications dataframe with relatie data merged to it

        Returns
        -------
        :
            filtered applications_relatie dataframe
        """
        filterer = SocratesRelevantDateFilter(
            ref_date_col="first_dtopvoer",
            begindatum_col="dtbegin_relatie",
            einddatum_col="dteinde_relatie",
            opvoer_col="dtopvoer_relatie",
            afvoer_col="dtafvoer_relatie",
        )
        return filterer.fit_transform(applications_relatie)

    @classmethod
    def calc_current_features(cls, applications_relatie: pd.DataFrame) -> pd.DataFrame:
        """Calculate current relatie features for one application.

        Parameters
        ----------
        applications_relatie:

        Returns
        -------
        :
            tuple of new features
        """
        filtered_applications = cls.filter_current_relatie_relevant_to_application(
            applications_relatie
        )

        relatie_categorie_features = cls._calc_relatie_categorie_features(
            filtered_applications
        )
        kostendeler_count = cls._count_kostendelers(filtered_applications)
        lives_with_partner = cls._lives_with_partner(filtered_applications)
        parenting_features = cls._parent_type(filtered_applications)

        return reduce(
            lambda df1, df2: pd.merge(
                df1, df2, left_index=True, right_index=True, how="outer"
            ),
            [
                relatie_categorie_features,
                kostendeler_count,
                lives_with_partner,
                parenting_features,
            ],
        )

    @classmethod
    def _calc_relatie_categorie_features(
        cls, applications_relatie: pd.DataFrame
    ) -> pd.DataFrame:
        """Create features based on the category of relaties:
        1: partnerrelatie
        2: medebewoner
        3: algemene relatie
        4: kindrelatie

        For each category, this function counts the number of relaties and a
        creates a boolean indicator whether there is at least one relatie of
        that category. The resulting features are called '{category}_count'
        and 'has_{category}'.

        We ignore the child relaties, because this information is not approved
        to be used as features.

        Use nunique() on the subjectnrrelatie, rather than len(), in case some
        relaties are in the data more than once.

        Parameters
        ----------
        applications_relatie: applications dataframe with relatie data merged to it

        Returns
        -------
        :
            dataframe with new features
        """
        category_mapping = {
            1: "partner",
            2: "medebewoner",
            3: "algemene_relatie",
            # 4: "child",
        }

        filtered_df = applications_relatie[
            applications_relatie["categorie_relatie"] != 4
        ]

        if not filtered_df.empty:
            counts = pd.pivot_table(
                filtered_df,
                index=["application_dienstnr"],
                columns=["categorie_relatie"],
                values="subjectnrrelatie_relatie",
                aggfunc=pd.Series.nunique,
            )

        # If `applications_relatie` is empty, create empty columns manually, because `pivot_table` will fail.
        else:
            counts = pd.DataFrame(columns=category_mapping.keys())

        features = cls._add_required_relatie_category_columns(counts, category_mapping)
        features_bool = features > 0
        features_bool.columns = [
            f"has_{category}" for category in category_mapping.values()
        ]
        return features.merge(
            features_bool, left_index=True, right_index=True, how="outer"
        )

    @staticmethod
    def _add_required_relatie_category_columns(counts, category_mapping):
        # Map category numbers to meaningful feature names.
        mapped_category_names = counts.columns.map(category_mapping)
        counts.columns = [f"{category}_count" for category in mapped_category_names]

        # Make sure that any missing columns are present.
        cols_required = set([str(c) + "_count" for c in category_mapping.values()])
        cols_missing = cols_required - set(counts.columns)
        counts = counts.assign(**dict.fromkeys(cols_missing, 0))

        return counts

    @classmethod
    def _count_kostendelers(cls, applications_relatie: pd.DataFrame) -> pd.DataFrame:
        """Count how many kostendelers a subject has.

        Kostendelers have `soort` equal to 40. We must also check that
        `kduitsluiting` equals 0, else the kostendeler does not count.

        Parameters
        ----------
        applications_relatie: applications dataframe with relatie data merged to it

        Returns
        -------
        :
            dataframe with integer counts
        """
        rel_rows = applications_relatie[
            (applications_relatie["soort_relatie"] == 40)
            & (applications_relatie["kduitsluiting_relatie"] == 0)
        ]
        feature = rel_rows.groupby("application_dienstnr").agg(
            kostendeler_count=pd.NamedAgg(
                column="subjectnrrelatie_relatie", aggfunc="nunique"
            )
        )
        return feature

    @classmethod
    def _lives_with_partner(cls, applications_relatie: pd.DataFrame) -> pd.Series:
        """Create boolean indicator whether subject lives together with partner
        or not.

        Parameters
        ----------
        applications_relatie: applications dataframe with relatie data merged to it

        Returns
        -------
        :
            boolean series
        """
        rel_rows = applications_relatie[
            (applications_relatie["categorie_relatie"] == 1)
            & (applications_relatie["samenwonend_relatie"] == 1)
        ]
        feature = pd.Series(
            True,
            index=rel_rows["application_dienstnr"].unique(),
            name="lives_with_partner",
        )
        return feature

    @classmethod
    def _parent_type(cls, applications_relatie: pd.DataFrame) -> pd.DataFrame:
        """Create two boolean indicators whether subject has parttime and/or
        fulltime parenting responsibility, respectively.

        Parameters
        ----------
        applications_relatie: applications dataframe with relatie data merged to it

        Returns
        -------
        :
            df with two boolean columns, the first indicating parttime parenting
            responsibility and the second fulltime

        Notes
        =====
        * The parttime and fulltime parenting booleans are not mutually
          exclusive, since one subject can have multiple children with varying
          responsibilities. It's also possible for both to be False, if the
          subject has no children at all.
        """
        applications_relatie["pt_parent"] = (
            applications_relatie["categorie_relatie"] == 4
        ) & (applications_relatie["aantaldagenouders_relatie"] < 7)
        applications_relatie["ft_parent"] = (
            applications_relatie["categorie_relatie"] == 4
        ) & (applications_relatie["aantaldagenouders_relatie"] == 7)

        result = applications_relatie.groupby("application_dienstnr").agg(
            pt_parent_sum=pd.NamedAgg(column="pt_parent", aggfunc="sum"),
            ft_parent_sum=pd.NamedAgg(column="ft_parent", aggfunc="sum"),
        )

        result["is_parttime_parent"] = result["pt_parent_sum"] > 0
        result["is_fulltime_parent"] = result["ft_parent_sum"] > 0

        return result[["is_parttime_parent", "is_fulltime_parent"]]

    @classmethod
    def _calculate_partner_changes_last_90d(
        cls, applications_relatie: pd.DataFrame
    ) -> pd.DataFrame:
        """Create booleans describing partner changes in the last 3 months.

        Features:
        - Whether someone changed their partner at least once (from having one
        partner to another one).
        - Whether someone separated from their partner (from having a partner to
        not having one).

        Parameters
        ----------
        applications_relatie: applications dataframe with relatie data merged to it

        Returns
        -------
        :
            dataframe with new features
        """
        filterer = SocratesRelevantPeriodFilter(
            ref_date_col="first_dtopvoer",
            begindatum_col="dtbegin_relatie",
            einddatum_col="dteinde_relatie",
            opvoer_col="dtopvoer_relatie",
            afvoer_col="dtafvoer_relatie",
            period=dt.timedelta(days=90),
        )
        filtered_applications = filterer.fit_transform(applications_relatie)

        # Filtering only partner relationships.
        filtered_applications = filtered_applications[
            filtered_applications["categorie_relatie"].isin([1])
        ]

        changed_partners_last_90d = cls._changed_partners(filtered_applications)
        changed_partners_last_90d = changed_partners_last_90d.rename(
            columns={"changed_partners": "changed_partners_last_90d"}
        )
        separated_from_partner_last_90d = cls._separated_from_partner(
            filtered_applications
        )

        features = changed_partners_last_90d.merge(
            separated_from_partner_last_90d,
            left_index=True,
            right_index=True,
            how="outer",
        )

        return features

    @classmethod
    def _changed_partners(cls, applications_relatie: pd.DataFrame) -> pd.DataFrame:
        """Check if someone changed their partner at least once (from having one
        partner to another one).

        Parameters
        ----------
        applications_relatie: applications dataframe with relatie data merged to it

        Returns
        -------
        :
            dataframe with new feature
        """
        nr_of_partners = applications_relatie.groupby("application_dienstnr").agg(
            changed_partners=pd.NamedAgg(
                column="subjectnrrelatie_relatie", aggfunc="nunique"
            ),
        )
        result = nr_of_partners > 1
        return result

    @classmethod
    def _separated_from_partner(cls, applications_relatie: pd.DataFrame) -> pd.Series:
        """Check if someone separated from their partner (from having a partner
        to not having one).

        Parameters
        ----------
        applications_relatie: applications dataframe with relatie data merged to it

        Returns
        -------
        :
            dataframe with new feature
        """
        last_partner = (
            applications_relatie.sort_values("dteinde_relatie", ascending=False)
            .drop_duplicates(subset=["application_dienstnr"], keep="first")
            .set_index("application_dienstnr")
        )
        result = last_partner["dteinde_relatie"] < last_partner["first_dtopvoer"]
        result.name = "separated_from_partner_last_90d"
        return result
