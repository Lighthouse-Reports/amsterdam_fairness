import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.common import replace_nan_with_zero
from wpi_onderzoekswaardigheid_aanvraag.components import Component
from wpi_onderzoekswaardigheid_aanvraag.components.component import (
    NoFittingRequiredMixin,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.rows_by_date_filter import (
    SocratesRelevantDateFilter,
)


class SocratesInkomenFeatures(NoFittingRequiredMixin, Component):

    # Inkomen categories that will be counted; listed here so that missing
    # columns can be added if a certain category is not in the df at all.
    inkomen_categorie_tuple = (1, 2, 3, 4)

    def _transform(  # type: ignore
        self,
        scoring: bool,
        applications: pd.DataFrame,
        inkomen: pd.DataFrame,
        ref_srtinkomen: pd.DataFrame,
        *args,
        **kwargs,
    ):
        applications_inkomen_raw = self.join_applications_inkomen(applications, inkomen)
        result = self.add_features(
            applications, applications_inkomen_raw, ref_srtinkomen
        )
        return result

    @classmethod
    def join_applications_inkomen(
        cls, applications: pd.DataFrame, inkomen: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge dienst data with inkomen data.

        Parameters
        ----------
        applications: applications dataframe
        inkomen: inkomen dataframe

        Returns
        -------
        :
            dienst and inkomen dataframe joined
        """
        applications_inkomen_raw = pd.merge(
            applications[pd.notnull(applications["subjectnr"])],
            inkomen.add_suffix("_ink"),
            how="inner",
            left_on="subjectnr",
            right_on="subjectnr_ink",
        )
        return applications_inkomen_raw

    @classmethod
    def add_features(
        cls,
        applications: pd.DataFrame,
        applications_inkomen: pd.DataFrame,
        ref_srtinkomen: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add inkomen features to applications.

        Parameters
        ----------
        applications: applications dataframe
        applications_inkomen: applications dataframe with inkomen data merged to it
        ref_srtinkomen: reference table of soort inkomen

        Returns
        -------
        :
            applications dataframe with inkomen features
        """
        applications_inkomen = cls.add_and_filter_income_categories(
            applications_inkomen, ref_srtinkomen
        )
        applications_inkomen = cls.filter_inkomens_relevant_to_application(
            applications_inkomen
        )
        bedrag_features = cls.calc_summed_bedragen(applications_inkomen)
        inkomen_categorie_counts = cls.count_inkomen_categorie(applications_inkomen)

        result = applications.merge(
            bedrag_features, how="left", on="application_dienstnr"
        ).merge(inkomen_categorie_counts, how="left", on="application_dienstnr")
        result = replace_nan_with_zero(
            result,
            columns=[
                "inkomen_cat1_count",
                "inkomen_cat2_count",
                "inkomen_cat3_count",
                "inkomen_cat4_count",
            ],
        )

        result.name = "applications_inkomen"
        return result

    @classmethod
    def add_and_filter_income_categories(cls, applications_inkomen, ref_srtinkomen):
        """For each income, add which overarching category of income it is. Then filter out
        income category 6.

        The categories are (roughly):
        1: Income from or related to work (so salary, but also e.g. uitkering or pension related to work)
        2: Income not from work, for example 'alimentatie', subsidies, wealth
        3: Income from own company
        4: Heffingskorting
        6: Income "ten behoeve van Armoedevoorzieningen"; these are duplicates used by other teams. Not relevant for us.

         Parameters
        ----------
        applications_inkomen: applications dataframe with inkomen data merged to it
        ref_srtinkomen: reference table of soort inkomen

        Returns
        -------
        :
            applications_inkomen with income categories
        """
        # Cast to float to be able to merge.
        ref_srtinkomen["srtinkomennr"] = ref_srtinkomen["srtinkomennr"].astype("float")

        # Add higher-level 'categorie' based on the soort inkomen.
        merged_df = applications_inkomen.merge(
            ref_srtinkomen, left_on="soort_ink", right_on="srtinkomennr"
        )

        return merged_df[
            merged_df["categorie"].astype(int).isin(cls.inkomen_categorie_tuple)
        ]

    @classmethod
    def filter_inkomens_relevant_to_application(cls, applications_inkomen):
        """Filter inkomens that are relevant to an application.

        Parameters
        ----------
        applications_inkomen: applications dataframe with inkomen data merged to it

        Returns
        -------
        :
            filtered applications_inkomen dataframe
        """
        filterer = SocratesRelevantDateFilter(
            ref_date_col="first_dtopvoer",
            begindatum_col="dtbegin_ink",
            einddatum_col="dteinde_ink",
            opvoer_col="dtopvoer_ink",
            afvoer_col="dtafvoer_ink",
        )
        result = filterer.fit_transform(applications_inkomen)

        return result

    @classmethod
    def calc_summed_bedragen(cls, applications_inkomen: pd.DataFrame) -> pd.DataFrame:
        """Calculate net and gross summed inkomen.

        Parameters
        ----------
        applications_inkomen: applications dataframe with inkomen data merged to it

        Returns
        -------
        :
            features
        """
        features = applications_inkomen.groupby("application_dienstnr").agg(
            sum_inkomen_netto=pd.NamedAgg(column="bedragnetto_ink", aggfunc="sum"),
            sum_inkomen_bruto=pd.NamedAgg(column="bedragbruto_ink", aggfunc="sum"),
        )
        return features

    @classmethod
    def count_inkomen_categorie(
        cls, appl_ink_with_category: pd.DataFrame
    ) -> pd.DataFrame:
        """Count how many incomes of a specific category a person has.

        Parameters
        ----------
        appl_ink_with_category: applications dataframe with inkomen and inkomen category data merged to it

        Returns
        -------
        :
            features
        """
        features = (
            pd.crosstab(
                appl_ink_with_category["application_dienstnr"],
                appl_ink_with_category["categorie"].apply(int),
            )
            .add_prefix("inkomen_cat")
            .add_suffix("_count")
            .reset_index()
        )
        features = cls._get_all_relevant_categorie_counts(features)
        return features

    @classmethod
    def _get_all_relevant_categorie_counts(cls, counts: pd.DataFrame) -> pd.DataFrame:
        """Make sure that all category count columns are in the df."""
        cols_required = [
            f"inkomen_cat{int(x)}_count" for x in cls.inkomen_categorie_tuple
        ]
        cols_missing = set(cols_required) - set(counts.columns)
        counts = counts.assign(**dict.fromkeys(cols_missing, 0))
        return counts[cols_required + ["application_dienstnr"]]
