import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.common import replace_nan_with_zero
from wpi_onderzoekswaardigheid_aanvraag.components import Component
from wpi_onderzoekswaardigheid_aanvraag.components.component import (
    NoFittingRequiredMixin,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.rows_by_date_filter import (
    SocratesRelevantDateFilter,
)


class SocratesVermogenFeatures(NoFittingRequiredMixin, Component):
    def _transform(  # type: ignore
        self,
        scoring: bool,
        applications: pd.DataFrame,
        vermogen: pd.DataFrame,
    ):
        applications_vermogen_raw = self.join_applications_vermogen(
            applications, vermogen
        )
        result = self.add_features(applications, applications_vermogen_raw)
        return result

    @classmethod
    def join_applications_vermogen(
        cls, applications: pd.DataFrame, vermogen: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge dienst data with vermogen data.

        Parameters
        ----------
        applications: dienst dataframe processed to contain applications
        vermogen: vermogen dataframe

        Returns
        -------
        :
            dienst and vermogen dataframe joined
        """
        applications_vermogen_raw = pd.merge(
            applications[pd.notnull(applications["subjectnr"])],
            vermogen.add_suffix("_vermogen"),
            how="inner",
            left_on="subjectnr",
            right_on="subjectnr_vermogen",
        )
        return applications_vermogen_raw

    @classmethod
    def add_features(
        cls,
        applications: pd.DataFrame,
        applications_vermogen: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add vermogen features to applications.

        Parameters
        ----------
        applications: dienst dataframe processed to contain applications
        applications_vermogen: applications dataframe with vermogen data merged to it

        Returns
        -------
        :
            applications dataframe with vermogen features
        """
        applications_vermogen = cls.filter_vermogens_relevant_to_application(
            applications_vermogen
        )
        features = cls.calc_total_vermogen(applications_vermogen)
        result = applications.merge(features, how="left", on="application_dienstnr")

        result["vermogen_unknown"] = result["total_vermogen"].isna()
        result = replace_nan_with_zero(
            result,
            columns=["total_vermogen"],
        )
        result.name = "applications_vermogen"
        return result

    @classmethod
    def filter_vermogens_relevant_to_application(cls, applications_vermogen):
        """Filter vermogens that are relevant to an application.

        Parameters
        ----------
        applications_vermogen: applications dataframe with vermogen data merged to it

        Returns
        -------
        :
            filtered applications_vermogen dataframe
        """
        filterer = SocratesRelevantDateFilter(
            ref_date_col="first_dtopvoer",
            begindatum_col="dtbegin_vermogen",
            einddatum_col="dteinde_vermogen",
            opvoer_col="dtopvoer_vermogen",
            afvoer_col="dtafvoer_vermogen",
        )
        return filterer.fit_transform(applications_vermogen)

    @classmethod
    def calc_total_vermogen(cls, applications_vermogen: pd.DataFrame) -> pd.DataFrame:
        """Calculate total vermogen.

        Parameters
        ----------
        applications_vermogen: applications dataframe with vermogen data merged to it

        Returns
        -------
        :
            features
        """
        features = applications_vermogen.groupby("application_dienstnr").agg(
            total_vermogen=pd.NamedAgg(column="bedrag_vermogen", aggfunc="sum"),
        )
        return features
