import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.common import replace_nan_with_zero
from wpi_onderzoekswaardigheid_aanvraag.components import Component
from wpi_onderzoekswaardigheid_aanvraag.components.component import (
    NoFittingRequiredMixin,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.rows_by_date_filter import (
    SocratesRelevantDateFilter,
)


class SocratesHuisvestingFeatures(NoFittingRequiredMixin, Component):
    def _transform(  # type: ignore
        self,
        scoring: bool,
        applications: pd.DataFrame,
        huisvesting: pd.DataFrame,
        *args,
        **kwargs,
    ):
        applications_huisvesting_raw = self.join_applications_huisvesting(
            applications, huisvesting
        )
        result = self.add_features(applications, applications_huisvesting_raw)
        return result

    @classmethod
    def join_applications_huisvesting(
        cls, applications: pd.DataFrame, huisvesting: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge dienst data with huisvesting data.

        Parameters
        ----------
        applications: applications dataframe
        huisvesting: huisvesting dataframe

        Returns
        -------
        :
            dienst and huisvesting dataframe joined
        """
        applications_huisvesting_raw = pd.merge(
            applications[pd.notnull(applications["subjectnr"])],
            huisvesting.add_suffix("_hv"),
            how="inner",
            left_on="subjectnr",
            right_on="subjectnr_hv",
        )
        return applications_huisvesting_raw

    @classmethod
    def add_features(
        cls, applications: pd.DataFrame, applications_huisvesting: pd.DataFrame
    ) -> pd.DataFrame:
        """Add huisvesting features to applications.

        Parameters
        ----------
        applications: applications dataframe
        applications_huisvesting: applications dataframe with huisvesting data merged to it

        Returns
        -------
        :
            applications dataframe with huisvesting features
        """
        applications_huisvesting = cls.filter_huisvesting_relevant_to_application(
            applications_huisvesting
        )

        is_huiseigenaar = cls.indicator_huiseigenaar(applications_huisvesting)

        result = applications.merge(
            is_huiseigenaar,
            how="left",
            left_on="application_dienstnr",
            right_index=True,
        )

        result = replace_nan_with_zero(
            result,
            columns=["is_huiseigenaar"],
        )
        result["is_huiseigenaar"] = result["is_huiseigenaar"].astype(int)

        result.name = "applications_huisvesting"
        return result

    @classmethod
    def filter_huisvesting_relevant_to_application(cls, applications_hv):
        """Filter huisvesting that is relevant to an application.

        Parameters
        ----------
        applications_hv: applications dataframe with inkomen data merged to it

        Returns
        -------
        :
            filtered applications_hv dataframe
        """
        filterer = SocratesRelevantDateFilter(
            ref_date_col="first_dtopvoer",
            begindatum_col="dtbegin_hv",
            einddatum_col="dteinde_hv",
            opvoer_col="dtopvoer_hv",
            afvoer_col="dtafvoer_hv",
        )
        return filterer.fit_transform(applications_hv)

    @classmethod
    def remove_vakantie_rows(cls, huisvesting: pd.DataFrame):
        return huisvesting[huisvesting["vakantie_hv"] != 1]

    @classmethod
    def indicator_huiseigenaar(cls, applications_hv):
        """Create boolean indicator whether the subject of the application is a
        huiseigenaar or not.

        Parameters
        ----------
        applications_hv: applications dataframe with huisvesting data merged to it

        Returns
        -------
        :
            series of boolean indicator per application
        """
        f_applications_hv = cls.remove_vakantie_rows(applications_hv)
        feature = (
            f_applications_hv.groupby("application_dienstnr")["huiseigenaar_hv"].sum()
            >= 1
        )
        feature.name = "is_huiseigenaar"
        return feature
