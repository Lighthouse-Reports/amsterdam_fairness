import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.components import Component
from wpi_onderzoekswaardigheid_aanvraag.components.component import (
    NoFittingRequiredMixin,
)


class GaloBusinessRules(NoFittingRequiredMixin, Component):
    def _transform(  # type: ignore
        self,
        scoring: bool,
        applications: pd.DataFrame,
        ewwb_berichten: pd.DataFrame,
        uitvalredenen: pd.DataFrame,
        *args,
        **kwargs,
    ):
        applications_business_rules_raw = self.join_applications_business_rules(
            applications, ewwb_berichten, uitvalredenen
        )
        result = self.business_rules_to_features(
            applications, applications_business_rules_raw
        )
        return result

    @classmethod
    def join_applications_business_rules(
        cls,
        applications: pd.DataFrame,
        ewwb_berichten: pd.DataFrame,
        uitvalredenen: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge applications data with GALO business rules data.

        Parameters
        ----------
        applications: dienst dataframe processed to contain applications
        ewwb_berichten: ewwb_berichten dataframe
        uitvalredenen: uitvalredenen dataframe

        Returns
        -------
        :
            applications and business rules joined
        """
        applications_business_rules_raw = applications.merge(
            ewwb_berichten,
            how="left",
            left_on="application_dienstnr",
            right_on="aanvraagid",
        ).merge(
            uitvalredenen,
            how="left",
            on="processid",
        )
        return applications_business_rules_raw

    @classmethod
    def business_rules_to_features(cls, applications, applications_business_rules):
        features = (
            pd.get_dummies(
                applications_business_rules[["application_dienstnr", "redenuitvalid"]],
                columns=["redenuitvalid"],
            )
            .drop_duplicates()
            .groupby(["application_dienstnr"], as_index=False)
            .sum()
        )
        result = pd.merge(
            applications,
            features,
            how="left",
            on="application_dienstnr",
        )
        return result
