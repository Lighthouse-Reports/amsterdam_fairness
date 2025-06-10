import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.components import Component
from wpi_onderzoekswaardigheid_aanvraag.components.component import (
    NoFittingRequiredMixin,
)


class SocratesPersoonPartijFeatures(NoFittingRequiredMixin, Component):
    def _transform(  # type: ignore
        self,
        scoring: bool,
        persoon: pd.DataFrame,
        partij: pd.DataFrame,
        *args,
        **kwargs,
    ):
        persoon_partij = SocratesPersoonPartijFeatures.join_persoon_partij(
            persoon, partij
        )
        result = SocratesPersoonPartijFeatures.add_features(persoon_partij, partij)
        return result

    @classmethod
    def join_persoon_partij(
        cls, persoon: pd.DataFrame, partij: pd.DataFrame
    ) -> pd.DataFrame:
        """Add partijnr to persoon.

        Parameters
        ----------
        persoon: persoon dataframe
        partij: partij dataframe

        Returns
        -------
        :
            joined data
        """
        # TODO: Filter on the specific partij that was relevant at the time of the proces
        # TODO: Need to clear up if a person can be in multiple partijen at the same time
        return persoon.merge(
            partij, left_on="subjectnr", right_on="persoonnr", how="left"
        )

    @classmethod
    def add_features(
        cls, persoon_partij: pd.DataFrame, partij: pd.DataFrame
    ) -> pd.DataFrame:
        """Add partij features to persoon_partij.

        Parameters
        ----------
        persoon_partij: persoon dataframe with partijnr
        partij: partij dataframe

        Returns
        -------
        :
            dienstpersoon dataframe with adres features
        """

        # To prevent issues from merging with NaN partijnr
        persoon_partij["partijnr"] = persoon_partij["partijnr"].astype(float)

        is_2p_partij = cls.is_2p_partij(partij)
        result = persoon_partij.merge(is_2p_partij, on="partijnr", how="left")
        result.name = "persoon_partij"
        return result

    @classmethod
    def is_2p_partij(cls, partij: pd.DataFrame) -> pd.DataFrame:
        """Create boolean indicating if a partij consists of two people.

        Parameters
        ----------
        partij: partij dataframe

        Returns
        -------
        :
            boolean feature
        """

        is_2p_partij = partij.groupby("partijnr")["persoonnr"].nunique() > 1
        is_2p_partij.name = "is_2p_partij"
        return is_2p_partij.reset_index()
