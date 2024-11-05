import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.components import Component
from wpi_onderzoekswaardigheid_aanvraag.components.component import (
    NoFittingRequiredMixin,
)


class SherlockProcesProcesstapOnderzoekJoin(NoFittingRequiredMixin, Component):
    def _transform(  # type: ignore
        self,
        scoring: bool,
        proces_processtap: pd.DataFrame,
        processtap_onderzoek: pd.DataFrame,
        *args,
        **kwargs,
    ):
        return SherlockProcesProcesstapOnderzoekJoin.join_proces_processtap_onderzoek(
            proces_processtap, processtap_onderzoek
        )

    @classmethod
    def join_proces_processtap_onderzoek(
        cls, proces_processtap: pd.DataFrame, processtap_onderzoek: pd.DataFrame
    ) -> pd.DataFrame:
        """Merges proces_processtap data with processtap_onderzoek data.

        Parameters
        ----------
        proces_processtap: proces dataframe with processtap data merged to it
        processtap_onderzoek: processtap dataframe

        Returns
        -------
        :
            joined data
        """
        result = proces_processtap.merge(
            processtap_onderzoek,
            on="prs_id",
            how="left",
            suffixes=["_proces", "_processtap_onderzoek"],
        )

        result.name = "proces_processtap_onderzoek"
        return result
