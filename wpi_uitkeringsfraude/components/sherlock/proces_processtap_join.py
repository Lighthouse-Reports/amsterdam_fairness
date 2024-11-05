import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.components import Component
from wpi_onderzoekswaardigheid_aanvraag.components.component import (
    NoFittingRequiredMixin,
)


class SherlockProcesProcesstapJoin(NoFittingRequiredMixin, Component):
    def _transform(  # type: ignore
        self,
        scoring: bool,
        proces: pd.DataFrame,
        processtap: pd.DataFrame,
        *args,
        **kwargs,
    ):
        return SherlockProcesProcesstapJoin.join_proces_processtap(proces, processtap)

    @classmethod
    def join_proces_processtap(
        cls, proces: pd.DataFrame, processtap: pd.DataFrame
    ) -> pd.DataFrame:
        """Merges proces data with processtap data.

        Parameters
        ----------
        proces: proces dataframe
        processtap: processtap dataframe

        Returns
        -------
        :
            joined data
        """
        result = proces.merge(
            processtap, on="pro_id", how="left", suffixes=["_proces", "_processtap"]
        )

        result.name = "proces_processtap"
        return result
