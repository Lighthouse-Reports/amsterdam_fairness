import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.components import Component
from wpi_onderzoekswaardigheid_aanvraag.components.component import (
    NoFittingRequiredMixin,
)


class SherlockProcesKlantJoin(NoFittingRequiredMixin, Component):
    def _transform(  # type: ignore
        self,
        scoring: bool,
        proces: pd.DataFrame,
        klant: pd.DataFrame,
        *args,
        **kwargs,
    ):
        return SherlockProcesKlantJoin.join_proces_klant(proces, klant)

    @classmethod
    def join_proces_klant(
        cls, proces: pd.DataFrame, klant: pd.DataFrame
    ) -> pd.DataFrame:
        """Add kln_adminnummer to proces dataframe to enable joining with
        Socrates data.

        Parameters
        ----------
        proces: proces dataframe
        klant: klant dataframe

        Returns
        -------
        :
            joined data
        """
        result = proces[pd.notnull(proces["kln_id"])].merge(
            klant[["kln_id", "kln_adminnummer"]],
            on="kln_id",
            how="inner",
            suffixes=["_proces", "_klant"],
        )
        result.name = "proces"
        return result
