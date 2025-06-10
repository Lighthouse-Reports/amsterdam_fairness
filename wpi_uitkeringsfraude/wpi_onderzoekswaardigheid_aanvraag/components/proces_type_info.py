import logging

import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.common import replace_nan_with_zero
from wpi_onderzoekswaardigheid_aanvraag.components import Component
from wpi_onderzoekswaardigheid_aanvraag.components.component import (
    NoFittingRequiredMixin,
)
from wpi_onderzoekswaardigheid_aanvraag.components.sherlock.proces_processtap_join import (
    SherlockProcesProcesstapJoin,
)
from wpi_onderzoekswaardigheid_aanvraag.components.sherlock.proces_processtap_onderzoek_join import (
    SherlockProcesProcesstapOnderzoekJoin,
)

logger = logging.getLogger(__name__)


class ProcesTypeInfo(NoFittingRequiredMixin, Component):
    """Add information about the proces type."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _transform(  # type: ignore
        self,
        scoring: bool,
        proces: pd.DataFrame,
        processtap: pd.DataFrame,
        processtap_onderzoek: pd.DataFrame,
    ) -> pd.DataFrame:
        """Add indicators for the type of proces:
        - Was it a screening (as opposed to a handhavingsonderzoek)?
        - Did it result in a VPO being planned?

        Parameters
        ----------
        proces: proces dataframe
        processtap: processtap dataframe
        processtap_onderzoek: processtap_onderzoek dataframe

        Returns
        -------
        :
            data with two additional boolean indicators
        """
        result = proces

        proces_processtap = SherlockProcesProcesstapJoin().transform(
            scoring=scoring, proces=proces, processtap=processtap
        )

        proces_processtap_onderzoek = SherlockProcesProcesstapOnderzoekJoin().transform(
            scoring=scoring,
            proces_processtap=proces_processtap,
            processtap_onderzoek=processtap_onderzoek,
        )

        result = self._add_indicator_screening(result)
        vpo_planned = self._get_indicator_vpo_planned(proces_processtap_onderzoek)
        result = result.merge(vpo_planned, on="pro_id", how="left")
        result = replace_nan_with_zero(
            result,
            columns=["vpo_planned"],
        )
        result["vpo_planned"] = result["vpo_planned"].astype(bool)

        return result

    @classmethod
    def _add_indicator_screening(cls, proces: pd.DataFrame) -> pd.DataFrame:
        """For analysis purposes, add boolean indicators whether the proces was:
        1. A screening done by the inkomensconsulent
        2. A screening done by handhaving
        3. A handhavingsonderzoek

        Number 1 can be identified by a missing `pro_id`, because the ICers don't
        work in Sherlock. Number 2 can be identified as follows:
        1. Between 23-03-2020 and 16-07-2020 we should assume that all the
            onderzoeken bij aanvraag were screenings.
        2. From 17-07-2020 'aanvraag screening' was added as an option for the
            reason for the onderzoek, so we can simply check if 'sre_id' == 744.

        The need to distinguish between them this way comes from the fact that
        during the corona period, screenings done by handhaving got registered
        in Sherlock under the same code as regular handhavingsonderzoeken bij
        aanvraag.

        Number 3 is everything else.

        Parameters
        ----------
        proces: proces dataframe

        Returns
        -------
        :
            proces with indicator
        """
        proces["is_screening_ic"] = proces["pro_id"].isna()
        proces["is_screening_hh"] = ~proces["is_screening_ic"] & (
            proces["pro_startdatum"].between("2020-03-23", "2020-07-16")
            | (proces["sre_id"] == 744)
        )
        proces["is_onderzoek_hh"] = (
            ~proces["is_screening_ic"] & ~proces["is_screening_hh"]
        )
        return proces

    @classmethod
    def _get_indicator_vpo_planned(
        cls, proces_processtap_onderzoek: pd.DataFrame
    ) -> pd.Series:
        """Add boolean indicator if a VPO was planned as a result of the proces.

        This can be determined by checking if there’s a 'pon_hercontroledatum'
        in the Sherlock table 'processtap_onderzoek'.

        Parameters
        ----------
        proces_processtap_onderzoek: proces dataframe with processtap_onderzoek data merged to it

        Returns
        -------
        :
            proces with indicator
        """
        heronderzoek = proces_processtap_onderzoek.dropna(
            subset=["pon_hercontroledatum"]
        )
        heronderzoek = heronderzoek[
            heronderzoek["pro_einddatum"] < heronderzoek["pon_hercontroledatum"]
        ]
        vpo_planned = heronderzoek.groupby("pro_id").size() >= 1
        vpo_planned.name = "vpo_planned"
        return vpo_planned
