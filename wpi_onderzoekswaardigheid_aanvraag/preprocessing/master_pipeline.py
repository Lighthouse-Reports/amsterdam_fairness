import logging
from typing import Any, Dict, List, Tuple, Union

import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.components import (  # SocratesPersoonPartijFeatures,
    ApplicationFeatures,
    DienstToApplicationsAndFiltering,
    LabelAndFiltering,
    ProcesTypeInfo,
)
from wpi_onderzoekswaardigheid_aanvraag.components.galo import GaloBusinessRules
from wpi_onderzoekswaardigheid_aanvraag.components.raak import (
    RaakAfsprakenFeatures,
    RaakDeelnamesFeatures,
    RaakKlanttyperingenFeatures,
    RaakWerkervaringFeatures,
)
from wpi_onderzoekswaardigheid_aanvraag.components.sherlock import (
    SherlockProcesFeatures,
    SherlockProcesKlantJoin,
)
from wpi_onderzoekswaardigheid_aanvraag.components.socrates import (
    SocratesAdresFeatures,
    SocratesDienstFeatures,
    SocratesDienstPersoonJoin,
    SocratesDienstSherlockProcesJoin,
    SocratesDienstsubjectPartijJoin,
    SocratesDienstWerkopdrachtJoin,
    SocratesFeitFeatures,
    SocratesHuisvestingFeatures,
    SocratesInkomenFeatures,
    SocratesRelatieFeatures,
    SocratesStopzettingFeatures,
    SocratesVermogenFeatures,
    SocratesVoorwaardeFeatures,
)
from wpi_onderzoekswaardigheid_aanvraag.datasets.galo import (
    GaloEwwbBerichtenDataset,
    GaloUitvalredenenDataset,
)
from wpi_onderzoekswaardigheid_aanvraag.datasets.raak import (
    RaakAfsprakenDataset,
    RaakDeelnamesDataset,
    RaakKlantDataset,
    RaakKlanttyperingenDataset,
)
from wpi_onderzoekswaardigheid_aanvraag.datasets.sherlock import (
    SherlockKlantDataset,
    SherlockProcesDataset,
    SherlockProcesstapDataset,
    SherlockProcesstapOnderzoekDataset,
)
from wpi_onderzoekswaardigheid_aanvraag.datasets.socrates import (
    SocratesAdresDataset,
    SocratesDienstDataset,
    SocratesDienstredenDataset,
    SocratesDienstSubjectDataset,
    SocratesFeitDataset,
    SocratesHuisvestingDataset,
    SocratesInkomenDataset,
    SocratesPartijRelatie,
    SocratesPersoonDataset,
    SocratesRefSrtInkomenDataset,
    SocratesRelatieDataset,
    SocratesStopzettingDataset,
    SocratesVermogenDataset,
    SocratesVoorwaardeDataset,
    SocratesWerkopdrachtDataset,
)
from wpi_onderzoekswaardigheid_aanvraag.settings.flags import PipelineFlag
from wpi_onderzoekswaardigheid_aanvraag.settings.settings import WPISettings

logger = logging.getLogger(__name__)


class MasterPipeline:
    """Prepares and combines all data.

    All datasets are expected to be in the status as if they were freshly downloaded

    Parameters
    ----------
    """

    def __init__(
        self,
        cached_data: Dict[str, Any] = None,
    ):
        self.is_fit = False
        self.cache: Dict[str, Any] = cached_data
        self.fitted_components: Dict[str, Any] = {}

    def __getstate__(self):
        """When pickling, don't pickle the cache"""
        return {k: (v if k != "cache" else None) for k, v in self.__dict__.items()}

    async def fit_transform(self) -> pd.DataFrame:
        (
            applications,
            proces_klant,
            afspraken,
        ) = await self.fit()
        df = await self.transform(
            input_data=(applications, proces_klant),
            scoring=False,
            afspraken=afspraken,
        )
        return df

    async def fit(self):
        scoring = False

        # Fetch all diensten because they form the basis of our dataset (applications).
        dienst = await SocratesDienstDataset().fetch_all("by_subjectnr")

        applications, proces_klant = await self._prep_applications_and_label(
            scoring,
            dienst,
        )

        afspraken = await RaakAfsprakenDataset().fetch(
            applications.subjectnr.unique(), "by_administratienummer"
        )

        self.fitted_components["raak_afspraken_features"] = RaakAfsprakenFeatures().fit(
            applications=applications,
            afspraken=afspraken,
        )

        self.is_fit = True

        # Return the already transformed data so that it can be re-used in the
        # `transform()` if this method was called by `fit_transform()`.
        return applications, proces_klant, afspraken

    async def transform(
        self,
        input_data: Union[List[Union[int, str]], Tuple[pd.DataFrame, pd.DataFrame]],
        scoring: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Clean and combine all datasets into one dataframe ready for training/scoring.

        Parameters
        ----------
        input_data
            either a list of dienstnr of applications to transform or a tuple with the
            output of `MasterPipeline()._prep_applications_and_label()`
        scoring
            whether to run in scoring mode. In scoring mode e.g., the label will not
            be processed, and we may need pretrained transformers
        Returns
        -------
        result:
            dataframe that contains combined and cleaned information
        """
        self._raise_if_not_fitted()

        # If the input diensten have not yet been processed into applications.
        if isinstance(input_data, list):
            input_diensten = await SocratesDienstDataset().fetch(
                input_data, "by_dienstnr"
            )

            if len(input_diensten) == 0:
                raise RuntimeError(
                    f"No diensten could be fetched for input data: {input_data}"
                )

            logger.info(f"Fetched the following dienst to score: {input_diensten}")

            applications, proces_klant = await self._prep_applications_and_label(
                scoring,
                input_diensten,
            )

            if set(applications["application_dienstnr"]) != set(input_data):
                raise RuntimeError(
                    f"Input dienstnr and fetched applications are not the same; the "
                    f"following dienstnr are in input, but not in fetched applications: "
                    f"{set(input_data).difference(set(applications['application_dienstnr']))}"
                    f"the following dienstnr were fetched but are not in input:"
                    f" {set(applications['application_dienstnr']).difference(set(input_data))}"
                )

        elif isinstance(input_data, tuple):
            logger.debug("input_data")
            logger.debug(input_data)
            applications, proces_klant = input_data[0], input_data[1]
        else:
            raise ValueError(
                f"`input_data` should either be a list of dienstnr of applications or a tuple with the "
                f"output of `MasterPipeline()._prep_applications_and_label()`, not:"
                f" {type(input_data)}"
            )

        if "afspraken" in kwargs:
            afspraken = kwargs["afspraken"]
        else:
            afspraken = await RaakAfsprakenDataset().fetch(
                applications.subjectnr, "by_administratienummer"
            )

        subnrs = applications.subjectnr

        # Socrates
        adres = await SocratesAdresDataset().fetch(subnrs, "by_subjectnr")
        dienst = await SocratesDienstDataset().fetch(subnrs, "by_subjectnr")
        feit = await SocratesFeitDataset().fetch(subnrs, "by_subjectnr")
        huisvesting = await SocratesHuisvestingDataset().fetch(subnrs, "by_subjectnr")
        inkomen = await SocratesInkomenDataset().fetch(subnrs, "by_subjectnr")
        partij = await SocratesPartijRelatie().fetch(subnrs, "by_persoonnr")
        relatie = await SocratesRelatieDataset().fetch(subnrs, "by_subjectnr")
        vermogen = await SocratesVermogenDataset().fetch(subnrs, "by_subjectnr")
        voorwaarde = await SocratesVoorwaardeDataset().fetch(subnrs, "by_subjectnr")

        dienstsubject = await SocratesDienstSubjectDataset().fetch(
            partij.partijnr, "by_subjectnr"
        )
        dienst_of_partij = await SocratesDienstDataset().fetch(
            dienstsubject.dienstnr, "by_dienstnr"
        )
        dienst = pd.concat([dienst, dienst_of_partij], axis=0)
        dienst = dienst.drop_duplicates()

        try:
            inkomen_soort = inkomen.soort.astype("int")
        except Exception:
            logger.warning(
                f"Impossible to cast inkomen.soort to int, original value: {inkomen.soort} "
            )
            inkomen_soort = inkomen.soort
        ref_srtinkomen = await SocratesRefSrtInkomenDataset().fetch(
            inkomen_soort, "by_srtinkomennr"
        )

        # Raak
        deelnames = await RaakDeelnamesDataset().fetch(subnrs, "by_administratienummer")
        klanttyperingen = await RaakKlanttyperingenDataset().fetch(
            subnrs, "by_administratienummer"
        )
        werkervaring = await RaakKlantDataset().fetch(subnrs, "by_administratienummer")

        # GALO
        ewwb_berichten = await GaloEwwbBerichtenDataset().fetch(
            applications.application_dienstnr, "by_aanvraagid"
        )
        uitvalredenen = await GaloUitvalredenenDataset().fetch(
            ewwb_berichten.processid, "by_processid"
        )

        dienstsubject_partij = SocratesDienstsubjectPartijJoin().transform(
            scoring=scoring,
            do_dtype_optimization=True,
            dienstsubject=dienstsubject,
            partij=partij,
        )

        applications, dienst_history = SocratesDienstFeatures().transform(
            scoring=scoring,
            do_dtype_optimization=True,
            applications=applications,
            dienst=dienst,
            dienstsubject_partij=dienstsubject_partij,
        )

        stopzetting = await SocratesStopzettingDataset().fetch(
            dienst_history.dienstnr_dienst,
            "by_dienstnr",
        )

        applications = SherlockProcesFeatures().transform(
            scoring=scoring,
            do_dtype_optimization=True,
            applications=applications,
            proces_klant=proces_klant,
        )

        # Socrates Persoon contains sensitive attributes used only in bias analysis and for reweighing.
        # They're not needed nor available in production.
        if not scoring:
            persoon = await SocratesPersoonDataset().fetch(subnrs, "by_subjectnr")
            applications = SocratesDienstPersoonJoin().transform(
                scoring=scoring,
                do_dtype_optimization=True,
                dienst=applications,
                persoon=persoon,
            )

        applications = SocratesAdresFeatures().transform(
            scoring=scoring,
            do_dtype_optimization=True,
            applications=applications,
            adres=adres,
        )

        # At training time, filter out people with an address belonging to bijzondere doelgroepen, as they're not in
        # scope of the model. In production, don't filter them out here, else they'll disappear. Instead, handle
        # them at predict time. Note that BDers that were investigated are already getting filtered out when we
        # filter for the relevant HH teams.
        if not scoring:
            n_before = len(applications)
            applications = applications[
                ~applications["bijzondere_doelgroep_address"].astype(bool)
            ]
            logger.warning(
                f"Filtered out {n_before - len(applications)} applications with bijzondere doelgroep "
                f"address in Socrates"
            )

        applications = SocratesFeitFeatures().transform(
            scoring=scoring,
            do_dtype_optimization=True,
            applications=applications,
            feit=feit,
        )

        applications = SocratesHuisvestingFeatures().transform(
            scoring=scoring,
            do_dtype_optimization=True,
            applications=applications,
            huisvesting=huisvesting,
        )

        applications = SocratesInkomenFeatures().transform(
            scoring=scoring,
            do_dtype_optimization=True,
            applications=applications,
            inkomen=inkomen,
            ref_srtinkomen=ref_srtinkomen,
        )

        # # TODO: Those partij features need to be added to `applications`
        # persoon_partij = SocratesPersoonPartijFeatures().transform(
        #     scoring=scoring, do_dtype_optimization=True, persoon=persoon, partij=partij
        # )

        applications = SocratesRelatieFeatures().transform(
            scoring=scoring,
            do_dtype_optimization=True,
            applications=applications,
            relatie=relatie,
        )

        applications = SocratesStopzettingFeatures().transform(
            scoring=scoring,
            do_dtype_optimization=True,
            applications=applications,
            dienst_history=dienst_history,
            stopzetting=stopzetting,
        )

        applications = SocratesVermogenFeatures().transform(
            scoring=scoring,
            do_dtype_optimization=True,
            applications=applications,
            vermogen=vermogen,
        )

        applications = SocratesVoorwaardeFeatures().transform(
            scoring=scoring,
            do_dtype_optimization=True,
            applications=applications,
            voorwaarde=voorwaarde,
        )

        applications = self.fitted_components["raak_afspraken_features"].transform(
            scoring=scoring,
            do_dtype_optimization=True,
            applications=applications,
            afspraken=afspraken,
        )

        applications = RaakDeelnamesFeatures().transform(
            scoring=scoring,
            do_dtype_optimization=True,
            applications=applications,
            deelnames=deelnames,
        )

        applications = RaakKlanttyperingenFeatures().transform(
            scoring=scoring,
            do_dtype_optimization=True,
            applications=applications,
            kltyp=klanttyperingen,
        )

        applications = RaakWerkervaringFeatures().transform(
            scoring=scoring,
            do_dtype_optimization=True,
            applications=applications,
            werkervaring=werkervaring,
        )

        applications = GaloBusinessRules().transform(
            scoring=scoring,
            do_dtype_optimization=True,
            applications=applications,
            ewwb_berichten=ewwb_berichten,
            uitvalredenen=uitvalredenen,
        )

        return applications

    @classmethod
    async def _prep_applications_and_label(
        cls,
        scoring,
        dienst,
    ):
        if WPISettings.get_settings().flags & PipelineFlag.DEVELOPMENT_MODE:
            dienst = dienst.head(1000000)
        dienst = DienstToApplicationsAndFiltering(
            core_productnr=WPISettings.get_settings()["model"]["core_product_numbers"],
        ).transform(
            scoring=scoring,
            do_dtype_optimization=True,
            dienst=dienst,
        )

        logger.info(f"DienstToApplicationsAndFiltering len: {len(dienst)}")

        werkopdracht = await SocratesWerkopdrachtDataset().fetch(
            dienst.application_dienstnr,
            "by_dienstnr",
        )

        logger.info(f"SocratesWerkopdrachtDataset len: {len(werkopdracht)}")

        applications = SocratesDienstWerkopdrachtJoin().transform(
            scoring,
            do_dtype_optimization=True,
            dienst=dienst,
            werkopdracht=werkopdracht,
        )

        logger.info(f"SocratesDienstWerkopdrachtJoin len: {len(applications)}")

        applications = ApplicationFeatures().transform(
            scoring=scoring, do_dtype_optimization=True, applications=applications
        )

        logger.info(f"ApplicationFeatures len: {len(applications)}")

        klant = await SherlockKlantDataset().fetch(
            applications.subjectnr, "by_kln_adminnummer"
        )

        logger.info(f"SherlockKlantDataset len: {len(klant)}")

        proces = await SherlockProcesDataset().fetch(klant.kln_id, "by_kln_id")

        logger.info(f"SherlockProcesDataset len: {len(proces)}")

        # This is needed not just for label generation but also for features
        # about previous processen, so should be run both for training and for
        # or scoring.
        proces_klant = SherlockProcesKlantJoin().transform(
            scoring=scoring,
            do_dtype_optimization=True,
            proces=proces,
            klant=klant,
        )

        logger.info(f"SherlockProcesKlantJoin len: {len(proces_klant)}")

        if not scoring:
            applications = SocratesDienstSherlockProcesJoin().transform(
                scoring=scoring,
                do_dtype_optimization=True,
                dienst_werkopdracht=applications,
                proces_klant=proces_klant,
            )

            processtap = await SherlockProcesstapDataset().fetch(
                applications.pro_id.dropna(),
                "by_pro_id",
            )
            processtap_onderzoek = await SherlockProcesstapOnderzoekDataset().fetch(
                processtap.prs_id,
                "by_prs_id",
            )

            applications = ProcesTypeInfo().transform(
                scoring=scoring,
                do_dtype_optimization=True,
                proces=applications,
                processtap=processtap,
                processtap_onderzoek=processtap_onderzoek,
            )

            dienstreden = await SocratesDienstredenDataset().fetch(
                applications.application_dienstnr,
                "by_dienstnr",
            )

            applications = LabelAndFiltering().transform(
                scoring=scoring,
                do_dtype_optimization=True,
                applications=applications,
                dienstreden=dienstreden,
            )

        return applications, proces_klant

    def _raise_if_not_fitted(self):
        if not self.is_fit:
            raise ValueError("not fitted. Did you call fit_transform()?")
