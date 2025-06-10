import logging

import pandas as pd

from wpi_onderzoekswaardigheid_aanvraag.components import Component
from wpi_onderzoekswaardigheid_aanvraag.components.component import (
    NoFittingRequiredMixin,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.function_decorators import (
    log_filtering_step,
)

logger = logging.getLogger(__name__)


class LabelAndFiltering(NoFittingRequiredMixin, Component):
    """Add labels and filter relevant rows."""

    # Onderzoekswaardige outcome codes for handhavingsonderzoeken.
    srp_id_onderzoekswaardig = [
        734,
        738,
        780,
        781,
        800,
        945,
        946,
        947,
        948,
        949,
        1089,
        1090,
        1091,
    ]

    # Some SRP_IDs are not a clear investigation outcome, for example when
    # the application was retracted, so they didn't finish the
    # investigation, or there was an error in the 'opvoer' of the
    # application, so the Handhaver sent it back to the Inkomensconsulent.
    # These processes are filtered out completely.
    srp_id_uitfilteren = [735, 736, 739, 1048, 1124]

    # Onderzoekswaardige reasons for rejecting an application by IC.
    dienstreden_afwijzing_onderzoekswaardig = [302, 46]

    # These codes have appeared very infrequently.
    dienstreden_afwijzing_uitfilteren = [
        1,
        9,
        15,
        40,
        45,
        73,
        80,
        95,
        369,
        420,
    ]

    # Not all teams that work in Sherlock are relevant to the project.
    # The hashing of the team IDs was probably overkill, but we left it, because
    # it's easier than going back and getting the WPD changed.
    relevant_team_ids = [
        "cf58787d1411ebe284eb5ab24eb973a220e36aa900b7401afc46ec17f7ef3240",  # Project Huisbezoeken
        "6ff5214ade9bcc25f5875eddb1166cc478c005d8ceff3b0af58d5679c89bce3f",  # Controle 1 Nieuw West
        "cd376ed3ab8b282dc8c2a15fe32e38fda9d425bdacddc693e216b716f47d694f",  # Controle 2 Noord
        "b2efa4e0254d4cda82b2572be96e0421bd575af4843cf5545bd0e9e73574a3d9",  # Controle 3 Centrum / Oost
        "1595ba227833fefbd6af53e9c1c073cb518c47159cb53246b7230691f2c6674e",  # Controle 4 Zuidoost
        "8af91530b5971bf634f777252121afbb0ed5a0eafe545b8894efba46fdf04a83",  # Controle 5 Centrum
        "6b1b6f1c7406a99460838e2fc5c5aa9549dcb1cd4b514481797e8fd3b7135b1b",  # Controle 6 Zuid / West
        "c1d5d55a5f057f1798c6ee3c4299766607ba518be2ef647758f6b66e30039235",  # Controle Project COM 1
        "a101c00b4e3b316d990d18ca75088cc85ba7fc352623aa3b25aa259ffe1be9b3",  # Controle Project COM 2
        "a585f9df9715cfcb69564e965217feaacea867736fbc7e71eb5cf7baa34ec062",  # Controle Project COM 3
        "b6930eb6976302a27eca54f3ce54afefab6da56e6d920bc7d6e60862f689cec4",  # Controle Pro - actief 2009
        "d985bf9feecae68a6b039506582d47033aa7cfaf380712bfd9e012ff3c078cb9",  # Controle Project Parttime Inkomsten
        "8c0a717abeafd70d9c5206e740d3da9c5b4db0c376ca37961455fc589a64a05e",  # Kunstenaars
        "bd862a44461c1924413f7ec4e91c43a6fad3c8987040cb032ef8bf209aa878ae",  # HH Intake
        "445a12bfc517a6c897de9181cf13ef82600ac1ce4df27f72b1789c5ab7fb3d0e",  # HH Projecten
        "f8f64d59e8187f9880612058b05f808d6cd3433a8b0bb18e132b6314db784a70",  # HH Werk
        "69a301d462e6cfe5d6e5985373b7e9da05a78e965a5df6b85a7581e63f3e07e7",  # HH Activering
        "f01df520d64894da92a4469d4559f07445a09c812ecb4022b23346fd550fb8e7",  # HH Noord - Oost
        "77c5ab199edd3c9a2aea1c287bd362e115efa92cf4de20fa5685125fae80a307",  # HH ZuidOost
        "43c566c62122780e3265063ff51636394e41e8c2340e5c485bfdd0c2d5909203",  # HH West
        "dd2f6aa1350b9b4e0c6e556ef15c179748755a10041cc6f16ab0392ab50e56a1",  # HH1 West - Noord
        "d8243f3df40b7ff20530fa816a721befb4c7540b7863afe61c95a2f0508fc180",  # HH2 Centrum - Oost - ZuidOost
        "be6c0cd08f8d1dbec247cd99d086e831de5af56e5c5bc2ecb0273cb5275a8898",  # HH3 NieuwWest - Zuid
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _transform(  # type: ignore
        self, scoring: bool, applications: pd.DataFrame, dienstreden: pd.DataFrame
    ) -> pd.DataFrame:
        """Add labels and filter the data for relevant rows.

        Parameters
        ----------
        scoring: if True, no label will be generated
        applications: applications dataframe with proces type information

        Returns
        -------
        :
            filtered data with label
        """
        result = applications
        if not scoring:
            result = (
                applications.pipe(self.drop_missing_label)
                .pipe(self.filter_proces_resultaatcodes)
                .pipe(self.filter_relevant_teams)
                .pipe(self.add_rejection_reason, dienstreden=dienstreden)
                .pipe(self.add_label)
            )
        else:
            logger.info("Not generating label because we are in scoring mode")
        return result

    @classmethod
    def drop_missing_label(cls, applications):
        """Drop rows where the outcome of the application is (yet) unknown.

        This means that if in Sherlock, it must have an `srp_id`, if in Socrates,
        it must have a `besluit`.
        """
        df = applications.copy()
        n_before = len(df)
        drop_criteria = (~df["is_screening_ic"] & df["srp_id"].isna()) | (
            df["is_screening_ic"] & df["besluit"].isna()
        )
        df = df[~drop_criteria]
        logger.debug(f"Dropped {n_before - len(df)} rows due to missing label")
        return df

    def filter_proces_resultaatcodes(self, applications: pd.DataFrame) -> pd.DataFrame:
        n_before = len(applications)
        result = applications[~(applications["srp_id"].isin(self.srp_id_uitfilteren))]
        logger.debug(
            f"Dropped {n_before - len(result)} rows due to proces resultaatcode not a clear result (hence no "
            f"label possible)"
        )
        return result

    def add_rejection_reason(
        self, applications: pd.DataFrame, dienstreden: pd.DataFrame
    ) -> pd.DataFrame:
        n_before = len(applications)

        # Only look at dienstreden related to an rejection
        dienstreden = dienstreden[dienstreden["soort"].isin([6, 774])]
        result = applications.merge(
            dienstreden.add_suffix("_dienstreden"),
            left_on="application_dienstnr",
            right_on="dienstnr_dienstreden",
            how="left",
        )
        # Filter out very uncommon dienstreden
        result = result[
            ~result["reden_dienstreden"].isin(self.dienstreden_afwijzing_uitfilteren)
        ]
        result = result.sort_values(
            ["dtafvoer", "dtopvoer"], ascending=True, na_position="last"
        ).drop_duplicates(["application_dienstnr"], keep="last")

        logger.debug(
            f"Dropped {n_before - len(result)} rejected screenings due to uncommon dienstreden"
        )
        return result

    def add_label(self, applications: pd.DataFrame) -> pd.DataFrame:
        """Add label indicating onderzoekswaardigheid to applications.

        Parameters
        ----------
        applications: applications dataframe with proces type information

        Returns
        -------
        :
            applications dataframe with additional column containing the label
        """
        ondzw_hh_onderzoek = applications["is_onderzoek_hh"].fillna(False).astype(
            bool
        ) & applications["srp_id"].isin(self.srp_id_onderzoekswaardig)
        ondzw_hh_screening = (
            applications["is_screening_hh"].fillna(False).astype(bool)
            & applications["vpo_planned"]
        )
        ondzw_ic_screening = (
            applications["is_screening_ic"].fillna(False).astype(bool)
            & (applications["besluit"] == 3)
            & applications["reden_dienstreden"].isin(
                self.dienstreden_afwijzing_onderzoekswaardig
            )
        )

        applications["onderzoekswaardig"] = (
            ondzw_hh_onderzoek | ondzw_hh_screening | ondzw_ic_screening
        ).astype(int)

        # Indicator if application was in the end rejected (besluit=3) or not.
        # This is not the label, but still interesting for evaluation, because
        # in practice onderzoeken that result in a rejection (afwijzing) or
        # change (wijziging) are counted as 'successful'.
        applications["afgewezen"] = applications["besluit"] == 3
        return applications

    @log_filtering_step
    def filter_relevant_teams(self, applications: pd.DataFrame):
        """Filter the Handhaving investigation part of the dataset on only the
        relevant Handhaving teams. Note that the filtering is only on investigations,
        any screenings will pass through.

        Parameters
        ----------
        applications: applications dataframe with proces information

        Returns
        -------
        :
            applications dataframe without HH investigations by irrelevant teams
        """
        return applications[
            applications["pro_teamactueelid"].isin(self.relevant_team_ids)
            | applications["pro_teamactueelid"].isna()
        ]
