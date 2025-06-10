import importlib.resources
import logging
import warnings
from typing import Tuple, Union

import joblib
import numpy as np
import pandas as pd
from fraude_preventie.supertimer import timer

from wpi_onderzoekswaardigheid_aanvraag.preprocessing.master_pipeline import (
    MasterPipeline,
)
from wpi_onderzoekswaardigheid_aanvraag.settings.settings import (
    WPISettings,
    setup_azure_logging,
)

logger = logging.getLogger(__name__)


# TODO: Discuss with DIA if they can translate the feature names to something meaningful. Maybe using an API on our
#  side to do the translation? Then they can be the code feature names in the database.
# Mapping of code feature names to meaningful feature names for the users.
feature_name_mapping = {
    "deelnames_started_percentage_last_year": "Percentage deelnames gestart",
    "at_least_one_address_in_amsterdam": "Adres in Amsterdam",
    "active_address_count": "Aantal adressen",
    "days_since_last_relocation": "Dagen sinds verhuizing",
    "days_since_last_dienst_end": "Dagen sinds einde dienst",
    "has_medebewoner": "Medebewoner",
    "avg_percentage_maatregel": "Gemiddelde percentage maatregel",
    "total_vermogen": "Totaal vermogen",
    "afspraken_no_show_count_last_year": "Aantal afspraken no show",
    "has_partner": "Partner",
    "sum_inkomen_bruto_was_mean_imputed": "Inkomen onbekend",
    "applied_for_same_product_last_year": "Eerder Levensonderhoud aangevraagd",
    "received_same_product_last_year": "Eerder Levensonderhoud ontvangen",
    "afspraken_no_contact_count_last_year": "Aantal afspraken geen contact",
    "sum_inkomen_bruto_value": "Totaal bruto inkomen",
}


class Scorer:
    """Class to create predictions with the pre-trained model.

    Prepares the reference data that is necessary at prediction time.

    The main entry point into this class is the `score()` method.

    The configuration of the model such as model and data flags is read from the
    model file.

    Parameters
    ----------
    classification_threshold : float
        Minimum score required to give a positive prediction. Should be between 0 and 1.
    pipeline_file : str, optional
        Pickle file that contains a fitted pipeline. If None, the pipeline shipped with
        this package is used. Default: None.
    model_file : str, optional
        Pickle file that contains a fitted model. If None, the model shipped with
        this package is used. Default: None.
    """

    def __init__(
        self,
        classification_threshold: float,
        pipeline_file: Union[str, MasterPipeline] = None,
        model_file: Union[str, dict] = None,
    ):
        if (classification_threshold < 0) & (classification_threshold > 1):
            raise ValueError(
                f"`classification_threshold` must be between 0 and 1, value given was: {classification_threshold}"
            )
        self.classification_threshold = classification_threshold
        if isinstance(pipeline_file, MasterPipeline):
            self.pipeline = pipeline_file
        else:
            self.pipeline = self._initialize_pipeline(pipeline_file)
        if isinstance(model_file, dict):
            self.model_dict = model_file
        else:
            self.model_dict = self._initialize_model(model_file)

        # Unpack the model
        self.model = self.model_dict["model"]
        self.prep = self.model[
            :-1
        ]  # All but the last step in the pipeline, so all transformers
        self.clf = self.model[-1]  # The actual model itself
        self.best_estimator = self.clf.best_estimator_

        if (
            not self.best_estimator.__class__.__name__
            == "ExplainableBoostingClassifier"
            and not self.best_estimator.__class__.__name__ == "EBMTransformer"
        ):
            logger.error(
                f"Scorer is only implemented for ExplainableBoostingClassifier or EBMTransformer, model provided is: "
                f"{self.best_estimator.__class__.__name__}"
            )

    async def score(
        self,
        application_dienstnr: Tuple[Union[int, str], ...],
        config_file_path: str = "production-config.yml",
        production_mode: bool = True,
    ) -> pd.DataFrame:
        """Make prediction for the given dienstnr of applications. The `application_dienstnr` should
        be valid dienstnr from the Socrates table Dienst.

        Parameters
        ----------
        application_dienstnr:
            tuple of dienstnr from the Socrates table Dienst to be scored; must be a tuple so that caching
            works, since a list cannot be hashed.
        config_file_path: str
            WPISettings config file path in case we are using multi-processing.
        production_mode: bool
            If the score is going to be run in the API in production env.

        Returns
        -------
        result:
            The output is pandas Dataframe with one row for every application ('aanvraag')
            requested. The index of the returned DataFrame is the application ID.

            For every application, the returned output contains the following columns:

            - subjectnr: Subject number belonging to the application. Also called administratienummer
              or adminnummer in some systems.
            - prediction: 1 if application is predicted to be 'onderzoekswaardig', i.e.
              if the score for the application is at least as high as the classification
              threshold; 0 if not.
            - score: Risk score of the application.
            - feature_contributions: Dictionary of the contributions of each feature to the final score,
              structured as {feature: contrib}
            - feature_values: Dictionary of the feature values, structured as {feature: value}
        """
        with warnings.catch_warnings(record=True) as caught_warnings:
            if production_mode:
                WPISettings.setup_production_settings(config_file=config_file_path)
                setup_azure_logging(WPISettings.get_settings()["logging"])
            logger.debug(
                f"score function input: {application_dienstnr}, {config_file_path}"
            )
            if len(application_dienstnr) == 0:
                raise ValueError("`application_dienstnr` cannot be empty")

            application_dienstnr_list = [i for i in application_dienstnr]

            with timer("Preprocessing applications...", loglevel=logging.INFO):
                transformed_df = await self.pipeline.transform(
                    application_dienstnr_list, scoring=True
                )

            with timer("Making predictions...", loglevel=logging.INFO):
                prepped_df = self.prep.transform(transformed_df)
                if prepped_df.isnull().sum().sum() > 0:
                    logger.warning(
                        f"The prepared dataframe contains {prepped_df.isnull().sum().sum()} null values."
                    )

                probs, contrib = self.best_estimator.predict_and_contrib(prepped_df)
                probs = probs[:, 1]
                preds = (probs >= self.classification_threshold).astype(int)

                if not ((probs >= 0) & (probs <= 1)).all():
                    raise RuntimeError(
                        "One or more calculated probabilities are not between 0 and 1"
                    )
                if (np.isnan(probs)).any():
                    raise RuntimeError("One or more calculated probabilities are NaN")
                if not ((preds == 0) | (preds == 1)).all():
                    raise RuntimeError(
                        "One or more predictions are NaN or not equal to 0 or 1"
                    )
                if len(application_dienstnr_list) != len(probs):
                    raise RuntimeError(
                        f"Predictions are not complete: the input application_dienstnr "
                        f"had length {len(application_dienstnr_list)}, calculated probs has "
                        f"length {len(probs)}"
                    )

                prepped_df.index = transformed_df["application_dienstnr"]
                contrib = pd.DataFrame(
                    contrib,
                    columns=self.best_estimator.feature_names,
                    index=prepped_df.index,
                )

            feature_contrib_dict = contrib.to_dict(orient="records")
            feature_vals_dict = prepped_df.to_dict(orient="records")

            logger.info("Finding most important features for each prediction...")

        string_warnings = []
        for warn in caught_warnings:
            if (
                "never awaited" not in str(warn.message)
                and str(warn.message) not in string_warnings
            ):
                string_warnings.append(str(warn))

        logger.info("Assembling results dataframe...")
        result = pd.DataFrame(index=transformed_df["application_dienstnr"]).assign(
            subjectnr=transformed_df["subjectnr"].values,
            prediction=preds,
            score=probs,
            feature_contributions=feature_contrib_dict,
            feature_values=feature_vals_dict,
            warnings=",  ".join(string_warnings),
        )

        with timer("Updating non-Amsterdam applications...", loglevel=logging.INFO):
            result = self.adjust_results_for_non_amsterdam_addresses(
                transformed_df, result
            )

        result = self.add_results_type(transformed_df, result)

        logger.info("Done scoring.")
        return result

    @staticmethod
    def find_n_most_important_features_and_values(n, scorer_output):
        """Create dataframe with as rows the applications and as columns the top n most important features in terms of
        contribution to the score, as well as the value of those features.

        Parameters
        ----------
        n:
            Top n most important features will be returned
        scorer_output:
            Output dataframe of Scorer.score()
        """
        contrib = pd.DataFrame(
            list(scorer_output["feature_contributions"]),
            index=scorer_output.index,
        )
        input_data = pd.DataFrame(
            list(scorer_output["feature_values"]), index=scorer_output.index
        )

        if n > input_data.shape[1]:
            raise ValueError(
                f"`n` cannot exceed number of input features: {n} > {input_data.shape[1]}"
            )

        # Find the n features per prediction with the highest contribution.
        contrib = contrib.drop(columns=["msg"]) if "msg" in contrib.columns else contrib
        order = np.argsort(-abs(contrib.values), axis=1)[:, :n]
        result = pd.DataFrame(
            np.array(contrib.columns)[order],
            columns=["most_important_feature_{}".format(i) for i in range(1, n + 1)],
            index=contrib.index,
        )

        for col in result.columns:
            # Look up the feature values for the most important features.
            result = result.set_index(col, append=True, drop=False)
            result[f"{col}_value"] = input_data.stack()
            result = result.reset_index(level=col, drop=True)

            for name_in_code, name_for_users in feature_name_mapping.items():
                result[col] = result[col].str.replace(name_in_code, name_for_users)

        return result

    @staticmethod
    def adjust_results_for_non_amsterdam_addresses(
        transformed_df: pd.DataFrame, result: pd.DataFrame
    ) -> pd.DataFrame:
        filter_non_amsterdam_addresses = (
            transformed_df["at_least_one_address_in_amsterdam"] == False
        ) & (transformed_df["active_address_unknown"] == False)
        non_amsterdam_application_dienstnrs = transformed_df[
            filter_non_amsterdam_addresses
        ]["application_dienstnr"]
        result_filter = result.index.isin(non_amsterdam_application_dienstnrs)
        result.loc[result_filter, "score"] = 0
        result.loc[result_filter, "prediction"] = 0

        # Don't return feature values and contributions if no Amsterdam address.
        nan_dict = {
            k: np.nan for k in result.loc[result.index[0], "feature_values"].keys()
        }
        result.loc[result_filter, "feature_values"] = np.repeat(  # type: ignore
            nan_dict, result_filter.sum()
        )
        result.loc[result_filter, "feature_contributions"] = np.repeat(  # type: ignore
            nan_dict, result_filter.sum()
        )

        result["has_ams_address"] = np.where(result_filter, 0, 1)
        return result

    @staticmethod
    def add_results_type(
        transformed_df: pd.DataFrame, result: pd.DataFrame
    ) -> pd.DataFrame:
        """The results type is used to add the right textual information to display in the Dataverkenner.

        1: Onderzoekswaardig
        2: Niet onderzoekswaardig
        3: IOAW/IOAZ
        4: Missing prediction other reasons
        """
        # Conditions are in this order, because they're evaluated in this order.
        # This way if it's IOAW/IOAZ it'll for sure get that number, even if a prediction has been made.
        conditions = {
            3: transformed_df["application_productnr"].isin([87, 88]),
            1: result["prediction"] == 1,
            2: result["prediction"] == 0,
            4: result["prediction"].isna(),
        }

        result["results_type"] = np.select(
            list(conditions.values()), list(conditions.keys()), default=np.nan
        )

        return result

    @staticmethod
    def _initialize_pipeline(pipeline_file):
        """Read fitted preprocessing pipeline into memory. If no file is provided,
        this loads the pipeline that ships with this package.
        """
        if pipeline_file is None:
            pipeline_file = importlib.resources.path(
                "wpi_onderzoekswaardigheid_aanvraag.resources", "pipeline.pkl"
            )
        logger.info(f"Loading pipeline from {pipeline_file}")
        return joblib.load(pipeline_file)

    @staticmethod
    def _initialize_model(model_file):
        """Read fitted model into memory. If no file is provided, this loads the
        model that ships with this package.
        """
        if model_file is None:
            model_file = importlib.resources.path(
                "wpi_onderzoekswaardigheid_aanvraag.resources", "model.pkl"
            )
        logger.info(f"Loading model from {model_file}")
        return joblib.load(model_file)
