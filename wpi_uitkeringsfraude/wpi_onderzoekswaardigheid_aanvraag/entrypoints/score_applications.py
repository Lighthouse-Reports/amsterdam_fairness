import asyncio
import logging
import os

import joblib
from azureml.core import Run
from azureml.core.model import Model
from fraude_preventie import setup_logging
from fraude_preventie.datasources.base import PoolFactory

from wpi_onderzoekswaardigheid_aanvraag.scorer import Scorer
from wpi_onderzoekswaardigheid_aanvraag.settings.settings import WPISettings


async def main():
    logger = logging.getLogger(__name__)
    run = Run.get_context()

    config_file = "dev-config.yml"
    # Do not print the settings! They can contain credentials etc.
    settings = WPISettings.set_from_yaml(config_file)

    setup_logging(settings["logging"])

    # This is required to make the asynchronous data fetching work.
    await PoolFactory.reset_lock()

    # Read application IDs from file.
    application_dienstnr_to_score = run.input_datasets[
        "application_dienstnr_to_score"
    ].to_pandas_dataframe()
    application_dienstnr = application_dienstnr_to_score[
        "application_dienstnr"
    ].values.tolist()

    classification_threshold = 0.56
    model_path = Model.get_model_path("wpi_model")
    model_file = joblib.load(f"{model_path}/wpi_model.pkl")
    pipeline_file = joblib.load(f"{model_path}/pipeline/pipeline.pkl")

    scorer = Scorer(
        classification_threshold=float(classification_threshold),
        pipeline_file=pipeline_file,
        model_file=model_file,
    )

    # Score!! :-)
    result = await scorer.score(tuple(application_dienstnr), production_mode=False)

    mounted_output_dir = run.output_datasets["scorer_results"]
    os.makedirs(os.path.dirname(mounted_output_dir), exist_ok=True)
    file_path = f"{mounted_output_dir}/scorer_results.csv"
    logger.info(f"Writing scorer results to {file_path}...")
    result.to_csv(file_path)
    run.upload_file("outputs/scorer_results.csv", file_path)
    logger.info(f"Done. Predictions have been written to {file_path}.")

    run.complete()


if __name__ == "__main__":
    asyncio.run(main())
