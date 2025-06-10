import asyncio
import logging
import os

import joblib
import pretty_errors  # noqa: F401
from azureml.core import Run
from fraude_preventie import setup_logging
from fraude_preventie.datasources.base import PoolFactory

from wpi_onderzoekswaardigheid_aanvraag.check_expectations import check_expectations
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.master_pipeline import (
    MasterPipeline,
)
from wpi_onderzoekswaardigheid_aanvraag.settings.settings import WPISettings

logger = logging.getLogger(__name__)


async def train_and_store_preprocessing_pipeline() -> None:
    """Fits the master pipeline on the data in a given path, and stores both the
    pipeline as well as the transformed data. The filenames will be ``pipeline.pkl``
    and ``transformed_data.pkl``.

    Parameters
    ----------
    data_path:
        where to store the pipeline and transformed data
    """
    logger.info("Fitting pipeline and transforming data...")

    pl = MasterPipeline()
    df = await pl.fit_transform()

    check_expectations(
        df,
        expectations_file="expectations_on_pipeline_output.json",
        include_train_expectations=True,
        raise_upon_fail=True,
    )

    logger.info("Storing results...")
    # Get the experiment run context
    run = Run.get_context()

    mounted_output_dir = run.output_datasets["transformed_data"]
    os.makedirs(os.path.dirname(mounted_output_dir), exist_ok=True)
    df.to_parquet(
        f"{mounted_output_dir}/transformed_data.parquet",
        index=False,
        use_dictionary=False,
    )

    mounted_output_dir = run.output_datasets["pipeline_file"]
    os.makedirs(os.path.dirname(mounted_output_dir), exist_ok=True)
    joblib.dump(pl, f"{mounted_output_dir}/pipeline.pkl")

    logger.info("Done. Written transformed data and pipeline files.")


async def main():
    run = Run.get_context()
    config_file = "dev-config.yml"
    config = WPISettings.set_from_yaml(config_file)
    setup_logging(config["logging"])

    await PoolFactory.reset_lock()
    await train_and_store_preprocessing_pipeline()
    run.complete()


if __name__ == "__main__":
    asyncio.run(main())
