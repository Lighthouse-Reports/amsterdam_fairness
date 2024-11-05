import logging

import pandas as pd
from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer

logger = logging.getLogger(__name__)


class SherlockProcesDataset(DatasetBase):
    """Create a dataset for the Sherlock Proces data."""

    # Set the class attributes.
    name = "sherlock_proces"
    table_name = "sherlock_proces"
    id_column = "pro_id"
    columns = [
        ("kln_id", "Int64"),
        ("pro_einddatum", "datetime64"),
        ("pro_id", "Int64"),
        ("pro_startdatum", "datetime64"),
        ("spr_id", "Int64"),
        ("sre_id", "Int64"),
        ("srp_id", "Int64"),
        ("pro_teamactueelid", str),
        ("pro_teamstartid", str),
    ]

    def generate_transformer(self) -> WPICleanTransformer:
        return WPICleanTransformer(
            do_dtype_optimization=True,
            col_type_mapping=self.columns,
            drop_duplicates=True,
        )

    def _prepare(self, *args, **kwargs) -> pd.DataFrame:
        if len(self.data) == 0:  # type: ignore
            logger.info(
                "No data was fetched for this dataset, skipping further preprocessing."
            )
            return self.data  # type: ignore
        super()._prepare()
        return self.data


class SherlockProcesByKlnIdAzureDatasource(AzureDatasource):
    name = "sherlock_proces_by_kln_id"
    dataset_name = "sherlock_proces"
    default_id_column = "kln_id"


class SherlockProcesByKlnIdAzureAPIDatasource(AzureAPIDatasource):
    name = "sherlock_proces_by_kln_id"
    dataset_class = SherlockProcesDataset
    dataset_name = "sherlock"
    table_name = "sherlock_processen"
    default_id_column = "kln_id"


DatasetDBSource.subclass_source_for_dataset(
    SherlockProcesDataset,
    default_tablename="sherlock_proces",
    name_suffix="by_kln_id",
    default_id_column="kln_id",
)
