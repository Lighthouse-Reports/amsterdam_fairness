import pandas as pd
from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class GaloUitvalredenenDataset(DatasetBase):
    """Create a dataset for the GALO Uitvalredenen data."""

    # Set the class attributes.
    name = "galo_uitvalredenen"
    table_name = "galo_uitvalredenen"
    id_column = "processid"
    columns = [
        ("datumuitval", "datetime64"),
        ("processid", str),
        ("redenuitvalid", str),
        ("redenuitval", str),
    ]

    def generate_transformer(self) -> WPICleanTransformer:
        return WPICleanTransformer(
            do_dtype_optimization=True, col_type_mapping=self.columns
        )

    def _prepare(self, *args, **kwargs) -> pd.DataFrame:
        super()._prepare()
        return self.data


class GaloUitvalredenenByProcessidAzureDatasource(AzureDatasource):
    name = "galo_uitvalredenen_by_processid"
    dataset_name = "galo_uitvalredenen"
    default_id_column = "processid"


class GaloUitvalredenenByProcessidAzureAPIDatasource(AzureAPIDatasource):
    name = "galo_uitvalredenen_by_processid"
    dataset_class = GaloUitvalredenenDataset
    dataset_name = "galo"
    table_name = "galo_uitvalredenen"
    default_id_column = "processid"


DatasetDBSource.subclass_source_for_dataset(
    GaloUitvalredenenDataset,
    default_tablename="galo_uitvalredenen",
    name_suffix="by_processid",
    default_id_column="processid",
)
