from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class SherlockProcesstapDataset(DatasetBase):
    """Create a dataset for the Sherlock Processtap data."""

    # Set the class attributes.
    name = "sherlock_processtap"
    table_name = "sherlock_processtap"
    id_column = "_technical_key"
    columns = [
        ("pro_id", "Int64"),
        ("prs_id", "Int64"),
    ]

    def generate_transformer(self) -> WPICleanTransformer:
        return WPICleanTransformer(do_dtype_optimization=True)


class SherlockProcesstapByProIdAzureDatasource(AzureDatasource):
    name = "sherlock_processtap_by_pro_id"
    dataset_name = "sherlock_processtap"
    default_id_column = "pro_id"


class SherlockProcesstapByProIdAzureAPIDatasource(AzureAPIDatasource):
    name = "sherlock_processtap_by_pro_id"
    dataset_class = SherlockProcesstapDataset
    dataset_name = "sherlock"
    table_name = "sherlock_processtappen"
    default_id_column = "pro_id"


DatasetDBSource.subclass_source_for_dataset(
    SherlockProcesstapDataset,
    default_tablename="sherlock_processtap",
    name_suffix="by_pro_id",
    default_id_column="pro_id",
)
