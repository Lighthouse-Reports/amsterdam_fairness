from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class SherlockKlantDataset(DatasetBase):
    """Create a dataset for the Sherlock Klant data."""

    # Set the class attributes.
    name = "sherlock_klant"
    table_name = "sherlock_klant"
    id_column = "kln_adminnummer"
    columns = [
        ("kln_adminnummer", "Int64"),
        ("kln_id", "Int64"),
    ]

    def generate_transformer(self) -> WPICleanTransformer:
        return WPICleanTransformer(do_dtype_optimization=True)


class SherlockKlantByKlnAdminnummerAzureDatasource(AzureDatasource):
    name = "sherlock_klant_by_kln_adminnummer"
    dataset_name = "sherlock_klant"
    default_id_column = "kln_adminnummer"


class SherlockKlantByKlnAdminnummerAzureAPIDatasource(AzureAPIDatasource):
    name = "sherlock_klant_by_kln_adminnummer"
    dataset_class = SherlockKlantDataset
    dataset_name = "sherlock"
    table_name = "sherlock_klanten"
    default_id_column = "kln_adminnummer"


DatasetDBSource.subclass_source_for_dataset(
    SherlockKlantDataset,
    default_tablename="sherlock_klant",
    name_suffix="by_kln_adminnummer",
    default_id_column="kln_adminnummer",
)
