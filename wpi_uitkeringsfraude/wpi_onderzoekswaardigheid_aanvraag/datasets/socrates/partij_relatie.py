from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class SocratesPartijRelatie(DatasetBase):
    """Create a dataset for the Socrates Partij data."""

    # Set the class attributes.
    name = "socrates_partij"
    table_name = "socrates_partij"
    id_column = "_technical_key"
    columns = [
        ("partijnr", "Int64"),
        ("persoonnr", "Int64"),
    ]

    def generate_transformer(self) -> WPICleanTransformer:
        return WPICleanTransformer(do_dtype_optimization=True)


class SocratesPartijByPersoonnrAzureDatasource(AzureDatasource):
    name = "socrates_partij_by_persoonnr"
    dataset_name = "socrates_partij"
    default_id_column = "persoonnr"


class SocratesPartijByPersoonnrAzureAPIDatasource(AzureAPIDatasource):
    name = "socrates_partij_by_persoonnr"
    dataset_class = SocratesPartijRelatie
    dataset_name = "socrates"
    table_name = "socrates_partijen"
    default_id_column = "persoonnr"


DatasetDBSource.subclass_source_for_dataset(
    SocratesPartijRelatie,
    default_tablename="socrates_partij",
    name_suffix="by_persoonnr",
    default_id_column="persoonnr",
)
