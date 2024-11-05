from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class SocratesWerkopdrachtDataset(DatasetBase):
    """Create a dataset for the Socrates Werkopdracht data."""

    # Set the class attributes.
    name = "socrates_werkopdracht"
    table_name = "socrates_werkopdracht"
    id_column = "_technical_key"
    columns = [
        ("dienstnr", str),
        ("dtopvoer", "datetime64"),
    ]

    def generate_transformer(self) -> WPICleanTransformer:
        return WPICleanTransformer(
            do_dtype_optimization=True,
            col_type_mapping=self.columns,
        )


class SocratesWerkopdrachtByDienstnrAzureDatasource(AzureDatasource):
    name = "socrates_werkopdracht_by_dienstnr"
    dataset_name = "socrates_werkopdracht"
    default_id_column = "dienstnr"


class SocratesWerkopdrachtByDienstnrAzureAPIDatasource(AzureAPIDatasource):
    name = "socrates_werkopdracht_by_dienstnr"
    dataset_class = SocratesWerkopdrachtDataset
    dataset_name = "socrates"
    table_name = "socrates_werkopdrachten"
    default_id_column = "dienstnr"


DatasetDBSource.subclass_source_for_dataset(
    SocratesWerkopdrachtDataset,
    default_tablename="socrates_werkopdracht",
    default_id_column="dienstnr",
    name_suffix="by_dienstnr",
)
