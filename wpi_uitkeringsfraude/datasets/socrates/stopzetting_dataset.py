from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class SocratesStopzettingDataset(DatasetBase):
    """Create a dataset for the Socrates Stopzetting data."""

    # Set the class attributes.
    name = "socrates_stopzetting"
    table_name = "socrates_stopzetting"
    id_column = "_technical_key"
    columns = [
        ("dienstnr", str),
        ("dtafvoer", "datetime64"),
        ("dtbeginstopzetting", "datetime64"),
        ("dtopvoer", "datetime64"),
        ("geldig", float),
        ("reden", float),
        ("status", float),
    ]

    def generate_transformer(self) -> WPICleanTransformer:
        return WPICleanTransformer(
            do_dtype_optimization=True,
            remove_invalidated_data=True,
            col_type_mapping=self.columns,
        )


class SocratesStopzettingByDienstnrAzureDatasource(AzureDatasource):
    name = "socrates_stopzetting_by_dienstnr"
    dataset_name = "socrates_stopzetting"
    default_id_column = "dienstnr"


class SocratesStopzettingByDienstnrAzureAPIDatasource(AzureAPIDatasource):
    name = "socrates_stopzetting_by_dienstnr"
    dataset_class = SocratesStopzettingDataset
    dataset_name = "socrates"
    table_name = "socrates_stopzettingen"
    default_id_column = "dienstnr"


DatasetDBSource.subclass_source_for_dataset(
    SocratesStopzettingDataset,
    default_tablename="socrates_stopzetting",
    name_suffix="by_dienstnr",
    default_id_column="dienstnr",
)
