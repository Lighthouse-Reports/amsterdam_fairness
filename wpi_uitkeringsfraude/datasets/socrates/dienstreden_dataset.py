from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class SocratesDienstredenDataset(DatasetBase):
    """Create a dataset for the Socrates Dienstreden data."""

    # Set the class attributes.
    name = "socrates_dienstreden"
    table_name = "socrates_dienstreden"
    id_column = "_technical_key"
    columns = [
        ("dienstnr", str),
        ("dtafvoer", "datetime64"),
        ("dtopvoer", "datetime64"),
        ("geldig", float),
        ("reden", float),
        ("soort", float),
        ("status", float),
    ]

    def generate_transformer(self) -> WPICleanTransformer:
        return WPICleanTransformer(
            do_dtype_optimization=True,
            remove_invalidated_data=True,
            col_type_mapping=self.columns,
        )


class SocratesDienstredenByDienstnrAzureDatasource(AzureDatasource):
    name = "socrates_dienstreden_by_dienstnr"
    dataset_name = "socrates_dienstreden"
    default_id_column = "dienstnr"


class SocratesDienstredenByDienstnrAzureAPIDatasource(AzureAPIDatasource):
    name = "socrates_dienstreden_by_dienstnr"
    dataset_class = SocratesDienstredenDataset
    dataset_name = "socrates"
    table_name = "socrates_dienstreden"
    default_id_column = "dienstnr"


DatasetDBSource.subclass_source_for_dataset(
    SocratesDienstredenDataset,
    default_tablename="socrates_dienstreden",
    name_suffix="by_dienstnr",
    default_id_column="dienstnr",
)
