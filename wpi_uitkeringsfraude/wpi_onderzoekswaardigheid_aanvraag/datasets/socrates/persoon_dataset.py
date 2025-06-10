from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class SocratesPersoonDataset(DatasetBase):
    """Create a dataset for the Socrates Persoon data."""

    # Set the class attributes.
    name = "socrates_persoon"
    table_name = "socrates_persoon"
    id_column = "_technical_key"
    columns = [
        ("dtafvoer", "datetime64"),
        ("dtopvoer", "datetime64"),
        ("geldig", float),
        ("status", float),
        ("subjectnr", "Int64"),
    ]

    def generate_transformer(self) -> WPICleanTransformer:
        return WPICleanTransformer(
            do_dtype_optimization=True,
            remove_invalidated_data=True,
            col_type_mapping=self.columns,
        )


class SocratesPersoonBySubjectnrAzureDatasource(AzureDatasource):
    name = "socrates_persoon_by_subjectnr"
    dataset_name = "socrates_persoon"
    default_id_column = "subjectnr"


class SocratesPersoonBySubjectnrAzureAPIDatasource(AzureAPIDatasource):
    name = "socrates_persoon_by_subjectnr"
    dataset_class = SocratesPersoonDataset
    dataset_name = "socrates"
    table_name = "socrates_personen"
    default_id_column = "subjectnr"


DatasetDBSource.subclass_source_for_dataset(
    SocratesPersoonDataset,
    default_tablename="socrates_persoon",
    name_suffix="by_subjectnr",
    default_id_column="subjectnr",
)
