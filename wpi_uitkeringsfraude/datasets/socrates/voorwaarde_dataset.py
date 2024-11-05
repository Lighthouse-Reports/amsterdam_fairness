from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class SocratesVoorwaardeDataset(DatasetBase):
    """Create a dataset for the Socrates Voorwaarde data."""

    # Set the class attributes.
    name = "socrates_voorwaarde"
    table_name = "socrates_voorwaarde"
    id_column = "_technical_key"
    columns = [
        ("dienstnr", str),
        ("dtafvoer", "datetime64"),
        ("dtbegin", "datetime64"),
        ("dteinde", "datetime64"),
        ("dtopvoer", "datetime64"),
        ("geldig", "float32"),
        ("soort", "float32"),
        ("status", "float32"),
        ("subjectnr", "Int64"),
    ]

    def generate_transformer(self) -> WPICleanTransformer:
        return WPICleanTransformer(
            do_dtype_optimization=True,
            remove_invalidated_data=True,
            col_type_mapping=self.columns,
            fix_no_end_date=["dteinde"],
        )


class SocratesVoorwaardeBySubjectnrAzureDatasource(AzureDatasource):
    name = "socrates_voorwaarde_by_subjectnr"
    dataset_name = "socrates_voorwaarde"
    default_id_column = "subjectnr"


class SocratesVoorwaardeBySubjectnrAzureAPIDatasource(AzureAPIDatasource):
    name = "socrates_voorwaarde_by_subjectnr"
    dataset_class = SocratesVoorwaardeDataset
    dataset_name = "socrates"
    table_name = "socrates_voorwaarden"
    default_id_column = "subjectnr"


DatasetDBSource.subclass_source_for_dataset(
    SocratesVoorwaardeDataset,
    default_tablename="socrates_voorwaarde",
    name_suffix="by_subjectnr",
    default_id_column="subjectnr",
)
