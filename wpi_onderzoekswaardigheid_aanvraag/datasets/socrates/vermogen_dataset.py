from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class SocratesVermogenDataset(DatasetBase):
    """Create a dataset for the Socrates Vermogen data."""

    # Set the class attributes.
    name = "socrates_vermogen"
    table_name = "socrates_vermogen"
    id_column = "_technical_key"
    columns = [
        ("bedrag", float),
        ("dtafvoer", "datetime64"),
        ("dtbegin", "datetime64"),
        ("dteinde", "datetime64"),
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
            fix_no_end_date=["dteinde"],
        )


class SocratesVermogenBySubjectnrAzureDatasource(AzureDatasource):
    name = "socrates_vermogen_by_subjectnr"
    dataset_name = "socrates_vermogen"
    default_id_column = "subjectnr"


class SocratesVermogenBySubjectnrAzureAPIDatasource(AzureAPIDatasource):
    name = "socrates_vermogen_by_subjectnr"
    dataset_class = SocratesVermogenDataset
    dataset_name = "socrates"
    table_name = "socrates_vermogens"
    default_id_column = "subjectnr"


DatasetDBSource.subclass_source_for_dataset(
    SocratesVermogenDataset,
    default_tablename="socrates_vermogen",
    name_suffix="by_subjectnr",
    default_id_column="subjectnr",
)
