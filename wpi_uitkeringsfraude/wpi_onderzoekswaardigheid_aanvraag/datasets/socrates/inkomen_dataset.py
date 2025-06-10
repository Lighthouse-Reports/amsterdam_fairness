from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class SocratesInkomenDataset(DatasetBase):
    """Create a dataset for the Socrates Inkomen data."""

    # Set the class attributes.
    name = "socrates_inkomen"
    table_name = "socrates_inkomen"
    id_column = "_technical_key"
    columns = [
        ("bedragbruto", "float32"),
        ("bedragnetto", "float32"),
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
            fix_no_end_date=["dtbegin", "dteinde"],
        )


class SocratesInkomenBySubjectnrAzureDatasource(AzureDatasource):
    name = "socrates_inkomen_by_subjectnr"
    dataset_name = "socrates_inkomen"
    default_id_column = "subjectnr"


class SocratesInkomenBySubjectnrAzureAPIDatasource(AzureAPIDatasource):
    name = "socrates_inkomen_by_subjectnr"
    dataset_class = SocratesInkomenDataset
    dataset_name = "socrates"
    table_name = "socrates_inkomens"
    default_id_column = "subjectnr"


DatasetDBSource.subclass_source_for_dataset(
    SocratesInkomenDataset,
    default_tablename="socrates_inkomen",
    name_suffix="by_subjectnr",
    default_id_column="subjectnr",
)
