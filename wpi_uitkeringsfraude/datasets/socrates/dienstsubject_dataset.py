from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class SocratesDienstSubjectDataset(DatasetBase):
    """Create a dataset for the Socrates DienstSubject data."""

    # Set the class attributes.
    name = "socrates_dienstsubject"
    table_name = "socrates_dienstsubject"
    id_column = "_technical_key"
    columns = [
        ("dienstnr", str),
        ("dtafvoer", "datetime64"),
        ("dtbegin", "datetime64"),
        ("dteinde", "datetime64"),
        ("dtopvoer", "datetime64"),
        ("geldig", "float32"),
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


class SocratesDienstsubjectBySubjectnrAzureDatasource(AzureDatasource):
    name = "socrates_dienstsubject_by_subjectnr"
    dataset_name = "socrates_dienstsubject"
    default_id_column = "subjectnr"


class SocratesDienstsubjectBySubjectnrAzureAPIDatasource(AzureAPIDatasource):
    name = "socrates_dienstsubject_by_subjectnr"
    dataset_class = SocratesDienstSubjectDataset
    dataset_name = "socrates"
    table_name = "socrates_dienstsubjecten"
    default_id_column = "subjectnr"


DatasetDBSource.subclass_source_for_dataset(
    SocratesDienstSubjectDataset,
    default_tablename="socrates_dienstsubject",
    name_suffix="by_subjectnr",
    default_id_column="subjectnr",
)
