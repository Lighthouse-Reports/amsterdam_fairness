from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class SocratesAdresDataset(DatasetBase):
    """Create a dataset for the Socrates Adres data."""

    # Set the class attributes.
    name = "socrates_adres"
    table_name = "socrates_adres"
    id_column = "_technical_key"
    columns = [
        ("dtafvoer", "datetime64"),
        ("dtbegin", "datetime64"),
        ("dteinde", "datetime64"),
        ("dtopvoer", "datetime64"),
        ("geldig", "float32"),
        ("huisletter", str),
        ("huisnr", "float32"),
        ("huisnrtoev", str),
        ("plaats", str),
        ("postcodenum", "float32"),
        ("soort", "float32"),
        ("status", "float32"),
        ("subjectnr", "Int64"),
        ("straat", str),
    ]

    def generate_transformer(self) -> WPICleanTransformer:
        return WPICleanTransformer(
            do_dtype_optimization=[
                "subjectnr",
                "soort",
                "huisnr",
                "huisletter",
                "huisnrtoev",
                "postcodenum",
                "geldig",
                "status",
            ],
            remove_invalidated_data=True,
            col_type_mapping=self.columns,
            fix_no_end_date=["dteinde"],
            clean_string_columns=["plaats"],
        )


class SocratesAdresBySubjectnrAzureDatasource(AzureDatasource):
    name = "socrates_adres_by_subjectnr"
    dataset_name = "socrates_adres"
    default_id_column = "subjectnr"


class SocratesAdresBySubjectnrAzureAPIDatasource(AzureAPIDatasource):
    name = "socrates_adres_by_subjectnr"
    dataset_class = SocratesAdresDataset
    dataset_name = "socrates"
    table_name = "socrates_adressen"
    default_id_column = "subjectnr"


DatasetDBSource.subclass_source_for_dataset(
    SocratesAdresDataset,
    default_tablename="socrates_adres",
    name_suffix="by_subjectnr",
    default_id_column="subjectnr",
)
