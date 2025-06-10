from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class SocratesHuisvestingDataset(DatasetBase):
    """Create a dataset for the Socrates Huisvesting data."""

    # Set the class attributes.
    name = "socrates_huisvesting"
    table_name = "socrates_huisvesting"
    id_column = "_technical_key"
    columns = [
        ("dtafvoer", "datetime64"),
        ("dtbegin", "datetime64"),
        ("dteinde", "datetime64"),
        ("dtopvoer", "datetime64"),
        ("geldig", float),
        ("huiseigenaar", float),
        ("status", float),
        ("subjectnr", "Int64"),
        ("vakantie", float),
    ]

    def generate_transformer(self) -> WPICleanTransformer:
        return WPICleanTransformer(
            do_dtype_optimization=True,
            remove_invalidated_data=True,
            col_type_mapping=self.columns,
            fix_no_end_date=["dteinde"],
        )


class SocratesHuisvestingBySubjectnrAzureDatasource(AzureDatasource):
    name = "socrates_huisvesting_by_subjectnr"
    dataset_name = "socrates_huisvesting"
    default_id_column = "subjectnr"


class SocratesHuisvestingBySubjectnrAzureAPIDatasource(AzureAPIDatasource):
    name = "socrates_huisvesting_by_subjectnr"
    dataset_class = SocratesHuisvestingDataset
    dataset_name = "socrates"
    table_name = "socrates_huisvestingen"
    default_id_column = "subjectnr"


DatasetDBSource.subclass_source_for_dataset(
    SocratesHuisvestingDataset,
    default_tablename="socrates_huisvesting",
    name_suffix="by_subjectnr",
    default_id_column="subjectnr",
)
