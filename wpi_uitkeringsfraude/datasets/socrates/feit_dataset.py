from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class SocratesFeitDataset(DatasetBase):
    """Create a dataset for the Socrates Feit data."""

    # Set the class attributes.
    name = "socrates_feit"
    table_name = "socrates_feit"
    id_column = "_technical_key"
    columns = [
        ("dtafvoer", "datetime64"),
        ("dtbeginmaatregel", "datetime64"),
        ("dtconstatering", "datetime64"),
        ("dteindemaatregel", "datetime64"),
        ("dtopvoer", "datetime64"),
        ("geldig", float),
        ("percentage", float),
        ("rdgeenmaatregel", float),
        ("soort", float),
        ("status", float),
        ("subjectnr", "Int64"),
    ]

    def generate_transformer(self) -> WPICleanTransformer:
        return WPICleanTransformer(
            do_dtype_optimization=True,
            remove_invalidated_data=True,
            col_type_mapping=self.columns,
            fix_no_end_date=["dteindemaatregel"],
        )


class SocratesFeitBySubjectnrAzureDatasource(AzureDatasource):
    name = "socrates_feit_by_subjectnr"
    dataset_name = "socrates_feit"
    default_id_column = "subjectnr"


class SocratesFeitBySubjectnrAzureAPIDatasource(AzureAPIDatasource):
    name = "socrates_feit_by_subjectnr"
    dataset_class = SocratesFeitDataset
    dataset_name = "socrates"
    table_name = "socrates_feiten"
    default_id_column = "subjectnr"


DatasetDBSource.subclass_source_for_dataset(
    SocratesFeitDataset,
    default_tablename="socrates_feit",
    name_suffix="by_subjectnr",
    default_id_column="subjectnr",
)
