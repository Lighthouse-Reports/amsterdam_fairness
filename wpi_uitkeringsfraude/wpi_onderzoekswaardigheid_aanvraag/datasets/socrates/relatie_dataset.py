from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class SocratesRelatieDataset(DatasetBase):
    """Create a dataset for the Socrates Relatie data."""

    # Set the class attributes.
    name = "socrates_relatie"
    table_name = "socrates_relatie"
    id_column = "_technical_key"
    columns = [
        ("aantaldagenouders", float),
        ("categorie", float),
        ("dtafvoer", "datetime64"),
        ("dtbegin", "datetime64"),
        ("dteinde", "datetime64"),
        ("dtopvoer", "datetime64"),
        ("geldig", float),
        ("kduitsluiting", float),
        ("samenwonend", float),
        ("soort", float),
        ("status", float),
        ("subjectnr", "Int64"),
        ("subjectnrrelatie", "Int64"),
    ]

    def generate_transformer(self) -> WPICleanTransformer:
        return WPICleanTransformer(
            do_dtype_optimization=True,
            remove_invalidated_data=True,
            col_type_mapping=self.columns,
            fix_no_end_date=["dtbegin", "dteinde"],
        )


class SocratesRelatieBySubjectnrAzureDatasource(AzureDatasource):
    name = "socrates_relatie_by_subjectnr"
    dataset_name = "socrates_relatie"
    default_id_column = "subjectnr"


class SocratesRelatieBySubjectnrAzureAPIDatasource(AzureAPIDatasource):
    name = "socrates_relatie_by_subjectnr"
    dataset_class = SocratesRelatieDataset
    dataset_name = "socrates"
    table_name = "socrates_relaties"
    default_id_column = "subjectnr"


DatasetDBSource.subclass_source_for_dataset(
    SocratesRelatieDataset,
    default_tablename="socrates_relatie",
    name_suffix="by_subjectnr",
    default_id_column="subjectnr",
)
