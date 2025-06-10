import pandas as pd
from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class SocratesDienstDataset(DatasetBase):
    """Create a dataset for the Socrates Dienst data."""

    # Set the class attributes.
    name = "socrates_dienst"
    table_name = "socrates_dienst"
    id_column = "_technical_key"
    columns = [
        ("besluit", float),
        ("dienstnr", str),
        ("dtaanvraag", "datetime64"),
        ("dtafvoer", "datetime64"),
        ("dtbegindienst", "datetime64"),
        ("dteindedienst", "datetime64"),
        ("dtopvoer", "datetime64"),
        ("geldig", float),
        ("productnr", float),
        ("srtdienst", float),
        ("status", float),
        ("subjectnrklant", "Int64"),
    ]

    def generate_transformer(self) -> WPICleanTransformer:
        return WPICleanTransformer(
            do_dtype_optimization=True,
            remove_invalidated_data=True,
            col_type_mapping=self.columns,
            fix_no_end_date=["dteindedienst"],
        )

    def _prepare(self, *args, **kwargs) -> pd.DataFrame:
        super()._prepare()
        return self.data


class SocratesDienstBySubjectnrAzureDatasource(AzureDatasource):
    name = "socrates_dienst_by_subjectnr"
    dataset_name = "socrates_dienst"
    default_id_column = "subjectnrklant"


class SocratesDienstBySubjectnrAzureAPIDatasource(AzureAPIDatasource):
    name = "socrates_dienst_by_subjectnr"
    dataset_class = SocratesDienstDataset
    dataset_name = "socrates"
    table_name = "socrates_diensten"
    default_id_column = "subjectnrklant"


class SocratesDienstBySubjectnrDBSource(DatasetDBSource):
    name = "socrates_dienst_by_subjectnr"
    dataset_class = SocratesDienstDataset
    default_tablename = "socrates_dienst"
    default_id_column = "subjectnrklant"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_where_clause(self) -> str:
        wc = super().build_where_clause()
        # Dataset for model is based on applications from 2015 onwards and we
        # have features going back at most 1 year, so we only ever need diensten
        # from 2014 or later.
        wc = f"({wc}) and (dtaanvraag >= '2014-01-01')"
        return wc


class SocratesDienstByDienstnrAzureDatasource(AzureDatasource):
    name = "socrates_dienst_by_dienstnr"
    dataset_name = "socrates_dienst"
    default_id_column = "dienstnr"


class SocratesDienstByDienstnrAzureAPIDatasource(AzureAPIDatasource):
    name = "socrates_dienst_by_dienstnr"
    dataset_class = SocratesDienstDataset
    dataset_name = "socrates"
    table_name = "socrates_diensten"
    default_id_column = "dienstnr"


class SocratesDienstByDienstnrDBSource(DatasetDBSource):
    name = "socrates_dienst_by_dienstnr"
    dataset_class = SocratesDienstDataset
    default_tablename = "socrates_dienst"
    default_id_column = "dienstnr"
