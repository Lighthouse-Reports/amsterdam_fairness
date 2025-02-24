import pandas as pd
from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class GaloEwwbBerichtenDataset(DatasetBase):
    """Create a dataset for the GALO EWWB Berichten data."""

    # Set the class attributes.
    name = "galo_ewwbberichten"
    table_name = "galo_ewwb"
    id_column = "processid"
    columns = [
        ("aanvraagid", str),
        ("admnr", "Int64"),
        ("berichtgeldig", "Int64"),
        ("dtaanvraagafgewezen", "datetime64"),
        ("dtaanvraag", "datetime64"),
        ("dtaanvraagtoegekend", "datetime64"),
        ("dtketenafgerond", "datetime64"),
        ("dtrecordaangepast", "datetime64"),
        ("dtrecordaanmaak", "datetime64"),
        ("dtuitgevallen", "datetime64"),
        ("processid", str),
    ]

    def generate_transformer(self) -> WPICleanTransformer:
        return WPICleanTransformer(
            do_dtype_optimization=True, col_type_mapping=self.columns
        )

    def _prepare(self, *args, **kwargs) -> pd.DataFrame:
        super()._prepare()
        return self.data


class GaloEwwbBerichtenByAanvraagidAzureDatasource(AzureDatasource):
    name = "galo_ewwbberichten_by_aanvraagid"
    dataset_name = "galo_ewwbberichten"
    default_id_column = "aanvraagid"


class GaloEwwbBerichtenByAanvraagidAzureAPIDatasource(AzureAPIDatasource):
    name = "galo_ewwbberichten_by_aanvraagid"
    dataset_class = GaloEwwbBerichtenDataset
    dataset_name = "galo"
    table_name = "galo_ewwbberichten"
    default_id_column = "aanvraagid"


DatasetDBSource.subclass_source_for_dataset(
    GaloEwwbBerichtenDataset,
    default_tablename="galo_ewwbberichten",
    name_suffix="by_aanvraagid",
    default_id_column="aanvraagid",
)
