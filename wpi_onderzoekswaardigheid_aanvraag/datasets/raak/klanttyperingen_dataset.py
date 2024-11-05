from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class RaakKlanttyperingenDataset(DatasetBase):
    """Create a dataset for the Raak Klanttyperingen data."""

    # Set the class attributes.
    name = "raak_klanttyperingen"
    table_name = "raak_klanttyperingen"
    id_column = "_technical_key"
    columns = [
        ("aanvang_klanttypering", "datetime64"),
        ("administratienummer", "Int64"),
        ("einde_klanttypering", "datetime64"),
        ("klanttypering", str),
    ]

    def generate_transformer(self) -> WPICleanTransformer:
        return WPICleanTransformer(
            do_dtype_optimization=True,
            col_type_mapping=self.columns,
            fix_no_end_date=["einde_klanttypering"],
        )


class RaakKlanttyperingenByAdministratienummerAzureDatasource(AzureDatasource):
    name = "raak_klanttyperingen_by_administratienummer"
    dataset_name = "raak_klanttyperingen"
    default_id_column = "administratienummer"


class RaakKlanttyperingenByAdministratienummerAzureAPIDatasource(AzureAPIDatasource):
    name = "raak_klanttyperingen_by_administratienummer"
    dataset_class = RaakKlanttyperingenDataset
    dataset_name = "raak"
    table_name = "raak_klanttyperingen"
    default_id_column = "administratienummer"


DatasetDBSource.subclass_source_for_dataset(
    RaakKlanttyperingenDataset,
    default_tablename="raak_klanttyperingen",
    name_suffix="by_administratienummer",
    default_id_column="administratienummer",
)
