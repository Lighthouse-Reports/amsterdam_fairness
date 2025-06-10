from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class RaakKlantDataset(DatasetBase):
    """Create a dataset for the Raak Klant data. This dataset is actually
    information about work experience.
    """

    # Set the class attributes.
    name = "raak_klant"
    table_name = "raak_klant"
    id_column = "_technical_key"
    columns = [
        ("administratienummer", "Int64"),
        ("belangrijkste_werkervaring", str),
    ]

    def generate_transformer(self) -> WPICleanTransformer:
        return WPICleanTransformer(
            do_dtype_optimization=True,
            col_type_mapping=self.columns,
            clean_string_columns=[
                "belangrijkste_werkervaring",
            ],
        )


class RaakKlantByAdministratienummerAzureDatasource(AzureDatasource):
    name = "raak_klant_by_administratienummer"
    dataset_name = "raak_klant"
    default_id_column = "administratienummer"


class RaakKlantByAdministratienummerAzureAPIDatasource(AzureAPIDatasource):
    name = "raak_klant_by_administratienummer"
    dataset_class = RaakKlantDataset
    dataset_name = "raak"
    table_name = "raak_klanten"
    default_id_column = "administratienummer"


DatasetDBSource.subclass_source_for_dataset(
    RaakKlantDataset,
    default_tablename="raak_klant",
    name_suffix="by_administratienummer",
    default_id_column="administratienummer",
)
