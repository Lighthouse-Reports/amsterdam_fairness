from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class RaakDeelnamesDataset(DatasetBase):
    """Create a dataset for the Raak Deelnames data."""

    # Set the class attributes.
    name = "raak_deelnames"
    table_name = "raak_deelnames"
    id_column = "_technical_key"
    columns = [
        ("administratienummer", "Int64"),
        ("datum_aanmelding", "datetime64"),
        ("einde_deelname", "datetime64"),
        ("reden_niet_geaccepteerd", str),
        ("reden_niet_gestart", str),
        ("reden_voortijdig_afgebroken", str),
        ("start_deelname", "datetime64"),
    ]

    def generate_transformer(self) -> WPICleanTransformer:
        return WPICleanTransformer(
            do_dtype_optimization=True,
            col_type_mapping=self.columns,
            clean_string_columns=[
                "reden_niet_geaccepteerd",
                "reden_niet_gestart",
                "reden_voortijdig_afgebroken",
            ],
        )


class RaakDeelnamesByAdministratienummerAzureDatasource(AzureDatasource):
    name = "raak_deelnames_by_administratienummer"
    dataset_name = "raak_deelnames"
    default_id_column = "administratienummer"


class RaakDeelnamesByAdministratienummerAzureAPIDatasource(AzureAPIDatasource):
    name = "raak_deelnames_by_administratienummer"
    dataset_class = RaakDeelnamesDataset
    dataset_name = "raak"
    table_name = "raak_deelnames"
    default_id_column = "administratienummer"


DatasetDBSource.subclass_source_for_dataset(
    RaakDeelnamesDataset,
    default_tablename="raak_deelnames",
    name_suffix="by_administratienummer",
    default_id_column="administratienummer",
)
