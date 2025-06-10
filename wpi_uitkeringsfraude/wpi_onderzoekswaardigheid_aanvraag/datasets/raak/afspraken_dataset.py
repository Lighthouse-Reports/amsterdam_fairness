import pandas as pd
from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class RaakAfsprakenDataset(DatasetBase):
    """Create a dataset for the Raak Afspraken data."""

    # Set the class attributes.
    name = "raak_afspraken"
    table_name = "raak_afspraken"
    id_column = "_technical_key"
    columns = [
        ("administratienummer", "Int64"),
        ("afspraakresultaat", str),
        ("afspraaktype", str),
        ("datum_afhandeling_gepland", "datetime64"),
        ("datum_invoer", "datetime64"),
    ]

    def generate_transformer(self) -> WPICleanTransformer:
        return WPICleanTransformer(
            do_dtype_optimization=True,
            col_type_mapping=self.columns,
            clean_string_columns=["afspraakresultaat", "afspraaktype"],
        )

    def _prepare(self, *args, **kwargs) -> pd.DataFrame:
        super()._prepare()
        self._clean_afspraakresultaat()
        return self.data

    def _clean_afspraakresultaat(self):
        self.data["afspraakresultaat"] = self.data["afspraakresultaat"].loc[
            (
                self.data["afspraakresultaat"]
                != "Melding onterecht ingevoerd (bug nav SEIN 2016.219)"
            )
        ]


class RaakAfsprakenByAdministratienummerAzureDatasource(AzureDatasource):
    name = "raak_afspraken_by_administratienummer"
    dataset_name = "raak_afspraken"
    default_id_column = "administratienummer"


class RaakAfsprakenByAdministratienummerAzureAPIDatasource(AzureAPIDatasource):
    name = "raak_afspraken_by_administratienummer"
    dataset_class = RaakAfsprakenDataset
    dataset_name = "raak"
    table_name = "raak_afspraken"
    default_id_column = "administratienummer"


DatasetDBSource.subclass_source_for_dataset(
    RaakAfsprakenDataset,
    default_tablename="raak_afspraken",
    name_suffix="by_administratienummer",
    default_id_column="administratienummer",
)
