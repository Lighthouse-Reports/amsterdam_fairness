from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class SherlockProcesstapOnderzoekDataset(DatasetBase):
    """Create a dataset for the Sherlock Processtap Onderzoek data."""

    # Set the class attributes.
    name = "sherlock_processtap_onderzoek"
    table_name = "sherlock_processtap_onderzoek"
    id_column = "_technical_key"
    columns = [
        ("pon_hercontroledatum", "datetime64"),
        ("prs_id", "Int64"),
    ]

    def generate_transformer(self) -> WPICleanTransformer:
        return WPICleanTransformer(
            do_dtype_optimization=True,
            col_type_mapping=self.columns,
            fix_no_end_date=["pon_hercontroledatum"],
        )


class SherlockProcesstapOnderzoekByPrsIdAzureDatasource(AzureDatasource):
    name = "sherlock_processtap_onderzoek_by_prs_id"
    dataset_name = "sherlock_processtap_onderzoek"
    default_id_column = "prs_id"


class SherlockProcesstapOnderzoekByPrsIdAzureAPIDatasource(AzureAPIDatasource):
    name = "sherlock_processtap_onderzoek_by_prs_id"
    dataset_class = SherlockProcesstapOnderzoekDataset
    dataset_name = "sherlock"
    table_name = "sherlock_processtap_onderzoeken"
    default_id_column = "prs_id"


DatasetDBSource.subclass_source_for_dataset(
    SherlockProcesstapOnderzoekDataset,
    default_tablename="sherlock_processtap_onderzoek",
    name_suffix="by_prs_id",
    default_id_column="prs_id",
)
