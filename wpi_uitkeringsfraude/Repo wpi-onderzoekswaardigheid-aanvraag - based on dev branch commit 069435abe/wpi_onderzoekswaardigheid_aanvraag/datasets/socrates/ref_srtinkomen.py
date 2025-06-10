from fraude_preventie.datasets import DatasetBase
from fraude_preventie.datasources import DatasetDBSource

from wpi_onderzoekswaardigheid_aanvraag.datasources.azure import (
    AzureAPIDatasource,
    AzureDatasource,
)
from wpi_onderzoekswaardigheid_aanvraag.preprocessing.clean import WPICleanTransformer


class SocratesRefSrtInkomenDataset(DatasetBase):
    """Create a dataset for the Socrates reference table srtinkomen."""

    # Set the class attributes.
    name = "socrates_ref_srtinkomen"
    table_name = "socrates_ref_srtinkomen"
    id_column = "_technical_key"
    columns = [
        ("categorie", str),
        ("srtinkomennr", "Int64"),
    ]

    def generate_transformer(self) -> WPICleanTransformer:
        return WPICleanTransformer(
            do_dtype_optimization=True, col_type_mapping=self.columns
        )


class SocratesRefSrtInkomenBySrtinkomennrAzureDatasource(AzureDatasource):
    name = "socrates_ref_srtinkomen_by_srtinkomennr"
    dataset_name = "socrates_ref_srtinkomen"
    default_id_column = "srtinkomennr"


class SocratesRefSrtInkomenBySrtinkomennrAzureAPIDatasource(AzureAPIDatasource):
    name = "socrates_ref_srtinkomen_by_srtinkomennr"
    dataset_class = SocratesRefSrtInkomenDataset
    dataset_name = "socrates"
    table_name = "socrates_ref_srtinkomens"
    default_id_column = "srtinkomennr"


DatasetDBSource.subclass_source_for_dataset(
    SocratesRefSrtInkomenDataset,
    default_tablename="socrates_ref_srtinkomen",
    name_suffix="by_srtinkomennr",
    default_id_column="srtinkomennr",
)
