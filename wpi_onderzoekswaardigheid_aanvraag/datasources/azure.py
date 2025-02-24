import logging
from traceback import format_exc
from typing import Any, Dict, List, Sequence, Tuple, Union

import aiohttp
import pandas as pd
from azureml.core import Dataset, Run
from fraude_preventie import DatasetBase, DatasourceBase, RestDatasource
from fraude_preventie.adamapi import AdamApi

from wpi_onderzoekswaardigheid_aanvraag.settings.settings import WPISettings

logger = logging.getLogger(__name__)


class NotFoundInApiError(Exception):
    pass


class AzureAPIDatasource(RestDatasource):
    """Datasource that retrieves data from the Azure API."""

    path_url = ""
    auth_username: str = "wpi_onderzoekswaardigheid_aanvraag"
    auth_password: str = None
    dataset_name: str = None
    table_name: str = None
    default_id_column: str = None
    failed_query_count = 0
    total_query_count = 0
    dataset_class: DatasetBase = None

    def __init__(self):
        self.base_url = (
            f"{WPISettings.get_settings()['data_api']['url']}/data/encrypted"
        )
        self.auth_password = WPISettings.get_settings()["pw_daso_api"]

    async def fetch(self, id_to_query: str) -> List[Dict]:
        """Download a record from an API endpoint.

        Parameters
        ----------
        id_to_query : str
            id to query
        """
        try:
            data = {
                "dataset_name": self.dataset_name,
                "table_name": self.table_name,
                "column_name": self.default_id_column,
                "column_value": str(id_to_query),
            }
            async with AdamApi(
                base_url=self.base_url,
                auth_username=self.auth_username,
                auth_password=self.auth_password,
                timeout=aiohttp.ClientTimeout(total=60, sock_read=60),
            ) as api:
                res = await api.post(path=self.path_url, data=data)
        except NotFoundInApiError:
            self.failed_query_count += 1
            logger.error(
                f'Id "{id_to_query}" not found in the API. '
                f"Number of failed queries: {self.failed_query_count} "
                f"out of {self.total_query_count}."
            )
            return []
        except Exception:
            # log exception and return empty result
            self.failed_query_count += 1
            logger.error(
                f'Failed to query the api for "{id_to_query}": {format_exc()}. '
                f"Number of failed queries: {self.failed_query_count} "
                f"out of {self.total_query_count}."
            )
            raise
        finally:
            self.total_query_count += 1
        try:
            api_result = self._extract_api_result(res)
        except Exception:
            logger.error(
                f'Failed to extract the api results for "{id_to_query}": {res}.'
            )
            api_result = []

        return api_result

    def _extract_api_result(self, res):
        unpacked_res = []
        for row in res:
            columns_to_unpack = [column[0] for column in self.dataset_class.columns]
            unpacked_row = {}
            for column in columns_to_unpack:
                unpacked_row[column] = row[column.upper()]
            # replace empty strings with None
            unpacked_row = {k: v if v != "" else None for k, v in unpacked_row.items()}
            unpacked_res.append(unpacked_row)
        return unpacked_res


class AzureDatasource(DatasourceBase):
    """Datasource that retrieves data from an Azure dataset.

    It's necessary to override the following class attributes:
        name = The name of the datasource
        dataset_name = The name of the dataset in Azure
        default_id_column = The column used to filter the data in the fetch function

    Example:
        class ExampleByIdDatasource(AzureDatasource):
            name = "exampleclass_by_id"
            dataset_name = "example"
            default_id_column = "id"
    """

    source_type = "azure"
    name: str = None
    dataset_name: str = None
    default_id_column: str = None

    def __init__(self):
        run = Run.get_context()
        self.ws = run.experiment.workspace

    def fetch_all(self, *query_params):
        train_tabular_dataset = Dataset.get_by_name(self.ws, name=self.dataset_name)
        data = train_tabular_dataset.to_pandas_dataframe()
        return data

    def fetch_many(
        self, query_params_list: Sequence[Union[Any, Tuple]]
    ) -> Union[List[Dict[str, Any]], pd.DataFrame]:
        """Fetch records for several arguments at once.

        Depending on the implementation the result may or may not include duplicates
        if the same argument is in `query_params_list` more than once, or if any entries
        result in the same records. It's the callers responsibility to deduplicate the
        result if required.

        Parameters
        ----------
        query_params_list : Sequence(Any) or Sequence(Tuple)
            list of arguments. Can either be a list of scalar values if the source's
            `fetch() function expects a single argument, or a list of tuples if the
            function expects more than one argument.
        """
        if len(query_params_list) == 0:
            logger.warning(
                "query_params_list is an empty list, returning empty dataframe"
            )
            return pd.DataFrame([])

        train_tabular_dataset = Dataset.get_by_name(self.ws, name=self.dataset_name)
        logger.info(f"AzureDatasource {self.dataset_name} retrieved dataset")

        # Cast both query params and ID column (below) to string to avoid matching problems when one is int and the other str.
        query_params_list = [str(p) for p in query_params_list]

        data = train_tabular_dataset.to_pandas_dataframe()
        records = data[
            data[self.default_id_column]
            .astype(str)
            .isin(list(map(str, query_params_list)))
        ]

        return records
