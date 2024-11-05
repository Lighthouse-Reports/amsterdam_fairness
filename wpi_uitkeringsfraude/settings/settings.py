import logging
import os
from typing import Any, Dict

from azure.identity import ManagedIdentityCredential
from azure.keyvault.secrets import SecretClient
from fraude_preventie.settings_helper import GenericSettings, Settings, strings2flags
from opencensus.ext.azure.log_exporter import AzureLogHandler
from pydantic import BaseModel

from wpi_onderzoekswaardigheid_aanvraag.settings.flags import PipelineFlag
from wpi_onderzoekswaardigheid_aanvraag.settings.settings_schema import WPISettingsSpec

logger = logging.getLogger(__name__)


class WPISettings(Settings):
    @classmethod
    def process_value(cls, k, v):
        v = super().process_value(k, v)
        if k == "flags":
            v = strings2flags(v, PipelineFlag)
        return v

    @classmethod
    def set_from_yaml(
        cls, filename: str, spec: BaseModel = WPISettingsSpec
    ) -> "GenericSettings":
        return super().set_from_yaml(filename, spec)

    @classmethod
    def setup_production_settings(
        cls,
        config_file,
    ):
        settings = cls.set_from_yaml(config_file, WPISettingsSpec())

        # Allow setting the API tokens through an environment variable to enable local testing.
        if ("pw_daso_api" in os.environ) and ("api_auth_token" in os.environ):
            WPISettings.default_settings["pw_daso_api"] = os.getenv("pw_daso_api")
            WPISettings.default_settings["api"]["auth_token"] = os.getenv(
                "api_auth_token"
            )
        # Retrieve API tokens from the keyvault when in production.
        else:
            keyVaultName = settings["key_vault_name"]
            KVUri = f"https://{keyVaultName}.vault.azure.net"
            credential = ManagedIdentityCredential()
            secret_client = SecretClient(vault_url=KVUri, credential=credential)
            WPISettings.default_settings["pw_daso_api"] = secret_client.get_secret(
                "model-api-login-to-data-api"
            ).value
            WPISettings.default_settings["api"][
                "auth_token"
            ] = secret_client.get_secret("data-api-login-to-model-api").value


def setup_azure_logging(cfg: Dict[str, Any]):
    """Sets up logging according to the configuration and add AzureLogHandler.

    Parameters
    ----------
    cfg:
        configuration part of the config.yml

    Returns
    -------
    None

    """
    logging.basicConfig(**cfg["basic_config"])
    credential = ManagedIdentityCredential()
    instrumentation_key = f"InstrumentationKey={os.getenv('AI_INSTRUMENTATIONKEY')}"
    azure_log_handler = AzureLogHandler(
        credential=credential, connection_string=instrumentation_key
    )
    for pkg in cfg["own_packages"]:
        logging.getLogger(pkg).setLevel(cfg["loglevel_own"])
        logging.getLogger(pkg).addHandler(azure_log_handler)
    for logger_, level in cfg["extra_loglevels"].items():
        logging.getLogger(logger_).setLevel(level)
        logging.getLogger(logger_).addHandler(azure_log_handler)
