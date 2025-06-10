from typing import Dict, List

from fraude_preventie.settings.settings_schema import SettingsSpec, SettingsSpecModel


class ReweighFeatureSpec(SettingsSpecModel):
    groups: List = []
    drop_after_use: bool = True


class TrainingSpec(SettingsSpecModel):
    algorithm: str = None
    handling_types: List[str] = None
    core_product_numbers: List[int] = None
    feature_selection: str = None
    fimp_threshold: float = None
    register_model: bool = None


class DataAPISpec(SettingsSpecModel):
    url: str = None


class APISpec(SettingsSpecModel):
    host: str = None
    port: int = None


class WPISettingsSpec(SettingsSpec):
    reweigh_features: Dict[str, ReweighFeatureSpec] = {}
    sensitive_features: Dict[str, List] = {}
    model: TrainingSpec = None
    key_vault_name: str = "{KEY_VAULT_NAME}"
    data_api: DataAPISpec = None
    api: APISpec = None
