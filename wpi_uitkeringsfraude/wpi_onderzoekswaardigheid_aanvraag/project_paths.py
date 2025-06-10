from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "wpi_onderzoekswaardigheid_aanvraag"

ANALYSIS_PATH = PROJECT_ROOT / "analysis"
ARTIFACT_PATH = PACKAGE_ROOT / "resources"
CONFIG_PATH = PROJECT_ROOT / "dev-config.yml"
DATA_PATH = PROJECT_ROOT / "data"
INFO_PATH = PROJECT_ROOT / "info"
