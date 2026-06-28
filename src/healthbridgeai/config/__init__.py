"""Config package — re-exports settings singleton and DiseaseRegistry loader."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .settings import Settings, settings

__all__ = ["settings", "Settings", "get_disease_registry"]

_DISEASES_YAML = Path(__file__).parent / "diseases.yaml"


@lru_cache(maxsize=1)
def get_disease_registry():
    """Return the process-wide DiseaseRegistry loaded from diseases.yaml."""
    from healthbridgeai.core.models.disease import DiseaseRegistry  # noqa: PLC0415
    return DiseaseRegistry.from_yaml(_DISEASES_YAML)
