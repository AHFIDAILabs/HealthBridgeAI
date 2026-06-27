from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class QueryIntent(str, Enum):
    SYMPTOMS = "symptoms"
    TREATMENT = "treatment"
    PREVENTION = "prevention"
    COMPLICATIONS = "complications"
    DRUG_INTERACTION = "drug_interaction"
    TRANSMISSION = "transmission"
    TESTING = "testing"
    STATISTICS = "statistics"
    DEFINITION = "definition"
    GENERAL = "general"


class RouteResult(BaseModel):
    """Output of the two-dimensional router: which disease(s) + what the user wants."""

    disease_ids: list[str] = Field(default_factory=list)
    disease_confidence: float = 0.0
    query_intent: QueryIntent = QueryIntent.GENERAL
    intent_confidence: float = 0.0
    is_general_health: bool = False   # No specific disease — answer from all namespaces
    is_emergency: bool = False        # Pipeline must abort; send emergency response
    is_personal: bool = False         # "I have…" / "my symptoms" — never cache

    @field_validator("disease_confidence", "intent_confidence")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class DiseaseConfig(BaseModel):
    """Per-disease configuration loaded from diseases.yaml."""

    name: str
    short_name: str
    enabled: bool = False
    pinecone_namespace: str
    kb_gcs_path: str
    chunk_size_hint: int = 400
    min_retrieval_score: float = 0.6
    search_domains: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)
    emergency_keywords: list[str] = Field(default_factory=list)
    system_prompt_extra: str = ""


class DiseaseRegistry:
    """
    In-memory registry loaded once at startup from diseases.yaml.
    Provides O(1) alias → disease_id lookup and disease_id → config lookup.
    """

    def __init__(self, configs: dict[str, DiseaseConfig]) -> None:
        self._configs = configs
        self._alias_map: dict[str, str] = {}
        for disease_id, cfg in configs.items():
            for alias in cfg.aliases:
                self._alias_map[alias.lower()] = disease_id
            self._alias_map[disease_id.lower()] = disease_id

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DiseaseRegistry":
        import yaml
        from pathlib import Path

        raw = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
        configs = {
            disease_id: DiseaseConfig(**cfg)
            for disease_id, cfg in raw["diseases"].items()
        }
        return cls(configs)

    def get(self, disease_id: str) -> Optional[DiseaseConfig]:
        return self._configs.get(disease_id)

    def resolve_alias(self, text: str) -> list[str]:
        """Return disease_ids whose aliases appear in text (lowercased). Fast O(n) scan."""
        text_lower = text.lower()
        found: set[str] = set()
        for alias, disease_id in self._alias_map.items():
            if alias in text_lower and self._configs[disease_id].enabled:
                found.add(disease_id)
        return sorted(found)

    def enabled_ids(self) -> list[str]:
        return [did for did, cfg in self._configs.items() if cfg.enabled]

    def all_ids(self) -> list[str]:
        return list(self._configs.keys())
