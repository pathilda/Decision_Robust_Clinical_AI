"""Per-client target registry with stable, Python-assigned identifiers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .schemas import CanonicalizationOutput, DiscoveredTarget, NewTargetCandidate


class RegistryTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: str
    canonical_label: str
    definition: str
    aliases: list[str]
    first_observed_session: int
    status: str = "active"


class TargetRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_id: str
    targets: list[RegistryTarget] = Field(default_factory=list)

    def _new_id(self) -> str:
        return f"T{len(self.targets) + 1:03d}"

    def get(self, target_id: str) -> RegistryTarget:
        for target in self.targets:
            if target.target_id == target_id:
                return target
        raise KeyError(f"Unknown target ID: {target_id}")

    def add_discovered(self, target: DiscoveredTarget, session_index: int = 1) -> str:
        target_id = self._new_id()
        aliases = _unique([target.proposed_label, *target.aliases_in_note])
        self.targets.append(
            RegistryTarget(
                target_id=target_id,
                canonical_label=target.proposed_label,
                definition=target.definition,
                aliases=aliases,
                first_observed_session=session_index,
            )
        )
        return target_id

    def apply_canonicalization(
        self,
        candidate: NewTargetCandidate,
        decision: CanonicalizationOutput,
        session_index: int,
    ) -> str:
        if decision.decision == "same_as_existing":
            target = self.get(decision.matched_target_id or "")
            target.aliases = _unique([*target.aliases, candidate.proposed_label])
            return target.target_id

        target_id = self._new_id()
        self.targets.append(
            RegistryTarget(
                target_id=target_id,
                canonical_label=decision.recommended_label,
                definition=decision.recommended_definition,
                aliases=_unique([candidate.proposed_label]),
                first_observed_session=session_index,
            )
        )
        return target_id


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result
