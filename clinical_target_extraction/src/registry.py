"""Per-client target registry with stable, Python-assigned identifiers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .schemas import DiscoveredTarget, NewTargetCandidate


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
        return self._add(
            label=target.proposed_label,
            definition=target.definition,
            aliases=target.aliases_in_note,
            session_index=session_index,
        )

    def add_new(self, target: NewTargetCandidate, session_index: int) -> str:
        """Assign the next stable ID to a model-declared genuinely new target."""

        return self._add(
            label=target.proposed_label,
            definition=target.definition,
            aliases=target.aliases_in_note,
            session_index=session_index,
        )

    def _add(
        self,
        *,
        label: str,
        definition: str,
        aliases: list[str],
        session_index: int,
    ) -> str:
        target_id = self._new_id()
        self.targets.append(
            RegistryTarget(
                target_id=target_id,
                canonical_label=label,
                definition=definition,
                aliases=_unique([label, *aliases]),
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
