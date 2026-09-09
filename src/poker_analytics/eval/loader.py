"""Loads Scenario objects from the YAML files under eval/scenarios/."""
from __future__ import annotations

from pathlib import Path

import yaml

from .schema import Scenario


def load_scenarios(scenarios_dir: Path | str) -> list[Scenario]:
    """Load every *.yaml file in scenarios_dir into a flat, id-sorted list of Scenarios.

    Raises ValueError on a duplicate id across files - each scenario must be independently
    addressable in a report regardless of which file it lives in.
    """
    scenarios_dir = Path(scenarios_dir)
    scenarios: list[Scenario] = []
    seen_ids: set[str] = set()

    for path in sorted(scenarios_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text()) or []
        for entry in raw:
            scenario = Scenario(
                id=entry["id"],
                category=entry["category"],
                prompt=entry["prompt"],
                expected_tool=entry.get("expected_tool"),
                expected_args=entry.get("expected_args", {}) or {},
                unexpected_args_absent=tuple(entry.get("unexpected_args_absent", []) or []),
                notes=entry.get("notes", ""),
            )
            if scenario.id in seen_ids:
                raise ValueError(f"duplicate scenario id {scenario.id!r} in {path}")
            seen_ids.add(scenario.id)
            scenarios.append(scenario)

    return sorted(scenarios, key=lambda s: s.id)
