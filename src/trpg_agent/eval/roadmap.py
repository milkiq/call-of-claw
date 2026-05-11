from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from trpg_agent.eval.scorecard import EvalFinding, EvalResult
from trpg_agent.memory.store import SqliteStore


class RoadmapItem(BaseModel):
    id: str
    priority: str
    title: str
    suggested_area: str
    dimensions: list[str] = Field(default_factory=list)
    case_ids: list[str] = Field(default_factory=list)
    acceptance_tests: list[str] = Field(default_factory=list)


class Roadmap(BaseModel):
    source_run_ids: list[str]
    items: list[RoadmapItem]

    def to_markdown(self) -> str:
        lines = ["# Derived TRPG Agent Roadmap", ""]
        if self.source_run_ids:
            lines.append(f"Source runs: {', '.join(self.source_run_ids)}")
            lines.append("")
        for item in self.items:
            lines.extend(
                [
                    f"## {item.priority}: {item.title}",
                    f"- Area: `{item.suggested_area}`",
                    f"- Dimensions: {', '.join(item.dimensions)}",
                    f"- Cases: {', '.join(item.case_ids)}",
                    f"- Acceptance: {'; '.join(item.acceptance_tests)}",
                    "",
                ]
            )
        if not self.items:
            lines.append("No failing findings. Keep expanding coverage.")
        return "\n".join(lines)


def derive_roadmap_from_results(results: list[EvalResult]) -> Roadmap:
    findings = [finding for result in results for finding in result.findings]
    grouped: dict[str, list[EvalFinding]] = defaultdict(list)
    for finding in findings:
        grouped[finding.suggested_area].append(finding)

    items: list[RoadmapItem] = []
    for index, (area, area_findings) in enumerate(
        sorted(grouped.items(), key=lambda entry: (-len(entry[1]), entry[0])),
        start=1,
    ):
        severity_counts = Counter(finding.severity for finding in area_findings)
        if severity_counts["critical"]:
            priority = "P0"
        elif severity_counts["high"]:
            priority = "P1"
        else:
            priority = "P2"
        dimensions = sorted({finding.dimension for finding in area_findings})
        case_ids = sorted({finding.case_id for finding in area_findings})
        items.append(
            RoadmapItem(
                id=f"roadmap-{index}",
                priority=priority,
                title=f"Improve {area} based on {len(area_findings)} finding(s)",
                suggested_area=area,
                dimensions=dimensions,
                case_ids=case_ids,
                acceptance_tests=[f"Make eval case `{case_id}` pass" for case_id in case_ids],
            )
        )
    return Roadmap(source_run_ids=[result.run_id for result in results], items=items)


def derive_roadmap_from_store(store: SqliteStore) -> Roadmap:
    results = [
        EvalResult.model_validate(row["payload"])
        for row in store.list_eval_runs()
        if isinstance(row.get("payload"), dict)
    ]
    return derive_roadmap_from_results(results)


def write_roadmap_yaml(roadmap: Roadmap, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(roadmap.model_dump(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
