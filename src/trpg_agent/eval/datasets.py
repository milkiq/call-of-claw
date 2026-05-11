from __future__ import annotations

from collections.abc import Iterable

from langsmith import Client

from trpg_agent.eval.cases import EvalCase


def sync_cases_to_langsmith_dataset(
    *,
    client: Client,
    dataset_name: str,
    cases: Iterable[EvalCase],
) -> str:
    """Create a LangSmith dataset from local eval cases.

    This function is intentionally opt-in; local development and CI can run without credentials.
    """

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="TRPG Agent evaluation cases synced from local YAML fixtures.",
    )
    for case in cases:
        client.create_example(
            inputs={"input": case.input or "", "kind": case.kind},
            outputs={"expectation": case.expectation.model_dump()},
            metadata={"case_id": case.id, "tags": case.tags},
            dataset_id=dataset.id,
        )
    return str(dataset.id)
