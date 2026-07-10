"""Dataset lifecycle coverage for source and paired pipeline state."""

from __future__ import annotations

from types import SimpleNamespace

from davinci_monet.pipeline.lifecycle import PipelineResourcePolicy
from davinci_monet.pipeline.stages.base import PipelineContext


class _Closable:
    def __init__(self) -> None:
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


def test_cleanup_closes_sources_and_pairs_once_by_dataset_identity() -> None:
    shared = _Closable()
    paired_only = _Closable()
    context = PipelineContext(
        sources={"source": SimpleNamespace(data=shared)},
        paired={
            "shared": SimpleNamespace(data=shared),
            "unique": SimpleNamespace(data=paired_only),
        },
    )

    PipelineResourcePolicy().cleanup_context_datasets(context)

    assert shared.close_count == 1
    assert paired_only.close_count == 1
