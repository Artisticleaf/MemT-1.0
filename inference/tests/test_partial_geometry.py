"""Tail boundaries and progress reservations derive from model write events."""

from types import SimpleNamespace

import pytest

from memt_sglang.runtime.features import validate_query_chunks
from memt_sglang.runtime.partial_scheduler import fragment_tokens, partial_swa_floor


@pytest.mark.parametrize("event", [512, 2048, 4096])
def test_tail_preserves_model_write_boundaries(event):
    assert (
        fragment_tokens("tail", 2 * event + 17, 0, event, chunk_tokens=event) == event
    )
    assert (
        fragment_tokens("tail", 2 * event + 17, event, event - 1, chunk_tokens=event)
        == 0
    )
    assert (
        fragment_tokens("tail", 2 * event + 17, 2 * event, 9, chunk_tokens=event) == 9
    )
    assert (
        partial_swa_floor("tail", 3, 128, 64, 99, chunk_tokens=event)
        == 3 * 256 + event + 99
    )


@pytest.mark.parametrize("event", [512, 4096])
def test_tail_admission_uses_resolved_event(monkeypatch, event):
    monkeypatch.setenv("MEMT_EXPERIMENT_COMMIT_CHUNK_POLICY", "tail")
    monkeypatch.setenv("MEMT_EXPERIMENT_OVERLAP", "0")
    monkeypatch.setenv("MEMT_EXPERIMENT_DECODE_ONLY_OVERLAP", "0")
    monkeypatch.setenv("MEMT_EXPERIMENT_PREFILL_OVERLAP", "0")
    monkeypatch.setenv("MEMT_EXPERIMENT_DECODE_OVERLAP", "0")
    args = SimpleNamespace(
        chunked_prefill_size=event,
        disable_overlap_schedule=True,
        enable_dynamic_chunking=False,
        schedule_policy="fcfs",
    )
    validate_query_chunks(args, chunk_tokens=event)
    args.chunked_prefill_size = event - 1
    with pytest.raises(ValueError, match="native token budget"):
        validate_query_chunks(args, chunk_tokens=event)
