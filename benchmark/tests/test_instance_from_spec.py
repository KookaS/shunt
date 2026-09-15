"""Instance resolution is spec-derived, never a HuggingFace dataset load.

A spec already carries the problem statement and image ref, so resolving a live cell's
instance must not call ``datasets.load_dataset`` — a network dependency that only ever
covered Verified and silently skipped every other source.
"""

from __future__ import annotations

from typing import Any

import pytest

from benchmark.runner import infer, swebench_specs


def _spec(**overrides: Any) -> swebench_specs.SwebenchSpec:
    fields: dict[str, Any] = {
        "instance_id": "psf__requests-1142",
        "repo": "psf/requests",
        "base_commit": "abc123",
        "version": "4.0",
        "difficulty_stratum": "medium",
        "fail_to_pass": ["test_x"],
        "pass_to_pass": ["test_y"],
        "image_ref": "swebench/x:latest",
        "dataset_revision": "rev",
        "problem_statement": "Fix the thing.",
    }
    fields.update(overrides)
    return swebench_specs.SwebenchSpec(**fields)


def test_instance_is_built_from_the_spec_fields() -> None:
    assert infer._instance_from_spec(_spec()) == {
        "instance_id": "psf__requests-1142",
        "problem_statement": "Fix the thing.",
        "image_name": "swebench/x:latest",  # the prebuilt Docker image, not multimodal assets
    }


def test_empty_problem_statement_is_refused() -> None:
    # A spec written before `problem_statement` existed would hand the agent an empty task.
    with pytest.raises(ValueError, match="problem_statement"):
        infer._instance_from_spec(_spec(problem_statement=""))


def test_resolution_never_touches_the_hf_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("instance resolution called datasets.load_dataset")

    monkeypatch.setattr(datasets, "load_dataset", _forbidden)
    instance = infer._instance_from_spec(_spec())
    assert instance["instance_id"] == "psf__requests-1142"
