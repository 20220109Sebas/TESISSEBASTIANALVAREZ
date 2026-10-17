"""Recheck the unchanged reference/inference gates with the new runtime lock.

The six old fixture tests intentionally still reject the old runtime checksum.
These tests do not rewrite that fixture or relax any hash check.
"""
from pathlib import Path

import pytest

from scripts.run_document_experience_evaluation_v1 import validate_design

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def design():
    directory = ROOT / "artifacts/document_validation_integrated_lock_v3"
    validate_design(directory)
    return directory


def test_historical_runtime_is_still_rejected_not_silently_relabelled():
    with pytest.raises(ValueError, match="python3.12"):
        validate_design(ROOT / "artifacts/document_experience_evaluation_design_lock_v1")


def test_reference_first_roundtrip_with_new_runtime(design, tmp_path, monkeypatch):
    from test_document_experience_evaluation_v1 import test_artifact_workflow_reference_first_reproduces_and_model_gets_sources_only
    test_artifact_workflow_reference_first_reproduces_and_model_gets_sources_only(design, tmp_path, monkeypatch)


def test_missing_reference_still_blocks_with_new_runtime(design, tmp_path, monkeypatch):
    from test_document_experience_evaluation_v1 import test_inference_without_reference_artifact_fails_before_model
    test_inference_without_reference_artifact_fails_before_model(design, tmp_path, monkeypatch)


def test_forged_manifest_still_rejected_with_new_runtime(design, tmp_path):
    from test_document_experience_evaluation_v1 import test_self_reported_binding_without_manifest_prerequisite_is_rejected
    test_self_reported_binding_without_manifest_prerequisite_is_rejected(design, tmp_path)


def test_observed_case_still_rejected_with_new_runtime(design, tmp_path):
    from test_document_experience_evaluation_v1 import test_previously_observed_case_cannot_become_new_validation_source
    test_previously_observed_case_cannot_become_new_validation_source(design, tmp_path)


def test_old_identity_conflict_still_stops_without_metrics(design, tmp_path):
    from test_document_experience_evaluation_v1 import test_observed_source_adapter_regression_stops_without_new_metrics
    test_observed_source_adapter_regression_stops_without_new_metrics(design, tmp_path)
