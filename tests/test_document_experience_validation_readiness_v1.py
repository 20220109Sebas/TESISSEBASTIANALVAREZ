import json
from pathlib import Path

import pytest

from scripts.prepare_document_experience_validation_readiness_v1 import collect_exclusions, known_rows

ROOT = Path(__file__).resolve().parents[1]


def test_excludes_unavailable_cases_and_all_previous_selected_and_excluded_ids():
    registry = collect_exclusions([
        ("selection", {"selected_opportunities": [{"opportunity_id": "ocds-old-selected"}],
                       "excluded_ocids": ["ocds-prior-development"]}),
        ("access", [{"opportunity_id": "ocds-failed", "status": "network_error"}]),
        ("substitute", {"pending_sources": [{"opportunity_id": "ocds-substitute"}]}),
    ])
    assert registry["excluded_ocids"] == ["ocds-failed", "ocds-old-selected", "ocds-prior-development", "ocds-substitute"]
    assert registry["known_pdf_sha256"] == []


def test_pdf_hash_exclusion_independent_of_ocid_and_with_all_lineage():
    registry = collect_exclusions([
        ("access-b", [{"opportunity_id": "ocds-b", "status": "downloaded_pdf", "sha256": "a" * 64}]),
        ("access-a", [{"opportunity_id": "ocds-a", "status": "downloaded_pdf", "sha256": "a" * 64}]),
    ])
    assert len(registry["excluded_ocids"]) == 2
    assert registry["known_pdf_sha256"] == ["a" * 64]
    assert registry["pdf_provenance"]["a" * 64] == ["access-a", "access-b"]


@pytest.mark.parametrize("payload", [
    [{"opportunity_id": "invalid"}],
    [{"opportunity_id": "ocds-a", "status": "downloaded_pdf", "sha256": "bad"}],
    [{"opportunity_id": "ocds-a", "status": "downloaded_pdf"}],
])
def test_invalid_exclusion_identity_or_pdf_hash_rejected(payload):
    with pytest.raises(ValueError):
        collect_exclusions([("source", payload)])


def test_unknown_schema_rejected():
    with pytest.raises(ValueError):
        known_rows({"unrecognized": []})


def test_readiness_snapshot_keeps_old_50_out_and_does_not_open_new_content(tmp_path):
    from scripts.prepare_document_experience_validation_readiness_v1 import build
    output = tmp_path / "readiness"
    summary = build(output)
    registry = json.loads((output / "exclusion_registry.json").read_text())
    old = json.loads((ROOT / "artifacts/experience_holdout_plan_v1/selection.json").read_text())
    assert len(old["selected_opportunities"]) == 50
    assert {r["opportunity_id"] for r in old["selected_opportunities"]} <= set(registry["excluded_ocids"])
    for key in ("ready_for_new_evaluation", "new_sample_selected", "new_source_content_inspected", "pdf_downloads_performed", "metrics_computed"):
        assert summary[key] is False
    assert len(summary["pending_gates"]) == 5
    replay = tmp_path / "replay"
    build(replay)
    for path in output.iterdir():
        assert path.read_bytes() == (replay / path.name).read_bytes()
    with pytest.raises(ValueError):
        build(output)
