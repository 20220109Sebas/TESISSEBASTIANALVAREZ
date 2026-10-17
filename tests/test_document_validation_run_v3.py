"""Test the two legitimate metadata partitions without reading new PDF text."""
import json
from pathlib import Path

import pytest

from scripts import document_validation_run_v3 as run
from scripts.register_browser_experience_sources import dump


@pytest.mark.parametrize("relative", ["data/raw/oece/contrast_candidate_versions_v1/synthetic.json", "data/raw/oece/initial_releases/synthetic.json"])
def test_both_frozen_partitions_are_accepted(relative):
    assert run.initial_source_path(relative) == run.ROOT / relative


@pytest.mark.parametrize("relative", ["/tmp/escape.json", "data/raw/oece/other/synthetic.json",
    "data/raw/oece/initial_releases/../../escape.json"])
def test_other_partitions_and_traversal_rejected(relative):
    with pytest.raises(ValueError):
        run.initial_source_path(relative)


@pytest.fixture(scope="module")
def integrated(tmp_path_factory):
    from scripts.freeze_document_source_design_v1 import freeze
    base = tmp_path_factory.mktemp("v3-metadata")
    source, design = base / "source", base / "design"
    freeze(source)
    run.freeze(source, design)
    return source, design


def test_revision_keeps_v2_code_and_seed_but_pins_v3_too(integrated):
    _, design = integrated
    lock = run.validate_run(design)
    assert lock["integration_revision"] == 3
    assert lock["seed"] == "20261014-document-experience-validation-v2"
    assert str(run.ROOT / "scripts/document_validation_run_v3.py") in lock["code_sha256"]


def test_revision_reproduces_before_selection(integrated, tmp_path):
    source, design = integrated
    output = tmp_path / "replay"
    run.freeze(source, output)
    manifest = json.loads((design / "manifest.json").read_text())
    for name in (*manifest["output_sha256"], "manifest.json"):
        assert (design / name).read_bytes() == (output / name).read_bytes()


def test_audit_rejects_metadata_changes_with_synthetic_inputs(tmp_path, monkeypatch):
    design, selected = tmp_path / "design", tmp_path / "selection"
    design.mkdir()
    dump(design / "manifest.json", {"test": "synthetic"})
    samples = [{"sample_number": n+1, "opportunity_id": f"ocds-synthetic-{n}", "status": "no_eligible_initial_pdf"} for n in range(50)]
    rows = [{"opportunity_id": s["opportunity_id"], "object_type": "services", "publication_date": "2025-01-01",
             "title": "Servicio", "description": "Servicio general"} for s in samples]
    run_info = {"source_design_manifest_sha256": "a" * 64}
    registry = {"excluded_ocids": [], "known_pdf_sha256": []}
    monkeypatch.setattr(run, "metadata", lambda *_: (run_info, registry, rows, samples, [], [], 50))
    from scripts.register_browser_experience_sources import sha
    selection = {"integrated_design_manifest_sha256": sha(design / "manifest.json"),
        "source_design_manifest_sha256": "a" * 64, "excluded_ocids": [], "known_development_pdf_sha256": [],
        "selected_opportunities": list(reversed(samples))}
    run.prior.write_artifact(selected, {"selection.json": selection, "summary.json": {}}, {})
    with pytest.raises(ValueError, match="independent"):
        run.audit_selection(design, selected, tmp_path / "audit")
