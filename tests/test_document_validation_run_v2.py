"""Selection integration tests use synthetic metadata only, never new PDF text."""
import copy
import json
from pathlib import Path

import pytest

from scripts.document_validation_run_v2 import selected_metadata, runtime_transition
from scripts.register_browser_experience_sources import dump, sha

ROOT = Path(__file__).resolve().parents[1]
URL = "https://prod1.seace.gob.pe/SeaceWeb-PRO/SdescargarArchivoAlfresco?fileCode=12345678-1234-1234-1234-123456789abc"


def inputs():
    rows = [{"opportunity_id": f"ocds-selection-synthetic-{n}", "object_type": "services",
             "publication_date": "2025-01-01", "title": "Servicio", "description": "Servicio general"} for n in range(70)]
    availability = [{"ocid": r["opportunity_id"], "text_published_at": "2025-01-01T12:00:00-05:00",
                     "sha256": "a" * 64, "source_path": "synthetic-only.json"} for r in rows]
    def load(meta):
        return {"releases": [{"ocid": meta["ocid"], "publishedDate": meta["text_published_at"],
            "tender": {"documents": [{"id": "one", "title": "Bases Administrativas", "documentType": "biddingDocuments",
                "format": "pdf", "url": URL, "datePublished": meta["text_published_at"]}]},
            "awards": [{"future_label": "must not enter selection"}]}]}, Path("/tmp/synthetic-only.json")
    return rows, availability, load


def test_exact_fifty_cases_no_replacement_for_duplicate_pdf_url():
    rows, availability, load = inputs()
    samples, inventory, queue, universe = selected_metadata(rows, availability, [], load)
    assert len(samples) == 50 and universe == 70 and len(queue) == 1 and len(inventory) == 50
    assert samples[0]["status"] == "eligible_initial_pdf_selected"
    assert all(s["status"] == "duplicate_initial_document_url_no_replacement" for s in samples[1:])
    assert [s["sample_number"] for s in samples] == list(range(1, 51))


def test_shuffle_and_future_awards_do_not_change_selection():
    rows, availability, load = inputs()
    before = selected_metadata(rows, availability, [], load)
    for row in rows:
        row.update(technical_text="Experiencia", awarded_supplier_id="future-winner")
    assert selected_metadata(list(reversed(rows)), list(reversed(availability)), [], load) == before


def test_all_previously_observed_ids_are_excluded_including_inaccessible():
    rows, availability, load = inputs()
    excluded = [r["opportunity_id"] for r in rows[:10]]
    samples, _, _, universe = selected_metadata(rows, availability, excluded, load)
    assert universe == 60 and not {s["opportunity_id"] for s in samples} & set(excluded)


@pytest.mark.parametrize("change", ["duplicate", "missing", "insufficient", "wrong_release", "wrong_date", "multiple_releases"])
def test_invalid_join_and_initial_release_stop_before_access(change):
    rows, availability, load = inputs()
    if change == "duplicate":
        availability += copy.deepcopy(availability)
    elif change == "missing":
        availability = []
    elif change == "insufficient":
        rows = rows[:49]
    original = load
    def changed(meta):
        package, path = original(meta)
        if change == "wrong_release":
            package["releases"][0]["ocid"] = "ocds-other"
        elif change == "wrong_date":
            package["releases"][0]["publishedDate"] = "2026-01-01T00:00:00Z"
        elif change == "multiple_releases":
            package["releases"] *= 2
        return package, path
    with pytest.raises(ValueError):
        selected_metadata(rows, availability, [], changed)


def test_missing_document_does_not_replace_sample():
    rows, availability, load = inputs()
    def no_docs(meta):
        package, path = load(meta)
        package["releases"][0]["tender"]["documents"] = []
        return package, path
    samples, inventory, queue, _ = selected_metadata(rows, availability, [], no_docs)
    assert len(samples) == 50 and not inventory and not queue
    assert all(s["status"] == "no_eligible_initial_pdf" for s in samples)


def test_runtime_transition_never_ignores_changed_code_or_data(tmp_path):
    runtime, code = tmp_path / "python", tmp_path / "code"
    dump(runtime, {"binary": "before"})
    dump(code, {"code": "same"})
    manifest = {"input_sha256": {str(runtime): sha(runtime), str(code): sha(code)}, "output_sha256": {}}
    old = tmp_path / "old"
    old.mkdir()
    dump(old / "manifest.json", manifest)
    dump(runtime, {"binary": "after"})
    transition = runtime_transition(old, {"python_executable": str(runtime)})
    assert len(transition["observed_changes"]) == 1 and transition["before_new_sample_selection"]
    dump(code, {"code": "changed"})
    with pytest.raises(ValueError, match="Unapproved"):
        runtime_transition(old, {"python_executable": str(runtime)})


@pytest.fixture(scope="module")
def integrated(tmp_path_factory):
    from scripts.freeze_document_source_design_v1 import freeze as freeze_source
    from scripts.document_validation_run_v2 import freeze
    base = tmp_path_factory.mktemp("integrated-run")
    source = base / "source-design"
    freeze_source(source)
    directory = base / "run"
    freeze(source, directory)
    return source, directory


def test_integrated_design_is_frozen_before_real_sample_and_legacy_compatible(integrated):
    from scripts.document_validation_run_v2 import validate_run
    source, directory = integrated
    run = validate_run(directory, before_selection=True)
    assert run["source_design_path"] == str(source)
    assert run["ready_to_select_new_evaluation_sample"] and not run["new_sample_selected"]
    assert not run["ready_to_compute_metrics"]
    summary = json.loads((directory / "summary.json").read_text())
    assert not summary["source_rules_changed"] and not summary["extractor_changed"]


def test_integrated_freeze_reproduction_and_no_overwrite(integrated, tmp_path):
    from scripts.document_validation_run_v2 import freeze
    source, directory = integrated
    replay = tmp_path / "replay"
    freeze(source, replay)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in (*manifest["output_sha256"], "manifest.json"):
        assert (directory / name).read_bytes() == (replay / name).read_bytes()
    with pytest.raises(ValueError):
        freeze(source, replay)


def test_source_package_cannot_escape_frozen_inventory():
    from scripts.document_validation_run_v2 import initial_source_path
    with pytest.raises(ValueError):
        initial_source_path("/tmp/not-a-frozen-initial-release.json")
    with pytest.raises(ValueError):
        initial_source_path("data/raw/oece/contrast_candidate_versions_v1/../../../../outside.json")


def test_reference_gate_requires_audit_manifest_of_this_sample(integrated, tmp_path):
    from scripts.document_validation_run_v2 import source_gate, write_artifact
    _, directory = integrated
    selection, audit = tmp_path / "selection", tmp_path / "audit"
    write_artifact(selection, {"summary.json": {}}, {})
    write_artifact(audit, {"summary.json": {"fixed_order_matches": True}}, {})
    with pytest.raises(ValueError, match="exact sample"):
        source_gate(directory, selection, audit, tmp_path / "no-source", tmp_path / "output")
