import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tesis_recommender.document_sampling_v1 import choose_document, select_ids
from tesis_recommender.document_source_preparation_v1 import (
    MAX_PAGE_CHARS, check_ocr, combine_batches, compose_documents, merge_page, prepare_native, raster_sizes, text_sha,
)

ROOT = Path(__file__).resolve().parents[1]
URL = "https://prod1.seace.gob.pe/SeaceWeb-PRO/SdescargarArchivoAlfresco?fileCode=12345678-1234-1234-1234-123456789abc"
OS_VERSION = "Version 26.2 (Build 25C56)"


class Object(dict):
    def get_object(self):
        return self


class Page(Object):
    def __init__(self, text, resources=None):
        super().__init__({"/Resources": resources})
        self.text = text

    def extract_text(self):
        return self.text


def native(*pages, encrypted=False):
    reader = SimpleNamespace(is_encrypted=encrypted, pages=list(pages))
    source = {"opportunity_id": "ocds-synthetic", "sha256": "a" * 64, "status": "downloaded_pdf", "bytes": 123}
    return prepare_native(reader, source)


def overlay(page=1, text="OCR text", status="ok"):
    return {"opportunity_id": "ocds-synthetic", "source_pdf_sha256": "a" * 64, "page": page,
            "status": status, "text": text, "text_sha256": text_sha(text),
            "ocr_json_path": "synthetic-only.json", "ocr_json_sha256": "b" * 64}


def raw_ocr():
    return {"engine": "Apple Vision VNRecognizeTextRequest", "revision": 3, "languages": ["es-ES", "en-US"],
            "languageCorrection": False, "osVersion": OS_VERSION, "page": 1,
            "text": "OCR text", "lines": [{"text": "OCR text", "confidence": 0.8}]}


@pytest.mark.parametrize("words,selected", [(0, True), (49, True), (50, False), (51, False)])
def test_sparse_native_screen_is_fixed_without_experience_detector(words, selected):
    doc = native(Page(" ".join(["word"] * words)))
    assert doc["pages"][0]["selected_for_ocr"] is selected
    modified = native(Page(" ".join(["experiencia"] * words)))
    assert modified["pages"][0]["selected_for_ocr"] is selected


@pytest.mark.parametrize("width,height,selected", [(600, 600, True), (599, 900, False), (900, 599, False), (1200, 900, True)])
def test_image_screen_on_dense_native_page(width, height, selected):
    resources = Object({"/XObject": Object({"image": Object({"/Subtype": "/Image", "/Width": width, "/Height": height})})})
    doc = native(Page("word " * 60, resources))
    assert doc["pages"][0]["selected_for_ocr"] is selected


def test_nested_forms_cycles_and_metadata_only_image_screen():
    image = Object({"/Subtype": "/Image", "/Width": 700, "/Height": 800})
    resources = Object({"/XObject": Object({"image": image})})
    form = Object({"/Subtype": "/Form", "/Resources": resources})
    resources["/XObject"]["cycle"] = form
    assert raster_sizes(resources) == [[700, 800]]
    for _ in range(10):
        resources = Object({"/XObject": Object({"form": Object({"/Subtype": "/Form", "/Resources": resources})})})
    with pytest.raises(ValueError):
        raster_sizes(resources)


def test_no_partial_text_for_encrypted_empty_oversized_or_text_over_limit():
    assert native(Page("text"), encrypted=True)["status"] == "encrypted_pdf_not_evaluable"
    assert native()["status"] == "physical_page_count_outside_limit"
    assert native(*([Page("text")] * 121))["pages"] == []
    assert native(Page("x" * (MAX_PAGE_CHARS + 1)))["status"] == "native_text_size_outside_limit"


def test_native_order_hashes_and_nonselected_pages_preserved():
    doc = native(Page("word " * 50), Page("text"))
    assert [p["page"] for p in doc["pages"]] == [1, 2]
    page = merge_page(doc["pages"][0])
    assert page["text"] == page["native_text"] == "word " * 50
    assert not page["ocr_selected"] and not page["selected_ocr_unresolved"]


@pytest.mark.parametrize("status", ["empty_ocr", "ocr_error", "not_attempted_after_error_stop"])
def test_ocr_failures_remain_native_with_visible_gap(status):
    doc = native(Page("native"))
    page = compose_documents([doc], [overlay(status=status)])[0]["pages"][0]
    assert page["text"] == page["native_text"] == "native"
    assert not page["ocr_used"] and page["selected_ocr_unresolved"]
    assert not page["absence_of_requirements_proven"]


def test_successful_overlay_keeps_native_and_json_lineage():
    doc = native(Page("native"))
    before = copy.deepcopy(doc)
    result = compose_documents([doc], [overlay()])
    page = result[0]["pages"][0]
    assert page["text"] == "OCR text" and page["native_text"] == "native"
    assert page["ocr_used"] and page["ocr_json_sha256"] == "b" * 64
    assert doc == before and not result[0]["all_document_content_proven_recovered"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "unplanned", "pdf", "hash", "lineage"])
def test_bad_overlay_ledger_rejected(change):
    doc = native(Page("native"))
    rows = [overlay()]
    if change == "missing":
        rows = []
    elif change == "duplicate":
        rows.append(overlay())
    elif change == "unplanned":
        rows[0]["page"] = 2
    elif change == "pdf":
        rows[0]["source_pdf_sha256"] = "c" * 64
    elif change == "hash":
        rows[0]["text_sha256"] = "c" * 64
    else:
        rows[0]["ocr_json_sha256"] = "bad"
    with pytest.raises(ValueError):
        compose_documents([doc], rows)


@pytest.mark.parametrize("field,value", [("revision", 2), ("languages", ["es-ES"]),
    ("languageCorrection", True), ("osVersion", "different"), ("page", True), ("engine", "other"),
    ("text", "different"), ("lines", [{"text": 12}])])
def test_ocr_engine_runtime_and_assembly_must_match(field, value):
    raw = raw_ocr()
    raw[field] = value
    with pytest.raises(ValueError):
        check_ocr(raw, 1, OS_VERSION)


def test_valid_ocr_metadata_and_no_unplanned_overlay():
    assert check_ocr(raw_ocr(), 1, OS_VERSION)
    doc = native(Page("word " * 60))
    with pytest.raises(ValueError):
        merge_page(doc["pages"][0], overlay())


def candidate_rows(count=80):
    return [{"opportunity_id": f"ocds-sample-{n}", "object_type": "services", "publication_date": "2025-03-01",
             "title": "Servicio", "description": "Descripción"} for n in range(count)]


def test_sampling_is_deterministic_excludes_observed_and_ignores_outcomes():
    rows = candidate_rows()
    selected, universe = select_ids(rows, {"ocds-sample-1"})
    assert len(selected) == 50 and universe == 79 and "ocds-sample-1" not in selected
    for row in rows:
        row.update(awards="future outcome", expected_experience="irrelevant", technical_text="not an eligibility detector")
    assert select_ids(list(reversed(rows)), {"ocds-sample-1"}) == (selected, universe)


def test_sampling_scope_and_insufficient_pool_not_silently_replaced():
    rows = candidate_rows()
    rows[0]["title"] = "Consultoría de obra"
    rows[1]["description"] = "CONSULTORIA EN OBRA"
    rows[2]["object_type"] = "goods"
    rows[3]["publication_date"] = "2026-01-01"
    selected, universe = select_ids(rows, set())
    assert universe == 76 and not set(selected) & {f"ocds-sample-{n}" for n in range(4)}
    with pytest.raises(ValueError):
        select_ids(rows[:49], set())
    with pytest.raises(ValueError):
        select_ids(rows + [rows[0]], set())


def source_doc(title="Bases Administrativas", date="2025-01-01T12:00:00-05:00", url=URL):
    return {"title": title, "documentType": "biddingDocuments", "format": "pdf", "datePublished": date, "url": url}


def test_fixed_document_priority_date_url_and_no_duplicate_url_fallback():
    docs = [source_doc("Bases Integradas"), source_doc(), source_doc("Términos de Referencia")]
    chosen, status = choose_document(docs, "2025-01-02T12:00:00-05:00", set())
    assert status == "eligible_initial_pdf_selected" and chosen["document_index"] == 1
    assert choose_document(docs, "2025-01-02T12:00:00-05:00", {URL}) == (None, "duplicate_initial_document_url_no_replacement")
    excluded = [source_doc("Propuesta"), source_doc(date="2026-01-01T00:00:00Z"), source_doc(url="https://example.com/file.pdf")]
    assert choose_document(excluded, "2025-01-02T12:00:00-05:00", set()) == (None, "no_eligible_initial_pdf")


@pytest.fixture(scope="module")
def design_lock(tmp_path_factory):
    from scripts.freeze_document_source_design_v1 import freeze
    output = tmp_path_factory.mktemp("source-design") / "lock"
    return output, freeze(output)


def test_design_lock_runtime_exclusions_and_open_gates(design_lock):
    from scripts.freeze_document_source_design_v1 import validate_design
    output, summary = design_lock
    lock = validate_design(output, before_selection=True)
    assert summary["sampling_design_locked"] and summary["source_preparation_code_locked"]
    assert summary["excluded_ocids"] == 100 and summary["known_pdf_hashes"] == 52
    assert summary["pypdf_version"] == "6.10.0"
    assert not lock["ready_to_select_new_evaluation_sample"] and not lock["comparison_adapter_locked"]


def test_design_snapshot_reproduces_and_prevents_overwrite(design_lock, tmp_path):
    from scripts.freeze_document_source_design_v1 import freeze
    original, _ = design_lock
    output = tmp_path / "replay"
    freeze(output)
    for path in original.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (output / path.relative_to(original)).read_bytes()
    with pytest.raises(ValueError):
        freeze(output)


def test_source_runtime_change_rejected(design_lock, monkeypatch):
    import scripts.freeze_document_source_design_v1 as module
    output, _ = design_lock
    real = module.source_runtime()
    monkeypatch.setattr(module, "source_runtime", lambda: dict(real, pypdf_version="different"))
    with pytest.raises(ValueError, match="runtime changed"):
        module.validate_design(output)


def access_fixture(tmp_path, design_lock):
    from scripts.register_browser_experience_sources import dump, sha
    lock_dir, _ = design_lock
    lock = json.loads((lock_dir / "source_design_lock.json").read_text())
    directory = tmp_path / "access"
    directory.mkdir()
    # Not a real PDF or authored document: native-worker tests mock its parser.
    blob = directory / "mock-download.bin"
    blob.write_bytes(b"synthetic binary test fixture, not a PDF")
    selected = {"opportunity_id": "ocds-synthetic", "url": URL, "ranking_eligible": False}
    row = dict(selected, status="downloaded_pdf", sha256=sha(blob), bytes=blob.stat().st_size, file_path=str(blob))
    plan = {"selected": [selected], "known_development_pdf_sha256": lock["known_pdf_sha256"],
            "source_design_manifest_sha256": sha(lock_dir / "manifest.json"),
            "input_sha256": {}, "limits": {"max_bytes_per_document": 10485760, "max_pages_per_document": 120, "timeout_seconds": 30}}
    plan_path = tmp_path / "access_plan.json"
    dump(plan_path, plan)
    dump(directory / "access_results.json", [row])
    dump(directory / "manifest.json", {"access_plan_path": str(plan_path), "access_plan_sha256": sha(plan_path),
         "output_sha256": {"access_results.json": sha(directory / "access_results.json"), blob.name: sha(blob)}})
    return lock_dir, directory, row


def native_fixture(tmp_path, design_lock, monkeypatch, pages=None):
    import scripts.prepare_document_sources_v1 as module
    lock_dir, access, row = access_fixture(tmp_path, design_lock)
    document = prepare_native(SimpleNamespace(is_encrypted=False, pages=pages or [Page("word " * 60), Page("native")]), row)
    original = module.subprocess.run
    def mocked(command, **kwargs):
        if command[-1] == "worker":
            return SimpleNamespace(stdout=json.dumps(document))
        return original(command, **kwargs)
    monkeypatch.setattr(module.subprocess, "run", mocked)
    output = tmp_path / "native"
    module.prepare_batch(lock_dir, access, output)
    return output


def test_native_plan_and_empty_ocr_no_model_invocation(tmp_path, design_lock, monkeypatch):
    native_dir = native_fixture(tmp_path, design_lock, monkeypatch)
    plan = json.loads((native_dir / "native_plan.json").read_text())
    assert [p["page"] for p in plan["selected_pages"]] == [2]
    assert plan["selection_before_ocr_and_inference"] and not plan["reference_values_used"]


def test_ocr_execution_and_source_composition_keep_lineage(tmp_path, design_lock, monkeypatch):
    import scripts.prepare_document_sources_v1 as module
    from scripts.register_browser_experience_sources import dump
    from tesis_recommender.document_experience_candidate_v1 import validate_document
    native_dir = native_fixture(tmp_path, design_lock, monkeypatch)
    original = module.subprocess.run
    def mocked(command, **kwargs):
        if Path(command[0]).name == "vision-pdf-ocr":
            raw = raw_ocr()
            raw["page"] = int(command[2])
            dump(Path(command[3]), raw)
            return SimpleNamespace()
        return original(command, **kwargs)
    monkeypatch.setattr(module.subprocess, "run", mocked)
    ocr_dir, output = tmp_path / "ocr", tmp_path / "view"
    module.run_ocr(native_dir, ocr_dir)
    summary = module.compose(native_dir, ocr_dir, output)
    docs = json.loads((output / "source_documents.json").read_text())
    assert summary["physical_pages"] == 2 and not summary["predictions_generated"]
    assert validate_document(docs[0])
    assert docs[0]["pages"][1]["ocr_used"] and docs[0]["pages"][1]["native_text"] == "native"


def test_real_installed_native_worker_on_previously_observed_pdf():
    from scripts.freeze_document_source_design_v1 import SOURCE_PYTHON
    from tesis_recommender.document_experience_candidate_v1 import validate_document
    docs = json.loads((ROOT / "artifacts/experience_holdout_source_view_v1/source_documents.json").read_text())
    original = min(docs, key=lambda d: len(d["pages"]))
    row = dict(original["source"])
    row["file_path"] = str((ROOT / row["file_path"]).resolve())
    result = subprocess.run([str(SOURCE_PYTHON), str(ROOT / "scripts/prepare_document_sources_v1.py"), "worker"],
                            input=json.dumps(row), text=True, capture_output=True, check=True, timeout=90)
    native_doc = json.loads(result.stdout)
    assert native_doc["status"] == "complete_native_view" and native_doc["total_pages"] == len(original["pages"])
    planned = [{"opportunity_id": row["opportunity_id"], "source_pdf_sha256": row["sha256"], "page": p["page"],
                "status": "not_attempted_after_error_stop"} for p in native_doc["pages"] if p["selected_for_ocr"]]
    prepared, = compose_documents([native_doc], planned)
    assert validate_document(prepared)
    assert not prepared["predictions_generated"]


def test_ocr_stops_after_two_errors_and_records_unattempted_pages(tmp_path, design_lock, monkeypatch):
    import scripts.prepare_document_sources_v1 as module
    native_dir = native_fixture(tmp_path, design_lock, monkeypatch, pages=[Page("native")] * 3)
    original = module.subprocess.run
    calls = []
    def mocked(command, **kwargs):
        if Path(command[0]).name == "vision-pdf-ocr":
            calls.append(command)
            raise subprocess.CalledProcessError(1, command)
        return original(command, **kwargs)
    monkeypatch.setattr(module.subprocess, "run", mocked)
    output = tmp_path / "ocr"
    summary = module.run_ocr(native_dir, output)
    rows = json.loads((output / "ocr_results.json").read_text())
    assert len(calls) == summary["pages_attempted"] == 2
    assert [r["status"] for r in rows] == ["ocr_error", "ocr_error", "not_attempted_after_error_stop"]
    view = tmp_path / "view"
    assert module.compose(native_dir, output, view)["unresolved_ocr_pages"] == 3


def test_cross_batch_duplicates_and_inaccessible_keep_full_denominator():
    doc = compose_documents([native(Page("word " * 50))], [])[0]
    other = copy.deepcopy(doc)
    other["opportunity_id"] = other["source"]["opportunity_id"] = "ocds-other"
    selected = [{"opportunity_id": name, "status": "eligible_initial_pdf_selected", "sample_number": n}
                for n, name in enumerate(("ocds-synthetic", "ocds-other", "ocds-failed", "ocds-unattempted"), 1)]
    batches = [{"source_documents": [doc], "coverage_ledger": [{"opportunity_id": "ocds-synthetic", "source_status": "complete_native_view"}]},
               {"source_documents": [other], "coverage_ledger": [{"opportunity_id": "ocds-other", "source_status": "complete_native_view"},
                   {"opportunity_id": "ocds-failed", "source_status": "access_not_complete_pdf"}]}]
    before = copy.deepcopy(batches)
    combined = combine_batches(selected, batches)
    assert len(combined["coverage_ledger"]) == 4 and not combined["source_documents"]
    assert len(combined["duplicate_documents_preserved"]) == 2
    assert [r["source_status"] for r in combined["coverage_ledger"]] == [
        "duplicate_pdf_across_batches_no_replacement", "duplicate_pdf_across_batches_no_replacement",
        "access_not_complete_pdf", "source_access_not_attempted"]
    assert batches == before and not combined["model_inference_performed"]


def test_combination_rejects_unselected_or_repeated_source_ledgers():
    selected = [{"opportunity_id": "ocds-one", "status": "no_eligible_initial_pdf"}]
    row = {"opportunity_id": "ocds-other", "source_status": "failed"}
    with pytest.raises(ValueError):
        combine_batches(selected, [{"source_documents": [], "coverage_ledger": [row]}])
    row["opportunity_id"] = "ocds-one"
    with pytest.raises(ValueError):
        combine_batches(selected, [{"source_documents": [], "coverage_ledger": [row, row]}])


def test_access_of_known_ocid_rejected_even_with_different_pdf_bytes(tmp_path, design_lock):
    import scripts.prepare_document_sources_v1 as module
    from scripts.register_browser_experience_sources import dump, sha
    lock_dir, access, row = access_fixture(tmp_path, design_lock)
    exclusions = json.loads((lock_dir / "exclusion_registry.json").read_text())
    known_id = exclusions["excluded_ocids"][0]
    plan_path = tmp_path / "access_plan.json"
    plan = json.loads(plan_path.read_text())
    plan["selected"][0]["opportunity_id"] = known_id
    row["opportunity_id"] = known_id
    dump(plan_path, plan)
    dump(access / "access_results.json", [row])
    manifest = json.loads((access / "manifest.json").read_text())
    manifest["access_plan_sha256"] = sha(plan_path)
    manifest["output_sha256"]["access_results.json"] = sha(access / "access_results.json")
    dump(access / "manifest.json", manifest)
    with pytest.raises(ValueError, match="Previously observed"):
        module.prepare_batch(lock_dir, access, tmp_path / "native")
