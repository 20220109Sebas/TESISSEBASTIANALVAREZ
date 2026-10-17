import hashlib

import pytest

from scripts.audit_holdout_evaluation_readiness import validate_source_document
from scripts.extend_holdout_reference_draft import literal_span
from scripts.prepare_holdout_source_reading import transcript


def doc():
    value = "Texto íntegro de la fuente"
    page = {"page": 1, "text": value, "text_sha256": hashlib.sha256(value.encode()).hexdigest(),
        "native_text": value, "native_text_sha256": hashlib.sha256(value.encode()).hexdigest(),
        "ocr_selected": False, "ocr_used": False, "selected_ocr_unresolved": False}
    return {"ranking_eligible": False, "historical_content_proven": False, "pages": [page]}


def test_valid_source_lineage():
    validate_source_document(doc())


@pytest.mark.parametrize("field,value", [("page", 2), ("text", "otra fuente"),
    ("native_text", "otro texto"), ("ocr_used", True), ("selected_ocr_unresolved", True)])
def test_invalid_page_lineage_stops(field, value):
    row = doc()
    row["pages"][0][field] = value
    with pytest.raises(ValueError):
        validate_source_document(row)


@pytest.mark.parametrize("field", ["ranking_eligible", "historical_content_proven"])
def test_holdout_sources_do_not_enter_historical_ranking(field):
    row = doc()
    row[field] = True
    with pytest.raises(ValueError):
        validate_source_document(row)


def test_repeated_page_rejected():
    row = doc()
    row["pages"].append(dict(row["pages"][0]))
    with pytest.raises(ValueError):
        validate_source_document(row)


def test_literal_source_span():
    context, evidence = literal_span("antes INICIO S/ 150.00 FIN después", "INICIO", "FIN", "S/ 150.00", 3)
    assert evidence["literal"] == "S/ 150.00" and evidence["page"] == 3
    assert context["char_start"] == 6


@pytest.mark.parametrize("text", ["INICIO S/ 150.00 INICIO FIN", "INICIO S/ 150.00 S/ 150.00 FIN", "INICIO otro FIN"])
def test_ambiguous_or_missing_literal_rejected(text):
    with pytest.raises(ValueError):
        literal_span(text, "INICIO", "FIN", "S/ 150.00", 1)


def test_source_transcript_preserves_physical_page_and_literal_text():
    row = doc()
    row.update(opportunity_id="synthetic", source={"sha256": "synthetic-pdf-hash"})
    row["pages"][0]["extraction_method"] = "native_pdf_text"
    result = transcript(row)
    assert "PHYSICAL PAGE 1" in result and row["pages"][0]["text"] in result
    assert "not a completed document review" in result
