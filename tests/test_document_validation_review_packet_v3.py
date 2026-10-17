"""Test actual partial progress registration, not fabricated complete reference."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_partial_notes_never_claim_all_documents_reviewed():
    notes = json.loads((ROOT / "docs/document_validation_source_review_notes_v3.json").read_text())
    assert notes["reference_locked"] is False and notes["predictions_consulted"] is False
    assert [d["sample_number"] for d in notes["documents"]] == [4, 25]
    assert [r["pass_number"] for r in notes["documents"][0]["readings"]] == [1, 2]
    assert [r["pass_number"] for r in notes["documents"][1]["readings"]] == [1]


def test_draft_with_unread_source_cannot_be_locked():
    from test_document_experience_evaluation_v1 import packet
    from tesis_recommender.document_experience_evaluation_v1 import build_reference
    documents, ledger, draft = packet()
    draft["documents"][0].update(scope="pending_reference", readings=[], conditions=[])
    with pytest.raises(ValueError):
        build_reference(draft, documents, ledger)


def test_real_notes_bound_to_exact_prepared_page_views():
    from tesis_recommender.document_experience_candidate_v1 import validate_document
    notes = json.loads((ROOT / "docs/document_validation_source_review_notes_v3.json").read_text())
    for entry, batch in zip(notes["documents"], ("batch_01", "batch_03")):
        documents = json.loads((ROOT / "artifacts/document_validation_source_views_v3" / batch / "source_documents.json").read_text())
        doc = next(d for d in documents if d["opportunity_id"] == entry["opportunity_id"])
        validate_document(doc)
        assert doc["source"]["sha256"] == entry["source_pdf_sha256"]
        assert [p["text_sha256"] for p in doc["pages"]] == entry["page_text_sha256"]
        for reading in entry["readings"]:
            assert sorted(reading["pages_read_order"]) == list(range(1, len(doc["pages"]) + 1))
            assert len(reading["page_notes"]) == len(doc["pages"])
