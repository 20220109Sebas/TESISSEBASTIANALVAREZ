import hashlib
import pytest
from scripts.complete_mixed_pdf_text import complete_page, needs_ocr, raster_sizes


def native():
    text = "Encabezado nativo"
    return {"page": 22, "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}


def ocr():
    text = "Texto reconocido de la página completa"
    return {"page": 22, "status": "ok", "text": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "ocr_json_path": "/tmp/evidence.json", "ocr_json_sha256": "test",
            "low_confidence_lines": 2}


def test_native_page_is_preserved_without_ocr():
    result = complete_page(native(), None, False)
    assert result["text"] == result["native_text"]
    assert not result["ocr_used"]
    assert not result["selected_ocr_unresolved"]


def test_mixed_header_does_not_hide_scanned_body():
    result = complete_page(native(), ocr(), True)
    assert result["text"] == ocr()["text"]
    assert result["native_text"] == native()["text"]
    assert result["ocr_used"]
    assert result["confidence_is_not_semantic_accuracy"]
    assert not result["absence_of_requirements_proven"]


@pytest.mark.parametrize("row", [None, {"status": "ocr_error"}, {"status": "ok", "text": " "}])
def test_missing_failed_or_empty_ocr_is_unresolved(row):
    result = complete_page(native(), row, True)
    assert result["selected_ocr_unresolved"]
    assert result["text"] == native()["text"]


def test_corrupted_native_hash_rejected():
    row = native()
    row["text"] += "changed"
    with pytest.raises(ValueError):
        complete_page(row, None, False)


@pytest.mark.parametrize("key,value", [("page", 23), ("text", "changed")])
def test_ocr_page_or_hash_mismatch_rejected(key, value):
    row = ocr()
    row[key] = value
    with pytest.raises(ValueError):
        complete_page(native(), row, True)


def test_raster_selection_does_not_use_text_or_relevance():
    assert needs_ocr([[600, 600]])
    assert not needs_ocr([[599, 900], [1500, 200]])
    assert not needs_ocr([])
    assert raster_sizes(None) == []
