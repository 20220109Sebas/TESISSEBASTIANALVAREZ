import hashlib

import pytest

from scripts.compose_holdout_source_view import merge_source_page
from scripts.prepare_holdout_sparse_ocr import sparse_scan


def native():
    text = "Texto nativo intacto"
    return {"page": 2, "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}


def overlay():
    original = native()
    text = original["text"] + "\nTexto de una imagen"
    return {"page": 2, "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "native_text": original["text"], "native_text_sha256": original["text_sha256"],
            "ocr_selected": True, "ocr_used": True, "selected_ocr_unresolved": False}


def test_unselected_page_preserves_native_and_does_not_prove_absence():
    result = merge_source_page(native())
    assert result["text"] == result["native_text"] == native()["text"]
    assert not result["ocr_selected"] and not result["absence_of_requirements_proven"]


def test_selected_page_preserves_ocr_native_lineage():
    result = merge_source_page(native(), overlay())
    assert result["native_text"] == native()["text"]
    assert "Texto de una imagen" in result["text"]


@pytest.mark.parametrize("field,value", [("page", 3), ("native_text", "otro"),
    ("native_text_sha256", "wrong"), ("text_sha256", "wrong"), ("ocr_selected", False)])
def test_corrupt_overlay_rejected(field, value):
    row = overlay()
    row[field] = value
    with pytest.raises(ValueError):
        merge_source_page(native(), row)


def test_corrupt_native_rejected_even_without_overlay():
    row = native()
    row["text"] += " cambio"
    with pytest.raises(ValueError):
        merge_source_page(row)


@pytest.mark.parametrize("words,sizes,large,expected", [
    (17, [[595, 841]], False, True), (40, [[300, 300]], False, True),
    (41, [[300, 300]], False, False), (0, [], False, False),
    (10, [[299, 900]], False, False), (10, [[900, 299]], False, False),
    (10, [[600, 600]], True, False), (20, [[100, 100], [400, 700]], False, True)])
def test_sparse_scan_rule_is_fixed_by_source_quality(words, sizes, large, expected):
    assert sparse_scan({"word_count": words, "image_pixel_sizes": sizes, "large_raster_present": large}) is expected
