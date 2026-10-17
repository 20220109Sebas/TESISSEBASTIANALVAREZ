import hashlib

import pytest

from scripts.verify_holdout_source_review import check_quote, quotations


def page():
    return {"native_text": "texto nativo", "text": "texto OCR", "ocr_used": True}


def evidence(basis="native_text_verbatim"):
    text = page()["native_text"] if basis == "native_text_verbatim" else page()["text"]
    return {"quote_basis": basis, "char_start": 0, "char_end": len(text), "quote": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest()}


@pytest.mark.parametrize("basis", ["native_text_verbatim", "ocr_text_verbatim_not_error_free"])
def test_native_and_ocr_quotes_keep_distinct_sources(basis):
    check_quote(page(), evidence(basis))


@pytest.mark.parametrize("field,value", [("char_start", -1), ("char_end", 100),
    ("char_end", 0), ("quote", "otra fuente"), ("text_sha256", "wrong"), ("quote_basis", "invented")])
def test_corrupt_quote_rejected(field, value):
    row = evidence()
    row[field] = value
    with pytest.raises(ValueError):
        check_quote(page(), row)


def test_ocr_quote_cannot_be_checked_against_native_only_page():
    row = page()
    row["ocr_used"] = False
    with pytest.raises(ValueError):
        check_quote(row, evidence("ocr_text_verbatim_not_error_free"))


def test_nested_evidence_including_multipliers_is_verified():
    value = {"documents": [{"evidence": [evidence()], "multiplier_evidence": [evidence()]}]}
    assert len(list(quotations(value))) == 2
