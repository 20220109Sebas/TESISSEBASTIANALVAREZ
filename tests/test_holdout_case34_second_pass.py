import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.record_holdout_condition_second_pass_v5 import build, compare_conditions, validate

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/experience_holdout_review_progress_v43"
SOURCE = ROOT / "artifacts/experience_holdout_source_view_v1"
REVIEW = ROOT / "docs/holdout_case34_second_source_pass_v1.json"


def inputs():
    review = json.loads(REVIEW.read_text())
    sources = {d["opportunity_id"]: d for d in json.loads((SOURCE / "source_documents.json").read_text())}
    first = {d["opportunity_id"]: d for d in json.loads((BASE / "compiled_source_reviews.json").read_text())["documents"]}
    return review, sources, first


def test_case34_actual_text_and_visual_coverage():
    review, sources, first = inputs()
    validate(review, sources, first)
    d = review["documents"][0]
    assert d["pages_read"] == list(range(1, 106))
    assert d["ocr_pages_read"] == list(range(25, 51))
    assert len(d["native_pages_read"]) == 79
    assert len(d["rendered_pages"]) == len(d["complete_rendered_pages_read"]) == 41
    assert len(d["native_pages_not_rendered"]) == 64
    assert not d["all_content_proven_recovered"] and not d["all_pages_visually_reviewed"]
    assert not d["new_ocr_executed"] and not d["entire_procurement_file_reviewed"]
    assert d["unreviewed_pages"] == [] and len(d["evidence"]) == 22
    assert all(o["replaces_archived_text"] is False for o in d["visual_observations"])
    assert d["domain"] == "consultoria_de_obra_supervision_salud"
    assert d["formal_services_evaluation_eligible"] is False
    assert all(not d[k] for k in ("historical_content_proven", "ranking_eligible", "predictions_generated"))


def test_case34_relative_amount_is_not_budget_or_actual_company_history():
    review, _, first = inputs()
    d = review["documents"][0]
    c = d["bidder_experience_conditions"][0]
    assert len(d["bidder_experience_conditions"]) == 1
    assert c["subject"] == "bidder" and c["section_role"] == "qualification"
    assert c["amount_expression"] == "UNA VEZ LA CUANTÍA DE LA CONTRATACIÓN"
    assert c["quantia_multiplier"] == 1 and c["quantia_context"] == 229693.56
    assert c["billed_amount"] is None and not c["absolute_amount_computed"]
    assert c["currency"] == "PEN" and c["quantia_context_page"] == 17
    assert c["currency_basis"] == "quantia_context_page_17_not_literal_absolute_experience_amount"
    assert c["lookback_years"] == 20 and c["reference_date"] == "presentacion_de_ofertas"
    assert c["window_anchor"] == "conformidad_o_emision_del_comprobante_de_pago_final"
    assert c["minimum_contracts"] is None and c["maximum_contracts_for_accreditation"] == 20
    assert c["company_applicability"] == "not_determined" and not c["document_reference_complete"]
    prior = first[d["opportunity_id"]]["bidder_experience_conditions"][0]
    for name, comparison in d["supplemental_first_pass_comparison"]["fields"].items():
        assert comparison == {"first": prior[name], "second": c[name], "status": "match"}


def test_case34_visual_transcription_is_separate_from_archived_ocr():
    review, sources, _ = inputs()
    d = review["documents"][0]
    t = d["visual_transcriptions"][0]
    c = d["bidder_experience_conditions"][0]
    page = sources[d["opportunity_id"]]["pages"][45]
    assert t["page"] == 46 and page["ocr_used"]
    assert t["text"] not in page["text"]
    assert "veinte (20) años" in t["text"] and "veinte (20) años" not in page["text"]
    assert "conformidad o emisión" in t["text"]
    assert "comprobante de pago final" in t["text"]
    assert t["replaces_archived_text"] is False and t["literal_in_archived_text"] is False
    assert t["human_annotation"] is False
    assert c["visual_transcription_ids"] == [t["id"]]
    rendered = next(r for r in d["rendered_pages"] if r["page"] == t["page"])
    assert rendered["path"] == t["image_path"] and (ROOT / t["image_path"]).is_file()
    assert t["supports_fields"] == ["amount_expression", "quantia_multiplier", "lookback_years", "reference_date", "window_anchor"]


def test_case34_personnel_annexes_and_conditional_admission_stay_separate():
    d = inputs()[0]["documents"][0]
    p = d["personnel_experience_observations"]
    assert list(p[0]["minimum_months_by_role"].values()) == [24, 24, 12]
    assert p[0]["start_anchor"] == "obtencion_de_colegiatura" and p[0]["lookback_years"] == 25
    assert p[0]["overlapping_periods_count_once"]
    assert p[1]["additional_years"] == 1 and p[1]["lookback_years"] == 25
    assert p[1]["intermediate_band_struck_out"] and p[1]["intermediate_points"] is None
    assert p[2]["blank_template"] and not p[2]["actual_personnel_history_proven"]
    assert all(not r["applied_to_bidder"] for r in p)
    others = {o["branch"]: o for o in d["other_bidder_experience_observations"]}
    admission = others["conditional_impediment_removal"]
    assert admission["lookback_years"] == 2 and admission["activation_condition"]
    assert [a["minimum_contracts"] for a in admission["alternatives"]] == [1, 4]
    assert admission["consecutive_years_for_services"] and not admission["applied_to_general_specialty"]
    annex = others["annex_specialty_template"]
    assert annex["blank_template"] and not annex["actual_company_history_proven"]
    assert [w["years"] for w in annex["template_windows"]] == [15, 20]
    assert not annex["applied_as_new_qualification"]
    reorg = others["reorganization_accreditation"]
    assert reorg["source_annex_reference"] == 13 and reorg["actual_annex_number"] == 12
    assert d["document_level_lookback_years"] is None


def test_case34_comparison_preserves_first_pass_and_unknowns():
    review, _, first = inputs()
    d = review["documents"][0]
    prior = first[d["opportunity_id"]]["bidder_experience_conditions"]
    before = copy.deepcopy(prior)
    comparison = compare_conditions(prior, d["bidder_experience_conditions"])
    assert prior == before and len(comparison) == 1
    fields = comparison[0]["fields"]
    assert fields["billed_amount"]["status"] == fields["similar_services"]["status"] == "unknown"
    assert fields["lookback_years"]["status"] == fields["window_anchor"]["status"] == "match"
    assert d["decision"] == "source_conditions_discrepant_pending_reference_lock"
    changed = copy.deepcopy(d["bidder_experience_conditions"])
    changed[0]["lookback_years"] = 25
    assert compare_conditions(prior, changed)[0]["fields"]["lookback_years"]["status"] == "disagreement"


@pytest.mark.parametrize("mutation", [
    "agreement", "missing_page", "fake_ocr", "missing_image", "personnel",
    "invented_quote", "eligibility", "lock", "pdf", "currency",
    "duplicate_image", "evidence", "duplicate_condition", "numeric_amount",
    "zero_window", "boolean_window", "annex_role", "section_missing_page",
    "hidden_unrendered", "false_visual_claim", "historical_use",
])
def test_case34_registrar_rejects_unsafe_structure(mutation):
    review, sources, first = inputs()
    d = review["documents"][0]
    c = d["bidder_experience_conditions"][0]
    if mutation == "agreement":
        d["decision"] = "source_conditions_agree_with_warnings_pending_reference_lock"
    elif mutation == "missing_page":
        d["pages_read"].remove(105)
    elif mutation == "fake_ocr":
        d["native_pages_read"].remove(105)
        d["ocr_pages_read"].append(105)
    elif mutation == "missing_image":
        d["complete_rendered_pages_read"].remove(46)
    elif mutation == "personnel":
        c["subject"] = "personnel"
    elif mutation == "invented_quote":
        d["evidence"].append({"id": "fake", "page": 46, "quote": "SIN EXPERIENCIA EXIGIDA"})
    elif mutation == "eligibility":
        c["company_applicability"] = "eligible"
    elif mutation == "lock":
        review["reference_locked"] = True
    elif mutation == "pdf":
        d["pdf_sha256"] = "0" * 64
    elif mutation == "currency":
        c["currency"] = "S/"
    elif mutation == "duplicate_image":
        d["rendered_pages"][1]["path"] = d["rendered_pages"][0]["path"]
    elif mutation == "evidence":
        c["evidence_ids"].append("missing")
    elif mutation == "duplicate_condition":
        d["bidder_experience_conditions"].append(copy.deepcopy(c))
    elif mutation == "numeric_amount":
        c["billed_amount"] = 229693.56
    elif mutation in ("zero_window", "boolean_window"):
        c["lookback_years"] = 0 if mutation == "zero_window" else True
    elif mutation == "annex_role":
        c["section_role"] = "annex"
    elif mutation == "section_missing_page":
        d["reading_sections"][-1]["pages"].remove(105)
    elif mutation == "hidden_unrendered":
        d["native_pages_not_rendered"].pop()
    elif mutation == "false_visual_claim":
        d["all_pages_visually_reviewed"] = True
    elif mutation == "historical_use":
        d["ranking_eligible"] = True
    with pytest.raises(ValueError):
        validate(review, sources, first)


def test_case34_build_appends_without_rewriting_history_or_unlocking_metrics(tmp_path):
    output = tmp_path / "progress"
    result = build(BASE, SOURCE, REVIEW, output)
    assert result["second_pass_full_supplied_pdf"] == 24
    assert result["condition_source_agreements_pending_reference"] == 4
    assert not result["f1_estimable"]
    for name in ("compiled_source_reviews.json", "partial_source_reviews.json", "second_source_review.json"):
        assert (BASE / name).read_bytes() == (output / name).read_bytes()
    previous = json.loads((BASE / "condition_second_source_reviews.json").read_text())
    current = json.loads((output / "condition_second_source_reviews.json").read_text())
    assert current[:-1] == previous and current[-1] == json.loads(REVIEW.read_text())
    rows = json.loads((output / "review_progress.json").read_text())
    assert len(rows) == 50
    assert [r["sample_number"] for r in rows if r["first_pass_status"] == "full_supplied_pdf"
            and r["second_pass_status"] == "not_performed"] == [1, 11, 40]
    case = next(r for r in rows if r["sample_number"] == 34)
    assert case["reference_adjudication_pending"] and case["second_pass_status"] == "full_supplied_pdf"
    assert all(not r["historical_ranking_eligible"] and not r["reference_locked"] for r in rows)
    summary = json.loads((output / "summary.json").read_text())
    assert all(summary[k] is False for k in ("reference_locked", "predictions_generated", "f1_estimable", "ranking_inputs_modified"))
    assert summary["positive_document_references_complete"] == 0
    evidence = [e for e in json.loads((output / "second_pass_evidence.json").read_text()) if e["opportunity_id"] == case["opportunity_id"]]
    assert len(evidence) == 22
    assert all(e["image_sha256"] and not e["human_or_expert"] and not e["independent_reviewer"] for e in evidence)
    manifest = json.loads((output / "manifest.json").read_text())
    for d in json.loads(REVIEW.read_text())["documents"]:
        for rendered in d["rendered_pages"]:
            path = (ROOT / rendered["path"]).resolve()
            assert manifest["input_sha256"][str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()
