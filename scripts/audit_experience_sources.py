"""Inventory the frozen 50-document sample and freeze a bounded annex access pilot."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def before_initial(value, initial):
    if not value:
        return "missing_document_date"
    try:
        published, cutoff = datetime.fromisoformat(value.replace("Z", "+00:00")), datetime.fromisoformat(initial.replace("Z", "+00:00"))
        if published.tzinfo is None or cutoff.tzinfo is None:
            return "date_without_timezone"
        return "declared_at_or_before_initial" if published <= cutoff else "declared_after_initial"
    except (ValueError, TypeError):
        return "invalid_document_date"


def eligible_url(url):
    p = urlparse(url)
    return p.scheme == "https" and p.hostname == "prod1.seace.gob.pe" and p.path == "/SeaceWeb-PRO/SdescargarArchivoAlfresco" and not p.username and not p.password and p.port in (None, 443)


def build(output, limit=5):
    if output.exists():
        raise ValueError("Use a new inventory directory")
    audit = ROOT / "artifacts/extraction_commitment_v1"
    m = json.loads((audit / "manifest.json").read_text())
    evidence = {}
    for name in ("review_packet.json", "summary.json"):
        p = audit / name
        if sha(p) != m["output_sha256"][name]:
            raise ValueError("Frozen audit changed")
        evidence[str(p)] = sha(p)
    packet = json.loads((audit / "review_packet.json").read_text())
    candidate_path = ROOT / "data/processed/contrast_candidates_v1/annual_v1/test_opportunities.csv"
    experiment = json.loads((ROOT / "artifacts/experiment_annual_v1/manifest.json").read_text())
    if sha(candidate_path) != experiment["input_sha256"][str(candidate_path.relative_to(ROOT))]:
        raise ValueError("Frozen candidates changed")
    evidence[str(candidate_path)] = sha(candidate_path)
    mentions = []
    with candidate_path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            for field in ("description", "technical_text"):
                if "experiencia" in row[field].casefold():
                    mentions.append({"opportunity_id": row["opportunity_id"], "field": field, "text": row[field]})
    documents = []
    for r in packet:
        p = ROOT / r["source_path"]
        if sha(p) != r["source_sha256"]:
            raise ValueError("Initial source changed")
        evidence[str(p)] = sha(p)
        source = json.loads(p.read_text())["releases"][0]
        if source.get("ocid") != r["opportunity_id"] or source.get("publishedDate") != r["source_published_at"]:
            raise ValueError("Initial source identity changed")
        for index, doc in enumerate((source.get("tender") or {}).get("documents") or []):
            temporal = before_initial(doc.get("datePublished"), r["source_published_at"])
            url = doc.get("url", "")
            eligible = doc.get("documentType") == "biddingDocuments" and temporal == "declared_at_or_before_initial" and eligible_url(url)
            documents.append({"opportunity_id": r["opportunity_id"], "sample_number": r["sample_number"],
                              "document_index": index, "document_id": doc.get("id"), "title": doc.get("title"),
                              "document_type": doc.get("documentType"), "format": doc.get("format"), "url": url,
                              "date_published_declared": doc.get("datePublished"), "initial_version_published_at": r["source_published_at"],
                              "initial_source_sha256": r["source_sha256"], "initial_source_uri": r["source_uri"],
                              "temporal_metadata_status": temporal, "access_pilot_eligible": eligible})
    unique = {}
    for doc in sorted(documents, key=lambda d: (d["sample_number"], d["document_index"])):
        if doc["access_pilot_eligible"]:
            unique.setdefault(doc["url"], doc)
    selected = list(unique.values())[:limit]
    summary = {"sample_opportunities": len(packet), "sample_document_entries": len(documents),
               "opportunities_with_document_links": len({d["opportunity_id"] for d in documents if d["url"]}),
               "types": dict(Counter(d["document_type"] for d in documents)),
               "temporal_metadata": dict(Counter(d["temporal_metadata_status"] for d in documents)),
               "eligible_unique_bidding_document_links": len(unique), "pilot_links": len(selected),
               "annual_opportunities_with_experience_word": len({r["opportunity_id"] for r in mentions}),
               "annual_fields_with_experience_word": len(mentions),
               "labels_read": False, "rankings_changed": False,
               "historical_pdf_content_proven": False}
    plan = {"purpose": "bounded_access_and_text_feasibility_not_validation_or_historical_pdf_proof",
            "selection": "first_5_unique_eligible_bidding_document_urls_in_frozen_50_sample_hash_order",
            "metadata_basis": "doc date at or before initial release, not proof of historical bytes",
            "limits": {"documents": limit, "max_bytes_per_document": 10 * 1024 * 1024, "timeout_seconds": 30,
                       "max_pages_per_document": 120, "workers": 1, "ocr": False},
            "allowed_host": "prod1.seace.gob.pe", "selected": selected,
            "no_outcome_documents": True, "no_manual_annotations": True,
            "script_sha256": sha(Path(__file__)), "input_sha256": evidence}
    output.mkdir(parents=True)
    for name, content in (("summary.json", summary), ("document_inventory.json", documents),
                          ("text_mentions.json", mentions), ("access_plan.json", plan)):
        (output / name).write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"input_sha256": evidence, "script_sha256": sha(Path(__file__)),
                "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir())}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/experience_source_inventory_v1")
    args = parser.parse_args()
    print(json.dumps(build(args.output), ensure_ascii=False))
