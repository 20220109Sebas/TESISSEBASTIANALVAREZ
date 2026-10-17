"""Fixed metadata-only selection functions; not executed on a new real sample."""
import hashlib

from .annex_sources import document_role, normalize_title, safe_annex_url

SEED = "20261014-document-experience-validation-v2"
SAMPLE_SIZE = 50
ROLES = ("administrative_bases", "terms_of_reference", "technical_requirements", "integrated_bases")


def select_ids(rows, excluded):
    eligible, seen = [], set()
    for row in rows:
        ocid = row["opportunity_id"]
        if not isinstance(ocid, str) or not ocid.startswith("ocds-"):
            raise ValueError("Invalid candidate identity")
        if ocid in seen:
            raise ValueError("Duplicate candidate identity")
        seen.add(ocid)
        domain = normalize_title(row.get("title", "") + " " + row.get("description", ""))
        if (ocid not in excluded and row["object_type"] == "services"
                and row["publication_date"].startswith("2025-")
                and "consultoria de obra" not in domain and "consultoria en obra" not in domain):
            eligible.append(ocid)
    ordered = sorted(eligible, key=lambda ocid: (hashlib.sha256((SEED + ":" + ocid).encode()).hexdigest(), ocid))
    if len(ordered) < SAMPLE_SIZE:
        raise ValueError("Insufficient fixed eligible population; do not shrink or replace silently")
    return ordered[:SAMPLE_SIZE], len(eligible)


def choose_document(documents, initial_timestamp, used_urls):
    from datetime import datetime
    options = []
    initial = datetime.fromisoformat(initial_timestamp.replace("Z", "+00:00"))
    if initial.tzinfo is None:
        raise ValueError("Initial version requires timezone")
    for index, doc in enumerate(documents):
        role = document_role(doc.get("title"), doc.get("documentType"))
        if role not in ROLES or str(doc.get("format", "")).casefold() != "pdf" or not safe_annex_url(doc.get("url")):
            continue
        try:
            date = datetime.fromisoformat(doc["datePublished"].replace("Z", "+00:00"))
        except (ValueError, KeyError, AttributeError, TypeError):
            continue
        if date.tzinfo is not None and date <= initial:
            options.append({"document_index": index, "document_role": role, "document": dict(doc)})
    if not options:
        return None, "no_eligible_initial_pdf"
    chosen = min(options, key=lambda d: (ROLES.index(d["document_role"]), d["document_index"], d["document"]["url"]))
    if chosen["document"]["url"] in used_urls:
        return None, "duplicate_initial_document_url_no_replacement"
    return chosen, "eligible_initial_pdf_selected"
