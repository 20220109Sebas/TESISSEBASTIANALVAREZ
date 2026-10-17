"""Select requirement documents before access, without consulting outcomes."""
import re
import unicodedata
from urllib.parse import parse_qs, urlparse
from uuid import UUID


def normalize_title(title):
    return " ".join("".join(c for c in unicodedata.normalize("NFKD", str(title or "")) if not unicodedata.combining(c)).casefold().split())


def document_role(title, kind):
    text = normalize_title(title)
    if kind != "biddingDocuments":
        return "excluded_type"
    if any(term in text for term in ("propuesta", "oferta", "buena pro", "adjudic", "evaluacion", "desierto", "resumen ejecutivo")):
        return "excluded_non_requirement_title"
    for phrase, role in (("bases administrativas", "administrative_bases"),
                         ("terminos de referencia", "terms_of_reference"),
                         ("requisitos tecnicos", "technical_requirements"),
                         ("especificaciones tecnicas", "technical_requirements"),
                         ("bases integradas", "integrated_bases")):
        if phrase in text:
            return role
    return "unknown_title_not_selected"


def safe_annex_url(url):
    try:
        parsed = urlparse(url)
        if (parsed.scheme != "https" or parsed.hostname != "prod1.seace.gob.pe"
                or parsed.path != "/SeaceWeb-PRO/SdescargarArchivoAlfresco"
                or parsed.port not in (None, 443) or parsed.username or parsed.password or parsed.fragment):
            return False
        query = parse_qs(parsed.query)
        if set(query) != {"fileCode"} or len(query["fileCode"]) != 1:
            return False
        UUID(query["fileCode"][0])
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def select_annexes(inventory):
    priority = {"administrative_bases": 0, "terms_of_reference": 1,
                "technical_requirements": 2, "integrated_bases": 3}
    eligible, excluded = [], []
    for doc in inventory:
        role = document_role(doc.get("title"), doc.get("document_type"))
        reason = role if role not in priority else "after_or_unknown_initial_date" if doc.get("temporal_metadata_status") != "declared_at_or_before_initial" else "unsafe_url" if not safe_annex_url(doc.get("url")) else None
        if reason:
            excluded.append(dict(doc, selection_reason=reason))
        else:
            eligible.append(dict(doc, document_role=role))
    ordered = sorted(eligible, key=lambda d: (d["sample_number"], priority[d["document_role"]], d["document_index"], d["url"]))
    chosen, seen_opportunities, seen_urls = [], set(), set()
    for doc in ordered:
        if doc["opportunity_id"] not in seen_opportunities and doc["url"] not in seen_urls:
            chosen.append(doc); seen_opportunities.add(doc["opportunity_id"]); seen_urls.add(doc["url"])
    return chosen, excluded, eligible
