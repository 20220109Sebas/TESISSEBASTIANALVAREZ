"""Choose same-process substitute requirement documents without reading their text."""
from datetime import datetime

from .annex_sources import document_role, safe_annex_url

PRIORITY = {'administrative_bases': 0, 'terms_of_reference': 1,
            'technical_requirements': 2, 'integrated_bases': 3}


def timestamp(value):
    try:
        t = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return t if t.tzinfo is not None else None
    except (ValueError, TypeError, AttributeError):
        return None


def select_substitute(package, original):
    ocid = original['opportunity_id']
    cutoff = timestamp(original['initial_version_published_at'])
    if cutoff is None:
        raise ValueError('Original cutoff is missing or ambiguous')
    matching = [r for r in package.get('records', []) if r.get('ocid') == ocid]
    if len(matching) != 1 or matching[0].get('compiledRelease', {}).get('ocid') != ocid:
        raise ValueError('Current record does not uniquely match original opportunity')
    compiled = matching[0]['compiledRelease']
    accepted, excluded, seen = [], [], set()
    for i, doc in enumerate((compiled.get('tender') or {}).get('documents') or []):
        role = document_role(doc.get('title'), doc.get('documentType'))
        published = timestamp(doc.get('datePublished'))
        modified = timestamp(doc.get('dateModified')) if doc.get('dateModified') else None
        reason = ('original_inaccessible_url_not_retried' if doc.get('url') == original['url'] else
                  'unsafe_or_unsupported_url' if not safe_annex_url(doc.get('url')) else
                  'excluded_document_role' if role not in PRIORITY else
                  'not_declared_pdf' if str(doc.get('format', '')).casefold() != 'pdf' else
                  'missing_document_identity' if not doc.get('id') else
                  'unknown_publication_date' if published is None else
                  'invalid_modification_date' if doc.get('dateModified') and modified is None else None)
        identity = (str(doc.get('id')), doc.get('url'))
        if reason is None and identity in seen:
            reason = 'duplicate_document_metadata'
        seen.add(identity)
        if reason:
            excluded.append({'document_id': doc.get('id'), 'title': doc.get('title'),
                             'url': doc.get('url'), 'reason': reason})
            continue
        after = published > cutoff or (modified is not None and modified > cutoff)
        row = {'opportunity_id': ocid, 'sample_number': original['sample_number'],
               'document_index': i, 'document_id': doc['id'], 'title': doc.get('title'),
               'document_type': doc.get('documentType'), 'document_role': role,
               'format': doc['format'], 'url': doc['url'],
               'date_published_declared': doc['datePublished'],
               'date_modified_declared': doc.get('dateModified'),
               'initial_version_published_at': original['initial_version_published_at'],
               'temporal_metadata_status': 'declared_after_or_modified_after_initial' if after else 'declared_at_or_before_initial',
               'metadata_from_current_compiled_record': True,
               'source_substitution': True, 'same_process_verified_by_ocid': True,
               'same_document_identity_declared': str(doc['id']) == str(original['document_id']),
               'replaces_original_document': original,
               'eligible_for_initial_time_ranking': False, 'historical_content_proven': False,
               'ranking_eligible': False,
               'substitute_use': 'separate_document_extraction_development_not_historical_ranking'}
        accepted.append(row)
    # Prioritize temporal metadata before document role, then earliest declared date.
    ordered = sorted(accepted, key=lambda r: (r['temporal_metadata_status'] != 'declared_at_or_before_initial',
        PRIORITY[r['document_role']], timestamp(r['date_published_declared']), str(r['document_id']), r['url']))
    return {'selected': ordered[0] if ordered else None, 'eligible_candidates': ordered,
            'excluded': excluded, 'selection_reads_document_text_or_outcomes': False,
            'all_selected_bytes_historical_status': 'not_proven'}
