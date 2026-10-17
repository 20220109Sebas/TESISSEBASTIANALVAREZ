#!/usr/bin/env python3
"""Independent, read-only source comparison; never produces human annotations."""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

FIELDS = ('title', 'description', 'technical_text', 'category', 'amount', 'currency', 'region')

def sha(data):
    return hashlib.sha256(data).hexdigest()

def normalized(value, field):
    if value is None or value == '':
        return None
    if field == 'amount':
        try:
            number = Decimal(str(value))
            return number if number.is_finite() else ('invalid', str(value))
        except InvalidOperation:
            return ('invalid', str(value))
    return ' '.join(str(value).split()).casefold() or None

def expected(release, field):
    """Implement the written guide directly, without importing project extractors."""
    tender = release.get('tender') or {}
    items = tender.get('items') or []
    if field in ('title', 'description'):
        return tender.get(field), {'tender.' + field: tender.get(field)}
    if field in ('amount', 'currency'):
        value = tender.get('value') or {}
        return value.get(field), {'tender.value': value}
    if field == 'technical_text':
        pieces = []
        for item in items:
            pieces.extend([item.get('description'), (item.get('classification') or {}).get('description')])
            pieces.extend(c.get('description') for c in item.get('additionalClassifications', []))
        return ' '.join(str(p) for p in pieces if p is not None and str(p).strip()), {'tender.items': items}
    if field == 'category':
        classification = (items[0].get('classification') or {}) if items else {}
        return classification.get('description') or classification.get('id') or tender.get('mainProcurementCategory'), {'first_item_classification': classification, 'tender.mainProcurementCategory': tender.get('mainProcurementCategory')}
    if field == 'region':
        references = [release.get('buyer') or {}, tender.get('procuringEntity') or {}]
        ids = {r.get('id') for r in references if r.get('id')}
        candidates = [p for p in release.get('parties', []) if p.get('id') in ids]
        if not candidates:
            candidates = [p for p in release.get('parties', []) if set(p.get('roles', [])) & {'buyer', 'procuringEntity'}]
        values = []
        for party in candidates:
            address = party.get('address') or {}
            value = next((address[k] for k in ('department', 'region', 'locality') if address.get(k)), None)
            if value is not None:
                values.append(value)
        if len({normalized(v, field) for v in values}) > 1:
            raise ValueError('Ambiguous buyer/procuringEntity addresses')
        return values[0] if values else None, {'buyer': references[0], 'procuringEntity': references[1], 'matching_parties': candidates}
    raise ValueError('Unknown field: ' + field)

def run(source, output, root, consolidated=None):
    before = source.read_bytes()
    rows = list(csv.DictReader(io.StringIO(before.decode('utf-8-sig'), newline='')))
    results, releases, limitations = [], {}, []
    for line, row in enumerate(rows, 2):
        result = {'csv_line': line, 'opportunity_id': row['opportunity_id'], 'field': row['field'], 'prediction': row['prediction'], 'release_path': row['release_path'], 'declared_sha256': row['release_sha256']}
        try:
            path = root / row['release_path']
            raw = path.read_bytes()
            package = json.loads(raw)
            matches = [r for r in package.get('releases', []) if r.get('ocid') == row['opportunity_id']]
            result.update(actual_sha256=sha(raw), sha256_matches=sha(raw) == row['release_sha256'], ocid_matches=len(matches) == 1)
            if len(matches) != 1:
                raise ValueError('Expected exactly one matching release')
            release = matches[0]
            releases[row['opportunity_id']] = release
            value, evidence = expected(release, row['field'])
            result.update(expected_value=value, expected_state='absent' if normalized(value, row['field']) is None else 'observed', evidence_fragment=evidence, release_id=release.get('id'), release_date=release.get('date'), release_uri=package.get('uri'), source_url=row['source_url'], source_url_is_record='/record/' in row['source_url'])
            result['value_matches'] = normalized(value, row['field']) == normalized(row['prediction'], row['field'])
            result['status'] = 'match' if result['value_matches'] and result['sha256_matches'] else 'mismatch'
        except (OSError, ValueError, KeyError, TypeError) as exc:
            result.update(status='error', error=str(exc))
        results.append(result)
    for ocid, release in releases.items():
        tender = release.get('tender') or {}
        if (tender.get('value') or {}).get('amount') == 0:
            limitations.append({'ocid': ocid, 'kind': 'zero_amount_observed', 'evidence': tender['value']})
        description = tender.get('description') or ''
        for index, item in enumerate(tender.get('items') or []):
            text = item.get('description') or ''
            if text and len(text) < len(description) and description.startswith(text):
                limitations.append({'ocid': ocid, 'kind': 'item_description_strict_prefix_possible_truncation', 'item_index': index, 'item_description': text, 'tender_description': description})
    pairs = Counter((r['opportunity_id'], r['field']) for r in rows)
    summary = {'audit_kind': 'automatic_AI_not_human_or_expert_validation', 'csv_sha256_before': sha(before), 'rows': len(rows), 'opportunities': len(releases), 'status_counts': dict(Counter(r['status'] for r in results)), 'by_field': {f: dict(Counter(r['status'] for r in results if r['field'] == f)) for f in FIELDS}, 'duplicate_pairs': [list(k) for k,v in pairs.items() if v != 1], 'missing_fields': {ocid: sorted(set(FIELDS) - {f for o,f in pairs if o == ocid}) for ocid in {r['opportunity_id'] for r in rows} if set(FIELDS) - {f for o,f in pairs if o == ocid}}, 'limitation_counts': dict(Counter(x['kind'] for x in limitations)), 'human_cells_populated': sum(bool(r.get(k)) for r in rows for k in ('gold_value','gold_state','annotator_id','reviewed_at','evidence_fragment')), 'network_requests': 0}
    if consolidated:
        raw = consolidated.read_bytes()
        package = json.loads(raw)
        records = package.get('records') or []
        versions = [r.get('compiledRelease', {}) for r in records] or package.get('releases', [])
        summary['provided_version_comparison'] = {'path': str(consolidated), 'sha256': sha(raw), 'versions': [{'ocid': r.get('ocid'), 'id': r.get('id'), 'date': r.get('date'), 'value': (r.get('tender') or {}).get('value')} for r in versions]}
    summary['csv_sha256_after'] = sha(source.read_bytes())
    summary['unknown_fields'] = sorted({r['field'] for r in rows} - set(FIELDS))
    summary['passed'] = bool(rows) and all(r['status'] == 'match' and r.get('sha256_matches') and r.get('ocid_matches') for r in results) and not any(summary[k] for k in ('duplicate_pairs', 'missing_fields', 'unknown_fields'))
    assert summary['csv_sha256_before'] == summary['csv_sha256_after']
    output.mkdir(parents=True, exist_ok=False)
    for name, data in [('row_results.json', results), ('summary.json', summary), ('source_limitations.json', limitations)]:
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (output / 'row_results.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        columns = sorted(set().union(*(r.keys() for r in results)))
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict,list)) else v for k,v in r.items()} for r in results)
    manifest = {'script_sha256': sha(Path(__file__).read_bytes()), 'source_csv_sha256': sha(before), 'outputs_sha256': {p.name: sha(p.read_bytes()) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source', type=Path, default=Path('artifacts/annotation_fields_v3/field_annotations.csv'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--consolidated', type=Path)
    args = parser.parse_args()
    run(args.root / args.source, args.root / args.output_dir, args.root, args.consolidated)
