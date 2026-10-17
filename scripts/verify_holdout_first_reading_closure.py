"""Reconcile completed first readings by OCID without creating new readings.

The current compiled inventory supersedes earlier partial notes. Legacy complete
readings remain in their original inventory. This verifier combines their exact
records for provenance checks, never as adjudicated labels or a second pass.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from scripts.audit_holdout_evaluation_readiness import verify_artifact
from scripts.register_browser_experience_sources import dump, protected_outputs, sha, verify_hashes
from scripts.verify_holdout_source_review import build as verify_quotations


def reconcile(rows, compiled, legacy):
    expected = {r['opportunity_id'] for r in rows if r['source_status'] == 'downloaded_unique_pdf'}
    chosen, origin = {}, {}
    for name, inventory in [('legacy', legacy), ('compiled', compiled)]:
        seen = set()
        for doc in inventory['documents']:
            ocid = doc['opportunity_id']
            if ocid in seen:
                raise ValueError('Duplicate OCID within inventory')
            seen.add(ocid)
            c = doc['coverage']
            if c['full_pdf_semantically_read'] is True and c['unreviewed_pages'] == []:
                chosen[ocid] = deepcopy(doc)
                origin[ocid] = name
    if set(chosen) != expected:
        raise ValueError('Full-reading records do not match downloaded PDF set')
    for row in rows:
        if row['opportunity_id'] in expected and row['first_pass_status'] != 'full_supplied_pdf':
            raise ValueError('Progress ledger still contains incomplete downloaded reading')
    return sorted(chosen.values(), key=lambda d: d['sample_number']), origin


def build(progress, legacy_path, source, output):
    if output.exists():
        raise ValueError('Use a new output directory')
    inputs = dict(protected_outputs())
    verify_artifact(progress, inputs)
    verify_artifact(source, inputs)
    inherited = json.loads((progress / 'manifest.json').read_text())['input_sha256']
    if inherited.get(str(legacy_path)) != sha(legacy_path):
        raise ValueError('Legacy inventory is not a protected progress input')
    inputs[str(legacy_path)] = sha(legacy_path)
    for name in ('review_progress.json', 'compiled_source_reviews.json'):
        inputs[str(progress / name)] = sha(progress / name)
    rows = json.loads((progress / 'review_progress.json').read_text())
    documents, origins = reconcile(rows,
        json.loads((progress / 'compiled_source_reviews.json').read_text()),
        json.loads(legacy_path.read_text()))
    review = {'review_kind': 'reconciled_existing_automatic_first_pass_records',
        'reviewer': 'Original document-specific automatic reviewers',
        **{k: False for k in ('human_annotation', 'expert_annotation', 'predictions_consulted',
            'model_rules_consulted', 'second_pass_performed', 'reference_locked', 'evaluation_ready')},
        'documents': documents}
    verify_hashes(inputs)
    output.mkdir(parents=True)
    dump(output / 'first_source_records.json', review)
    verified = verify_quotations((output / 'first_source_records.json').resolve(), source,
        output / 'quotation_verification')
    dump(output / 'inventory_index.json', [{'sample_number': d['sample_number'],
        'opportunity_id': d['opportunity_id'], 'inventory': origins[d['opportunity_id']],
        'physical_pages': d['total_pages']} for d in documents])
    summary = {'downloaded_pdf': len(documents), 'first_readings_complete': len(documents),
        'compiled_records': sum(v == 'compiled' for v in origins.values()),
        'legacy_complete_records': sum(v == 'legacy' for v in origins.values()),
        'literal_quotations_verified': verified['literal_quotations_verified'],
        'new_readings_created_by_verifier': 0, 'second_passes_created': 0,
        'reference_locked': False, 'model_accuracy_evaluated': False,
        'scope': 'existing_reading_coverage_and_literal_provenance_not_semantic_gold'}
    dump(output / 'summary.json', summary)
    verify_hashes(inputs)
    dump(output / 'manifest.json', {'input_sha256': inputs, 'script_sha256': sha(__file__),
        'output_sha256': {str(p.relative_to(output)): sha(p)
            for p in sorted(output.rglob('*')) if p.is_file()}})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('progress', 'legacy', 'source', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.progress.resolve(), args.legacy.resolve(),
        args.source.resolve(), args.output.resolve())))
