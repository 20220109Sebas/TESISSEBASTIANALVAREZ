from copy import deepcopy

import pytest

from scripts.verify_holdout_first_reading_closure import reconcile


def document(ocid='one', complete=True):
    return {'opportunity_id': ocid, 'sample_number': 1,
        'coverage': {'full_pdf_semantically_read': complete,
            'unreviewed_pages': [] if complete else [1]}}


def row(ocid='one', status='full_supplied_pdf'):
    return {'opportunity_id': ocid, 'source_status': 'downloaded_unique_pdf',
        'first_pass_status': status}


def test_preserves_legacy_complete_and_supersedes_with_current_complete():
    legacy = {'documents': [document(), document('two', False)]}
    compiled = {'documents': [document('two')]}
    saved = deepcopy((legacy, compiled))
    docs, origins = reconcile([row(), row('two')], compiled, legacy)
    assert {d['opportunity_id'] for d in docs} == {'one', 'two'}
    assert origins == {'one': 'legacy', 'two': 'compiled'}
    assert (legacy, compiled) == saved
    docs, origins = reconcile([row()], {'documents': [document()]},
        {'documents': [document()]})
    assert len(docs) == 1
    assert origins == {'one': 'compiled'}


@pytest.mark.parametrize('compiled,rows', [
    ({'documents': [document('different')]}, [row()]),
    ({'documents': [document(complete=False)]}, [row()]),
    ({'documents': [document(), document()]}, [row()]),
    ({'documents': [document()]}, [row(status='partial')]),
])
def test_rejects_missing_mismatched_duplicate_or_partial_progress(compiled, rows):
    with pytest.raises(ValueError):
        reconcile(rows, compiled, {'documents': []})
