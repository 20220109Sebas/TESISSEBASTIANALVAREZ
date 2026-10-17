import json

from scripts.show_source_reading_delta import display


def test_only_exact_spans_reused_and_changed_number_remains(tmp_path, capsys):
    prefix = ' '.join(f'word{i}' for i in range(55))
    old = prefix + ' 20 años'
    new = prefix + ' 25 años'
    sources = [{'source': {'sample_number': n},
        'pages': [{'page': 1, 'text': t, 'ocr_used': False}]}
        for n, t in [(1, old), (2, new)]]
    path = tmp_path / 'source.json'
    review = tmp_path / 'review.json'
    path.write_text(json.dumps(sources))
    review.write_text(json.dumps({'documents': [{'sample_number': 1,
        'coverage': {'full_pdf_semantically_read': True, 'unreviewed_pages': []}}]}))
    display(path, review, 2, 1, 1)
    output = capsys.readouterr().out
    assert 'EXACT REUSE: sample 1, page 1' in output
    assert '25 años' in output
    assert '20 años' not in output


def test_partial_reading_cannot_hide_repeated_text(tmp_path, capsys):
    text = ' '.join(f'word{i}' for i in range(55))
    source = tmp_path / 'source.json'
    review = tmp_path / 'review.json'
    source.write_text(json.dumps([{'source': {'sample_number': n},
        'pages': [{'page': 1, 'text': text, 'ocr_used': False}]} for n in [1, 2]]))
    review.write_text(json.dumps({'documents': [{'sample_number': 1,
        'coverage': {'full_pdf_semantically_read': False, 'unreviewed_pages': [1]}}]}))
    display(source, review, 2, 1, 1)
    output = capsys.readouterr().out
    assert 'EXACT REUSE' not in output
    assert text in output
