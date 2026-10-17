"""Synthetic negative controls for the independent audit (standard library only)."""
import contextlib
import csv
import io
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
_script = Path(__file__).resolve().parents[1] / 'scripts' / 'audit_annotation_fields.py'
_spec = importlib.util.spec_from_file_location('independent_field_audit', _script)
_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_audit)
run, sha = _audit.run, _audit.sha

class AuditControls(unittest.TestCase):
    def check_case(self, field='amount', prediction='0.0', bad_hash=False, bad_ocid=False, description='SUPERVISIÓN Ñ Á'):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = {'uri': 'https://example.test/release/exact', 'releases': [{'ocid': 'source-id', 'id': 'exact', 'tender': {'value': {'amount': 0, 'currency': 'PEN'}, 'description': description}}]}
            raw = json.dumps(payload, ensure_ascii=False).encode()
            (root / 'release.json').write_bytes(raw)
            row = dict(opportunity_id='wrong-id' if bad_ocid else 'source-id', field=field, prediction=prediction, source_url='https://example.test/record/source-id', release_path='release.json', release_sha256='0'*64 if bad_hash else sha(raw), gold_value='existing human value', gold_state='observed', annotator_id='real-person', reviewed_at='2025-01-01', evidence_fragment='existing evidence')
            with (root / 'input.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)
            before = (root / 'input.csv').read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                run(root / 'input.csv', root / 'output', root)
            self.assertEqual(before, (root / 'input.csv').read_bytes())
            return json.loads((root / 'output/row_results.json').read_text())[0]

    def test_zero_is_observed_and_human_input_is_unchanged(self):
        result = self.check_case()
        self.assertEqual(result['status'], 'match')
        self.assertEqual(result['expected_state'], 'observed')

    def test_wrong_hash_rejected(self):
        self.assertEqual(self.check_case(bad_hash=True)['status'], 'mismatch')

    def test_multiline_csv_preserved(self):
        result = self.check_case(field='description', prediction='primera\nsegunda', description='primera\nsegunda')
        self.assertEqual(result['prediction'], 'primera\nsegunda')
        self.assertEqual(result['status'], 'match')

    def test_wrong_ocid_rejected(self):
        self.assertEqual(self.check_case(bad_ocid=True)['status'], 'error')

    def test_wrong_prediction_rejected(self):
        self.assertEqual(self.check_case(prediction='609172.5')['status'], 'mismatch')

    def test_utf8_and_whitespace_case_normalization(self):
        result = self.check_case(field='description', prediction=' supervisión   ñ á ')
        self.assertEqual(result['status'], 'match')
        self.assertEqual(result['expected_value'], 'SUPERVISIÓN Ñ Á')

if __name__ == '__main__':
    unittest.main()
