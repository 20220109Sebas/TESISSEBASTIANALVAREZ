"""Cache current OCDS records for the 26 inaccessible original documents."""
import argparse
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_experience_documents import sha
from scripts.fetch_pending_experience_sources import check_code


class RecordRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        p = urlparse(newurl)
        if p.scheme != 'https' or p.hostname != 'contratacionesabiertas.oece.gob.pe' or p.username or p.password:
            raise ValueError('Unexpected OCDS record redirect')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def fetch_records(output):
    if output.exists():
        raise ValueError('Choose a new metadata output directory')
    plan_path = ROOT / 'artifacts/experience_pending26_plan_v1/pending_plan.json'
    manifest_path = plan_path.parent / 'manifest.json'
    if sha(plan_path) != json.loads(manifest_path.read_text())['output_sha256'][plan_path.name]:
        raise ValueError('Original pending plan changed')
    plan = json.loads(plan_path.read_text())
    check_code(plan)
    inputs = {str(plan_path): sha(plan_path), str(manifest_path): sha(manifest_path)}
    output.mkdir(parents=True)
    opener = urllib.request.build_opener(RecordRedirect())
    rows = []
    for original in plan['pending_sources']:
        ocid = original['opportunity_id']
        if not re.fullmatch(r'ocds-[A-Za-z0-9-]+', ocid):
            raise ValueError('Unexpected OCID')
        url = 'https://contratacionesabiertas.oece.gob.pe/api/v1/record/' + ocid
        row = {'opportunity_id': ocid, 'original_document': original, 'record_url': url,
               'started_at_utc': datetime.now(timezone.utc).isoformat()}
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Thesis-document-audit/1.0', 'Accept': 'application/json'})
            with opener.open(req, timeout=30) as response:
                raw = response.read(4 * 1024 * 1024 + 1)
                row['http_status'] = response.status
            if len(raw) > 4 * 1024 * 1024:
                row['status'] = 'record_exceeds_4mb_limit'
            else:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ValueError('Unexpected OCDS package type')
                p = output / (hashlib.sha256(url.encode()).hexdigest() + '.json')
                p.write_bytes(raw)
                row.update(status='record_saved', file_path=str(p), bytes=len(raw), sha256=sha(p))
        except urllib.error.HTTPError as exc:
            row.update(status='http_error', http_status=exc.code, error=str(exc))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            row.update(status='record_access_or_parse_error', error=str(exc))
        row['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        rows.append(row)
        (output / 'record_access.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'record': len(rows), 'status': row['status']}), flush=True)
    check_code(plan)
    summary = {'records_attempted': len(rows), 'statuses': dict(Counter(r['status'] for r in rows)),
               'extraction_rules_unchanged': True, 'labels_read': False, 'original_sources_replaced': False,
               'historical_content_proven': False, 'no_automatic_retries': True}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'manifest.json').write_text(json.dumps({'input_sha256': inputs,
        'code_sha256': {str(Path(__file__)): sha(__file__)},
        'output_sha256': {p.name: sha(p) for p in sorted(output.iterdir())}}, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(fetch_records(a.output.resolve())))
