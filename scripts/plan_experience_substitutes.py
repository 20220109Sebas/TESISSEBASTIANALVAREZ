"""Freeze a documented amendment with one metadata-selected substitute per process."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'src'))
from tesis_recommender.substitute_sources import select_substitute
from scripts.fetch_experience_documents import sha
from scripts.fetch_pending_experience_sources import check_code


def prepare(records_dir, output):
    if output.exists(): raise ValueError('Choose a new substitution plan directory')
    path = records_dir / 'record_access.json'
    manifest = json.loads((records_dir / 'manifest.json').read_text())
    if sha(path) != manifest['output_sha256'][path.name]: raise ValueError('Record access changed')
    inputs = {str(path):sha(path), str(records_dir / 'manifest.json'):sha(records_dir / 'manifest.json')}
    old_plan_path = ROOT / 'artifacts/experience_pending26_plan_v1/pending_plan.json'
    old_plan = json.loads(old_plan_path.read_text());check_code(old_plan)
    inputs[str(old_plan_path)] = sha(old_plan_path)
    selected, decisions = [], []
    for access in json.loads(path.read_text()):
        if access['status'] != 'record_saved':
            decisions.append({'opportunity_id':access['opportunity_id'], 'status':'no_record_available', 'record_access_status':access['status']})
            continue
        if sha(access['file_path']) != access['sha256']: raise ValueError('Current record changed')
        inputs[access['file_path']] = access['sha256']
        decision = select_substitute(json.loads(Path(access['file_path']).read_text()), access['original_document'])
        decision.update(opportunity_id=access['opportunity_id'], current_record_url=access['record_url'], current_record_sha256=access['sha256'])
        if decision['selected']:
            item = dict(decision['selected'], current_record_source_uri=access['record_url'],
                        current_record_file_path=access['file_path'], current_record_sha256=access['sha256'],
                        current_record_retrieved_at_utc=access['finished_at_utc'])
            selected.append(item)
        decisions.append(decision)
    if len({r['url'] for r in selected}) != len(selected): raise ValueError('Substitute URL shared by different opportunities')
    inputs.update(old_plan['frozen_rule_and_runner_sha256'])
    output.mkdir(parents=True)
    plan = {'purpose':'user_authorized_same_process_substitution_amendment',
            'authorization':'User approved looking for substitute sources on 2026-10-10. Original failed sources remain preserved.',
            'pending_sources':selected, 'original_requested_processes':len(decisions), 'batches':[],
            'frozen_rule_and_runner_sha256':old_plan['frozen_rule_and_runner_sha256'],
            'input_sha256':inputs, 'labels_read':False, 'automatic_retries':False,
            'historical_content_proven':False, 'ranking_eligible':False,
            'selection':'same_OCID_metadata_role_temporal_status_date_identity_no_text_or_outcome_selection'}
    for i in range(0,len(selected),5):
        folder=output/'batches'/f'batch_{i//5+1:02d}';folder.mkdir(parents=True)
        p=folder/'access_plan.json'
        part={'purpose':plan['purpose'], 'selected':selected[i:i+5], 'input_sha256':inputs,
              'allowed_host':'prod1.seace.gob.pe', 'limits':{'documents':5, 'max_bytes_per_document':10*1024*1024,
              'timeout_seconds':30, 'max_pages_per_document':120, 'workers':1, 'ocr':False},
              'labels_read':False,'automatic_retries':False,'historical_content_proven':False,'ranking_eligible':False}
        p.write_text(json.dumps(part,ensure_ascii=False,indent=2)+'\n')
        (folder/'manifest.json').write_text(json.dumps({'output_sha256':{p.name:sha(p)}},indent=2)+'\n')
        plan['batches'].append({'path':str(p),'sha256':sha(p),'sources':len(part['selected'])})
    summary={'original_processes':len(decisions),'substitutes_selected':len(selected),'processes_without_substitute':len(decisions)-len(selected),
             'roles':dict(Counter(r['document_role'] for r in selected)),
             'temporal_metadata':dict(Counter(r['temporal_metadata_status'] for r in selected)),
             'source_amendment_not_original_recovery':True,'historical_content_proven':False,'rankings_changed':False}
    for name,obj in [('pending_plan.json',plan),('decisions.json',decisions),('summary.json',summary)]:
        (output/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    for p,expected in inputs.items():
        if sha(p)!=expected: raise ValueError('Input changed while planning')
    check_code(old_plan)
    (output/'manifest.json').write_text(json.dumps({'input_sha256':inputs,
        'code_sha256':{str(p):sha(p) for p in [Path(__file__),ROOT/'src/tesis_recommender/substitute_sources.py',ROOT/'src/tesis_recommender/annex_sources.py']},
        'output_sha256':{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}},indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,default=ROOT/'artifacts/experience_substitute_records_v1')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.records.resolve(),a.output.resolve())))
