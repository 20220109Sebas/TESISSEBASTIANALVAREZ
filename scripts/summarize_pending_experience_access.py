"""Close the fixed 26-source access attempt without turning failures into negatives."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_experience_documents import sha
from scripts.fetch_pending_experience_sources import check_code


def summarize(output):
    if output.exists():
        raise ValueError("Choose a new summary directory")
    inputs = {}
    def read(directory, name):
        folder = ROOT / 'artifacts' / directory
        p, m = folder / name, folder / 'manifest.json'
        if sha(p) != json.loads(m.read_text())["output_sha256"][name]:
            raise ValueError("Summary input changed")
        inputs[str(p)] = sha(p);inputs[str(m)] = sha(m)
        return json.loads(p.read_text())
    plan = read('experience_pending26_plan_v1', 'pending_plan.json')
    check_code(plan)
    probes = read('experience_pending26_probe_v1', 'head_results.json')
    accesses = [r for i in range(1,7) for r in read(f'experience_pending26_access_v1/batch_{i:02d}', 'access_results.json')]
    coverage = read('experience_document_coverage_v3', 'summary.json')
    variable_summary = read('experience_formulation_candidates_v3', 'summary.json')
    planned = [r['url'] for r in plan['pending_sources']]
    if planned != [r['url'] for r in accesses] or planned != [r['url'] for r in probes]:
        raise ValueError("Access or diagnostic coverage differs from fixed sources")
    by_probe = {r['url']: r for r in probes}
    pending = [{"opportunity_id": r['opportunity_id'], 'sample_number':r['sample_number'],
                'document_id':r['document_id'], 'title':r['title'], 'source_url':r['url'],
                'initial_source_uri':r['initial_source_uri'], 'GET_status':r.get('http_status'),
                'GET_started_at_utc':r['retrieval_started_at_utc'],
                'HEAD_status':by_probe[r['url']].get('http_status'),
                'HEAD_started_at_utc':by_probe[r['url']]['started_at_utc'],
                'retrieval_state':'no_complete_pdf_available',
                'rule_application_state':'blocked_without_source_document',
                'negative_label':False, 'automatic_retry_authorized_by_this_plan':False}
               for r in accesses if r['status']!='downloaded_pdf']
    complete = sum(r['status']=='downloaded_pdf' for r in accesses)
    if complete:
        raise ValueError("This failure summary cannot omit newly available PDFs")
    summary = {'requested_sources':len(planned), 'sources_attempted':len(accesses),
               'new_complete_pdfs':complete,'new_documents_with_rules_applied':0,
               'GET_statuses':dict(Counter(str(r.get('http_status')) for r in accesses)),
               'HEAD_diagnostic_statuses':dict(Counter(str(r.get('http_status')) for r in probes)),
               'retrieval_complete':False,'rules_unchanged':True,'fixed_sample_size':coverage['sample_size'],
               'complete_pdfs_cumulative':coverage['complete_pdf_opportunities'],
               'sources_without_complete_pdf':coverage['opportunities_without_complete_pdf'],
               'unchanged_existing_formulation_candidates':variable_summary['additional_literal_candidates'],
               'negative_labels_created':False,'f1_estimable':False,'rankings_changed':False,
               'new_ocr_performed':False,'new_embeddings_generated':False,
               'pending_action':'obtain_access_to_same_source_documents_or_explicitly_approve_a_source_amendment'}
    for p, expected in inputs.items():
        if sha(p)!=expected:raise ValueError('Input changed during summary')
    check_code(plan)
    output.mkdir(parents=True)
    for name,obj in [('pending_sources.json',pending),('summary.json',summary)]:
        (output/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps({'input_sha256':inputs,
        'code_sha256':{str(Path(__file__)):sha(__file__)},
        'output_sha256':{p.name:sha(p) for p in sorted(output.iterdir())}},indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(summarize(a.output.resolve())))
