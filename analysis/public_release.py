"""Portable public-data loading, presentation export and optional SAP reproduction.

Uses the unchanged scientific kernels and frozen presentation implementation.
Raw request/response logs are neither distributed nor read by this module.
"""
from copy import deepcopy
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_gzip(path, *, lines=False):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()] if lines else json.load(stream)


def checked_data(root=ROOT):
    """Authenticate the explicit public derivative before reading its values."""
    root = Path(root)
    manifest = json.loads((root/'provenance/data_manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        path = root/relative
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or sha256(path) != expected:
            raise ValueError('Public data missing or changed: '+relative)
    return manifest


def load_public_results(root=ROOT):
    checked_data(root)
    data = read_gzip(Path(root)/'data/reference_results.json.gz')
    if data['type'] != 'public_benchmark_reference' or not data['summary']['study_complete']:
        raise ValueError('Complete public benchmark reference required')
    return data


def validate_public_manifest(manifest):
    from analysis.methods.expanded import ROSTER, DECISIONS
    from benchmarking.configuration_ids import CURRENT_NAMESPACE
    if (manifest.get('public_derivative') != 'structured-results-v1'
            or manifest.get('configuration_namespace') != CURRENT_NAMESPACE
            or [c['config_id'] for c in manifest['configurations']] != list(ROSTER)):
        raise ValueError('Public manifest requires the exact 28-configuration roster')
    items = manifest['items']
    if len(items) != 1273 or len({q['item_id'] for q in items}) != 1273 or any(q['labels'] != list('ABCD') for q in items):
        raise ValueError('Public manifest requires the original 1,273 four-option items')
    expected = {(cid,rep) for cid in ROSTER for rep in ((1,2,3) if cid in DECISIONS else (1,))}
    runs = manifest['runs']
    if len(runs) != 54 or {(r['config_id'],r['replicate']) for r in runs} != expected or len({r['run_id'] for r in runs}) != 54:
        raise ValueError('Public run roster changed')
    for r in runs:
        if (r['role'],r['experiment_id']) != (('primary','E1') if r['replicate']==1 else ('secondary','E2')):
            raise ValueError('Public run role changed')
    return manifest


def validate_public_record(record, manifest):
    from common import TERMINAL_STATUSES
    identity = record['identity']
    if record['status'] not in TERMINAL_STATUSES:
        raise ValueError('Unknown public terminal status')
    if (identity['benchmark'] != 'medqa' or identity['split'] != 'evaluation'
            or identity['item_id'] not in manifest['_public_item_ids']
            or manifest['_public_runs'].get(identity['run_id']) != identity['config_id']):
        raise ValueError('Public record identity is outside the complete study')
    if (record['status']=='valid') != (record['selected_label'] in 'ABCD' if isinstance(record['selected_label'],str) and len(record['selected_label'])==1 else False):
        raise ValueError('Public validity and selected label disagree')
    return record


def load_structured(root=ROOT):
    """Rebuild only analysis input records, with first-attempt outcomes retained."""
    from common import attempt_identity
    from benchmarking.configuration_ids import CURRENT_NAMESPACE
    root = Path(root)
    checked_data(root)
    meta = json.loads((root/'data/study.json').read_text())
    outcomes = read_gzip(root/'data/outcomes.jsonl.gz', lines=True)
    call_rows = read_gzip(root/'data/calls.jsonl.gz', lines=True)
    calls = {r['attempt_id']:r for r in call_rows}
    if len(calls)!=69318 or len(call_rows)!=69318 or len(outcomes)!=68742:
        raise ValueError('Public call/outcome coverage changed')
    manifest = {'schema_version':3,'execution_workflow':'analysis-reporting-ids-v1',
        'public_derivative':'structured-results-v1','configuration_namespace':CURRENT_NAMESPACE,
        'study_id':meta['study_id'],'provenance':'definitive','benchmark':'medqa','split':'evaluation',
        'items':meta['items'],'configurations':meta['configurations'],'runs':meta['runs'],
        'bindings':meta['source_bindings'],'source_packages':[],'missing_configurations':[],
        'reviewed_decision_settings':meta['methods']['decision_models'],
        '_public_item_ids':{q['item_id'] for q in meta['items']},
        '_public_runs':{r['run_id']:r['config_id'] for r in meta['runs']}}
    validate_public_manifest(manifest)
    terminal, identities, used = [], set(), set()
    for row in outcomes:
        history = [calls[aid] for aid in row['call_ids']]
        identity = row['identity']
        if not history or [c['retry_index'] for c in history]!=list(range(len(history))) or len(history)>11:
            raise ValueError('Public retry indices are not contiguous')
        if any(c['record']['identity']!=identity for c in history):
            raise ValueError('Public history crosses question/model/run identities')
        valid = [c for c in history if c['record']['status']=='valid']
        selected = valid[0] if valid else history[-1]
        if (valid and selected is not history[-1]) or (not valid and len(history)!=11):
            raise ValueError('Public history violates first-valid or exhausted-retry selection')
        selection = row['selection']
        original = history[0]
        if (selection['selected_attempt_id'] != selected['attempt_id']
                or selection['original_attempt_id'] != original['attempt_id']
                or selection['selected_retry_index'] != selected['retry_index']
                or (row['status'],row['selected_label']) != (selected['record']['status'],selected['record']['selected_label'])
                or (row['original_status'],row['original_selected_label']) != (original['record']['status'],original['record']['selected_label'])):
            raise ValueError('Public selected/original attempt fields disagree')
        # Historical call rows used source identities; selected rows use reporting IDs.
        source_record = lambda c:{**deepcopy(c['record']),'identity':deepcopy(c['source_identity'])}
        record = {**deepcopy(selected['record']),'source_identity':row['source_identity'],
            'source_namespace':row['source_namespace'],'original_attempt':source_record(original),
            'analysis_selection':deepcopy(selection),
            'analysis_attempts':[{'record':source_record(c),'retry_index':c['retry_index'],'source_package':c['source_package']} for c in history]}
        validate_public_record(record,manifest)
        ident = attempt_identity(record)
        if ident in identities:raise ValueError('Duplicate public selected identity')
        identities.add(ident);used.update(row['call_ids']);terminal.append(record)
    expected={(manifest['benchmark'],manifest['split'],q['item_id'],r['config_id'],r['run_id']) for q in manifest['items'] for r in manifest['runs']}
    if identities!=expected or used!=set(calls):raise ValueError('Public study coverage is incomplete')
    coverage={'expected':68742,'terminal':68742,'complete':True,'unstarted':[],'indeterminate':[],
        'collection_conditions':{},'last_event_sha256':None,'missing_configurations':[],
        'pending_outcomes':0,'pending_recovery':[],'missing_outcome_reasons':{}}
    return meta,manifest,terminal,coverage,outcomes


def upstream_keys(path, meta):
    """Reconstruct keys locally from the exact upstream file; never infer them."""
    from common import Option,Question,question_hash
    path = Path(path)
    if sha256(path)!=meta['source_test_sha256']:
        raise ValueError('Upstream MedQA test file differs from the evaluated source SHA-256')
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows)!=1273:raise ValueError('Expected 1,273 upstream test questions')
    keys={}
    for source,item in zip(rows,meta['items']):
        question=Question('medqa',item['item_id'],'evaluation',source['question'],
            tuple(Option(k,v) for k,v in source['options'].items()))
        if question_hash(question)!=item['content_sha256']:
            raise ValueError('Upstream question/options/order disagree: '+item['item_id'])
        label=source['answer_idx']
        if label not in 'ABCD' or len(label)!=1:raise ValueError('Invalid upstream answer key')
        keys[('medqa','evaluation',item['item_id'])]={'reference_answer':label}
    return keys


def scientific_projection(summary):
    result=deepcopy(summary)
    for key in ['bindings','source_packages','collection_conditions','inputs']:
        result.pop(key,None)
    result['coverage'].pop('collection_conditions',None)
    return result


def recompute(root, medqa, output):
    """Explicit optional full numerical reproduction with original scientific kernels."""
    from analysis.methods.pipeline import analyze
    meta,manifest,terminal,coverage,outcomes=load_structured(root)
    keys=upstream_keys(medqa,meta)
    for row in outcomes:
        key=keys[('medqa','evaluation',row['identity']['item_id'])]['reference_answer']
        if (row['correct']!=(row['status']=='valid' and row['selected_label']==key)
                or row['original_correct']!=(row['original_status']=='valid' and row['original_selected_label']==key)):
            raise ValueError('Stored public correctness disagrees with reconstructed upstream keys')
    output=Path(output)
    if output.exists():raise FileExistsError('Choose a new reproduction output directory')
    summary=analyze(manifest,terminal,coverage,keys,seed=20261008,methods=meta['methods'])
    expected=load_public_results(root)['summary']
    observed=scientific_projection(summary)
    if observed!=expected:
        # Preserve the result for a field-level review; never silently change reference values.
        output.mkdir(parents=True)
        (output/'recomputed_summary.json').write_text(json.dumps(observed,ensure_ascii=False,indent=2)+'\n')
        raise ValueError('Recomputed summary differs from the reference; inspect retained output')
    output.mkdir(parents=True)
    (output/'recomputed_summary.json').write_text(json.dumps(observed,ensure_ascii=False,indent=2)+'\n')
    (output/'verification.json').write_text(json.dumps({'scientific_summary_identical':True,'source_result_sha256':meta['source_result_sha256']},indent=2)+'\n')
    print('All scientific summary fields match the accepted reference:',output)


def export_public_report(folder, figures, record, *, root=ROOT, source_before=None):
    """Export reproduced graphics and copy the verified public table projections."""
    from analysis.support.source_record import presentation_code_record
    checked_data(root)
    if source_before is None or presentation_code_record(root)!=source_before:
        raise ValueError('Presentation source changed; restart and render again')
    folder=Path(folder)
    if folder.exists():raise FileExistsError('Choose a new output directory; existing exports are retained')
    folder.mkdir(parents=True)
    files={}
    for name,fig in figures.items():
        for fmt in ['pdf','svg','png']:
            path=folder/(name+'.'+fmt);fig.savefig(path,format=fmt,dpi=300);files[path.name]=sha256(path)
    for source in (Path(root)/'results/verified_exports').glob('*.csv'):
        target=folder/source.name;shutil.copyfile(source,target);files[target.name]=sha256(target)
    expected=json.loads((Path(root)/'provenance/presentation.json').read_text())
    for name,row in expected['expected_main_pngs'].items():
        if files[name+'.png']!=row['png_sha256']:
            raise ValueError('Main PNG differs from the frozen preview: '+name)
    provenance={'source_result_sha256':expected['source_result_sha256'],
        'source_presentation_freeze_sha256':expected['source_freeze_sha256'],
        'presentation_code':presentation_code_record(root),'settings':record['settings'],
        'plotted_values':{n:r['plotted_values'] for n,r in record['figures'].items()},
        'tables':'Exact copies of verified, disclosure-filtered public CSV projections',
        'files':files}
    if presentation_code_record(root)!=source_before:
        raise ValueError('Public presentation source changed during export')
    (folder/'public_export_manifest.json').write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+'\n')
    return folder


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('verify',help='Check public file hashes, complete coverage and retry selection')
    reproduction=sub.add_parser('recompute',help='Recalculate SAP results using locally acquired MedQA keys')
    reproduction.add_argument('--medqa-test',required=True,type=Path)
    reproduction.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.command=='verify':
        _,_,terminal,_,_=load_structured(ROOT)
        print('Verified 68,742 outcomes, 69,318 calls, 28 configurations and 54 runs; no model calls.')
    else:recompute(ROOT,args.medqa_test,args.output)


if __name__=='__main__':main()
