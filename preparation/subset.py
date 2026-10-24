"""Deterministic development subset from a hash-bound existing question/key package."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarking.common import file_hash, identity, load_keys, load_questions, question_record, strict_json, write_new, utc_now

def subset(parent_manifest, questions_path, keys_path, output, count=10):
    # Verify both parent files and their row mapping before taking the frozen
    # prefix. This cost-test subset is not a fresh random draw or an error sample.
    parent=strict_json(Path(parent_manifest).read_text());questions=load_questions(questions_path);keys=load_keys(keys_path,questions)
    artifacts=parent['artifacts']
    if artifacts['questions']['sha256']!=file_hash(questions_path) or artifacts['keys']['sha256']!=file_hash(keys_path): raise ValueError('parent package hash mismatch')
    if type(count) is not int or count<1 or count>len(questions) or any(q.split!='development' for q in questions): raise ValueError('subset requires eligible development parent')
    output=Path(output)
    if output.exists(): raise FileExistsError('fresh subset directory required')
    if parent.get('ordered_selected_ids') != [q.item_id for q in questions] or len(parent.get('selected_source_rows',[]))!=len(questions): raise ValueError('parent IDs/source-row mapping mismatch')
    output.mkdir(parents=True)
    selected=questions[:count] # Freeze parent randomized order; no outcomes read.
    qpath=output/'questions.jsonl';kpath=output/'answers/keys.jsonl'
    write_new(qpath,[question_record(q) for q in selected],jsonl=True)
    write_new(kpath,[keys[identity(q)] for q in selected],jsonl=True);kpath.chmod(0o600)
    content={'schema_version':1,'type':'subset_content','algorithm':'first N items in frozen parent selection order; no outcomes',
        'count':count,'selected_source_rows':parent['selected_source_rows'][:count],
        'ordered_selected_ids':[q.item_id for q in selected],
        'parent_questions_sha256':file_hash(questions_path),'parent_keys_sha256':file_hash(keys_path),
        'parent_source_sha256':parent['source_sha256'],'parent_algorithm':parent['algorithm'],'parent_seed':parent['seed'],
        'questions_sha256':file_hash(qpath),'keys_sha256':file_hash(kpath)}
    write_new(output/'content-manifest.json',content)
    manifest={'schema_version':2,'kind':'development_subset','algorithm':content['algorithm'],'created_at':utc_now(),
        'count':count,'selected_source_rows':parent['selected_source_rows'][:count],'ordered_selected_ids':[q.item_id for q in selected],
        'parent_manifest_sha256':file_hash(parent_manifest),'parent_questions_sha256':file_hash(questions_path),
        'parent_keys_sha256':file_hash(keys_path),'content_manifest_sha256':file_hash(output/'content-manifest.json'),
        'artifacts':{'questions':{'path':'questions.jsonl','sha256':file_hash(qpath)},'keys':{'path':'answers/keys.jsonl','sha256':file_hash(kpath)}}}
    write_new(output/'manifest.json',manifest)
    # Keys are resolved only during preparation/offline scoring. The public
    # handoff binds their digest without exposing a key-file path to inference.
    write_new(output/'inference-manifest.json',{'schema_version':2,'type':'inference_input','provenance':'prepared',
        'benchmark':selected[0].benchmark,'split':'development','count':count,
        'questions_sha256':file_hash(qpath),'keys_sha256':file_hash(kpath),
        'preparation_manifest_sha256':file_hash(output/'manifest.json'),
        'content_manifest_sha256':file_hash(output/'content-manifest.json'),
        'ordered_ids':[q.item_id for q in selected],'source_rows':parent['selected_source_rows'][:count]})
    return output/'manifest.json'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent-manifest','questions','keys','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--count',type=int,default=10);a=p.parse_args()
    try: print(subset(a.parent_manifest,a.questions,a.keys,a.output,a.count))
    except (OSError,ValueError,KeyError) as exc: p.exit(1,'Blocked: '+str(exc)+'\n')

if __name__=='__main__': main()
