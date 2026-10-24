"""Manual-only expanded decision collection: readiness, development and evaluation."""
import argparse
import base64
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import platform
import random
import socket
import sys
import threading
import time
import tomllib
import urllib.error

from benchmarking import adapters, native_decisions
from benchmarking.common import (ROOT, artifact_ref, empty_probability, file_hash, fingerprint,
    load_questions, measurement, question_hash, strict_json, utc_now, write_new)
from benchmarking.configuration_ids import CURRENT_NAMESPACE, EXTENSION_IDS
from benchmarking.extension_contract import (WORKFLOW, SCHEMA, SOURCES, Ledger, attempt_id,
    config_errors, configurations, read_extension_package, runs)
from benchmarking.run import append_event, _graceful_signals, _retry_after_seconds, validate_input_manifest

CONFIG = ROOT/'benchmarking/extension.toml'


def environment():
    return {'python': sys.version, 'implementation': platform.python_implementation(),
            'platform': platform.platform(), 'transport': 'stdlib urllib; redirects/retries disabled'}


def readiness(config, *, ids=None, questions=None, base=ROOT, development=False):
    """Review evidence is recorded in config; no private approval/authorization JSON."""
    problems = config_errors(config)
    if not config['execution'].get('pacing_confirmed'):
        problems.append('confirm accepted five-second per-provider start spacing')
    if not development and questions is None:
        problems.append('evaluation question file required for complete-request preflight')
    for cid in ids or config['model_roster']:
        model = config['models'][cid]
        try:
            native_decisions.validate_model(model)
        except ValueError as exc:
            problems.append(cid+': '+str(exc))
        if development:
            continue  # Explicit development calls establish the missing route evidence.
        verified = model.get('verification', {})
        if questions is not None:
            try:
                for q in questions:
                    native_decisions.build_request(q, model, config['prompt'])
            except (ValueError, KeyError, TypeError) as exc:
                problems.append(cid+': complete-request preflight failed ('+str(exc)+')')
        for field in ('route','controls','schema','full_input'):
            if verified.get(field) is not True:
                problems.append(cid+': unverified '+field)
        try:
            receipt_path = base/verified.get('development_receipt', '')
            manifest, records, coverage = read_extension_package(receipt_path)
            if (manifest['provenance'] != 'development' or not coverage['complete'] or
                file_hash(receipt_path/'events.jsonl') != verified.get('development_receipt_sha256')):
                raise ValueError('complete, hash-bound live development receipt required')
            record = next(r for r in records if r['identity']['config_id'] == cid)
            saved = manifest['models'][cid]
            fields = ('model_id','surface','endpoint','route','provider_slug','expected_returned_model',
                      'request_settings','response_schema')
            if record['status'] != 'valid' or any(saved[k] != model[k] for k in fields):
                raise ValueError('development transport/identity differs from evaluation')
            saved_config = tomllib.loads((receipt_path/'source/config.toml').read_text())
            if saved_config['prompt'] != config['prompt']:
                raise ValueError('development prompt differs from evaluation')
            if not record['probability']['valid']:
                raise ValueError('development evidence must demonstrate four eligible native probabilities')
        except (OSError, ValueError, KeyError, StopIteration) as exc:
            problems.append(cid+': development receipt unavailable/incompatible ('+type(exc).__name__+')')
        # Field semantics can remain explicitly unavailable; this blocks confidence,
        # not valid-answer collection, under SAP v1.7.
        # Optional detailed evidence, when the route review calls for token counts.
        # Supplied audits must still match every current request and their digest.
        if questions is not None and verified.get('input_audit'):
            try:
                path = base/verified.get('input_audit', '')
                audit = strict_json(path.read_text())
                if file_hash(path) != verified.get('input_audit_sha256'):
                    raise ValueError('input audit digest mismatch')
                if (audit['config_id'] != cid or audit['model_id'] != model['model_id'] or
                    audit['provider_slug'] != model['provider_slug'] or audit['endpoint'] != model['endpoint'] or
                    not audit['tokenizer_evidence'] or not audit['effective_limit_evidence'] or
                    audit['includes_instruction_and_schema'] is not True or audit['silent_truncation_excluded'] is not True):
                    raise ValueError('route/envelope input-limit evidence required')
                limit = audit['effective_input_limit_tokens']
                if type(limit) is not int or limit < 1:
                    raise ValueError('verified effective limit required')
                rows = audit['questions']
                if len(rows) != len(questions) or len({r['item_id'] for r in rows}) != len(rows):
                    raise ValueError('input audit coverage mismatch')
                lookup = {r['item_id']: r for r in rows}
                for q in questions:
                    row = lookup[q.item_id]
                    payload = native_decisions.build_request(q, model, config['prompt'])
                    if (row['question_sha256'] != question_hash(q) or row['request_fingerprint'] != fingerprint(payload)
                        or type(row['input_tokens']) is not int or not 0 < row['input_tokens'] <= limit):
                        raise ValueError('full input exceeds verified limit or differs from audited request')
            except (OSError, ValueError, KeyError, TypeError):
                problems.append(cid+': full evaluation request/tokenizer/limit audit unavailable or incompatible')
    return problems


def prepare(config, config_path, questions_path, public_path, protocol_path, output, *,
            mock, development=False, ids=None):
    errors = config_errors(config)
    if errors:
        raise ValueError('; '.join(errors))
    ids = list(ids or config['model_roster'])
    if not ids or len(ids) != len(set(ids)) or set(ids)-set(EXTENSION_IDS):
        raise ValueError('unique selected extension IDs required')
    if not development and tuple(ids) != EXTENSION_IDS:
        raise ValueError('evaluation requires the complete extension roster')
    prep = validate_input_manifest(public_path, questions_path)
    questions = load_questions(questions_path)
    if development and (len(questions) != 1 or prep['split'] != 'development'):
        raise ValueError('development requires exactly one prepared development question')
    if not mock:
        if prep['provenance'] != 'prepared':
            raise ValueError('live mode requires prepared inputs')
        if not development and (len(questions) != 1273 or prep['split'] != 'evaluation'):
            raise ValueError('all 1273 evaluation inputs required')
        problems = readiness(config, ids=ids, questions=questions, development=development)
        if problems:
            raise ValueError('\n'.join(problems))
    # Build every complete request before claiming identities or dispatching.
    requests = {(cid,q.item_id): native_decisions.build_request(q,config['models'][cid],config['prompt'])
                for cid in ids for q in questions}
    snapshots = {'config.toml': Path(config_path), 'questions.jsonl': Path(questions_path),
                 'inference-manifest.json': Path(public_path), 'study-protocol.md': Path(protocol_path),
                 'README.md': ROOT/'README.md'}
    # Preserve exact reviewed evidence alongside the source inventory.
    if not mock and not development:
        for cid in ids:
            v = config['models'][cid]['verification']
            if v.get('input_audit'):
                snapshots[f'evidence/{cid}-input-audit.json'] = ROOT/v['input_audit']
            receipt = ROOT/v['development_receipt']
            for path in receipt.rglob('*'):
                if path.is_file() and path.name != '.lock':
                    snapshots[f'evidence/{cid}-development/{path.relative_to(receipt)}'] = path
    code = {name:file_hash(ROOT/name) for name in SOURCES}
    inventory = {'workflow':WORKFLOW, 'code':code, 'environment':environment(),
                 'snapshots':{name:file_hash(path) for name,path in snapshots.items()}}
    study = 'medqa-decision-extension-'+Path(output).name
    order = [q.item_id for q in questions]
    random.Random(20261006).shuffle(order)
    manifest = {'schema_version':SCHEMA,'execution_workflow':WORKFLOW,'config_namespace':CURRENT_NAMESPACE,
        'campaign':config['campaign'],'study_id':study,
        'provenance':'mock' if mock else ('development' if development else 'definitive'),
        'purpose':'development' if development else 'evaluation',
        'benchmark':prep['benchmark'],'split':prep['split'],
        'items':[{'item_id':q.item_id,'content_sha256':question_hash(q),'labels':list(q.labels)} for q in questions],
        'configurations':configurations(config,ids), 'models':{cid:config['models'][cid] for cid in ids},
        'runs':runs(study,ids,development), 'schedule':{'item_order':order,'policy':config['execution']},
        'key_manifest':{'sha256':prep['keys_sha256'],'preparation_sha256':prep['preparation_manifest_sha256']},
        'bindings':{'execution_record':fingerprint(inventory),'code':fingerprint(code),
            'environment':fingerprint(inventory['environment']),'config':fingerprint(config),
            'questions':file_hash(questions_path),'keys':prep['keys_sha256'],
            'input_manifest':file_hash(public_path),'study_protocol':file_hash(protocol_path),
            'prompt':fingerprint(config['prompt'])}}
    return manifest, inventory, snapshots, questions, requests


@contextmanager
def registry_claim(config, manifest, output):
    """Lock the registry throughout collection; package-name changes cannot reset caps."""
    root = ROOT/config['execution']['registry']
    # Mock/development calls cannot claim evaluation identities.
    root = root/manifest['provenance']/manifest['purpose']
    if manifest['purpose'] == 'development':
        # Deliberate fresh development probes may repeat the same question.
        root = root/fingerprint(str(output.resolve()))
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            claims = {fingerprint([CURRENT_NAMESPACE, manifest['benchmark'], manifest['split'], r['config_id'], r['replicate']])
                      for r in manifest['runs']}
            for tid in sorted(claims):
                path = root/(tid+'.json')
                claim = {'run_claim_id':tid,'package':str(output.resolve()),
                         'manifest_sha256':fingerprint(manifest)}
                if path.exists():
                    if strict_json(path.read_text()) != claim:
                        raise ValueError('duplicate campaign claim for original identity: '+tid)
                else:
                    write_new(path,claim)
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _mock(request, model, question):
    body = {'model':model['expected_returned_model'],'provider':model['route'],
            'id':'fictional', 'answers':{'answer':{'type':'choice','choice':'A',
                'probabilities':{'A':0.1,'B':0.2,'C':0.3,'D':0.4},'confidence':7}},
            'usage':{'input_tokens':100,'output_tokens':0,'cost':0}}
    return {'status':200,'headers':{},'body_base64':base64.b64encode(json.dumps(body).encode()).decode()}


class StartGate:
    def __init__(self, interval):
        self.interval = interval
        self.lock = threading.Lock()
        self.next = {}

    def enter(self, provider):
        while True:
            with self.lock:
                delay = self.next.get(provider,0)-time.monotonic()
                if delay <= 0:
                    self.next[provider] = time.monotonic()+self.interval
                    return
            time.sleep(min(delay, .1))


def worker(request, model, q, mock, transport, queued, gate):
    gate.enter(model['route'])
    start = utc_now(); clock = time.monotonic()
    raw = None; failure = None; error = None
    try:
        raw = (transport or _mock)(request,model,q) if mock else native_decisions.dispatch(request,model,600)
    except (TimeoutError,socket.timeout) as exc:
        failure = ('timeout',type(exc).__name__)
    except (urllib.error.URLError,OSError) as exc:
        failure = ('transport_error',type(exc).__name__)
    except BaseException as exc:
        error = exc
    return {'raw':raw,'failure':failure,'error':error,'started_at':start,'finished_at':utc_now(),
            'duration_seconds':time.monotonic()-clock,'queue_wait_seconds':clock-queued,
            'started_monotonic':clock}


def queue_phase(ledger, queued, now):
    """Start session-local waits only when the primary/secondary phase is eligible."""
    pending = [tid for tid in ledger.targets if tid not in ledger.selected]
    role = 'primary' if any(ledger.targets[tid]['role'] == 'primary' for tid in pending) else 'secondary'
    for tid in pending:
        if ledger.targets[tid]['role'] == role:
            queued.setdefault(tid, now)


def save_response(output, adir, ref, result, model, target, lineage):
    if result['error']:
        raise result['error']  # Persisted start stays indeterminate; never automatically resend.
    raw = result['raw']; rawref = None
    if raw is not None:
        write_new(adir/'raw_response.json',raw)
        rawref = artifact_ref(adir/'raw_response.json',output)
        parsed = native_decisions.parse(raw,model,tuple('ABCD'))
    else:
        parsed = {'status':result['failure'][0],'failure_reason':result['failure'][1],
            'selected_label':None,'probability':empty_probability(),'provider_scalars':{
                path:{**measurement(),'semantics':meaning} for path,meaning in model['scalar_semantics'].items()},
            'usage':{},'observed_usd':None,'returned_model':None,'returned_route':None,
            'provider_request_id':None,'effective_controls':None,'finish_reason':None,'final':None,'reasoning':None}
    refs = {'request':ref,'raw_response':rawref,'final':None,'reasoning':None}
    for channel in ('final','reasoning'):
        if parsed[channel] is not None:
            write_new(adir/(channel+'.json'),parsed[channel])
            refs[channel] = artifact_ref(adir/(channel+'.json'),output)
    usage = adapters.usage_record(parsed['usage'],{**model,'adapter':'openai_decisions_gateway'})
    cid = target['identity']['config_id']
    record = {**lineage,'identity':target['identity'], 'source_namespace':CURRENT_NAMESPACE,
        'source_config_id':cid,'current_config_id':cid,
        **{key:parsed[key] for key in ('status','selected_label','failure_reason','probability','provider_scalars')},
        'probability_semantics':model['probability_semantics'],'usage':usage,
        'billing':adapters.billing_record(parsed,usage,model),
        'timing':{key:result[key] for key in ('started_at','finished_at')},
        'route':{'requested_model':model['model_id'],'requested_route':model['route'],
                 **{key:parsed[key] for key in ('returned_model','returned_route','effective_controls','provider_request_id','finish_reason')},
                 'endpoint':model['endpoint'],'provider_slug':model['provider_slug'],
                 'advertised_backend':model['advertised_backend'],
                 'backend_version_status':model['backend_version_status'],'region':model['region']},
        'artifacts':refs}
    for key in ('duration_seconds','queue_wait_seconds'):
        record['timing'][key] = measurement(result[key])
    record['missing_fields'] = {key:'provider_unavailable' for key in
        ('returned_model','returned_route','effective_controls','provider_request_id','finish_reason') if record['route'][key] is None}
    return record


def collect(config_path, questions_path, public_path, protocol_path, output, *, mock,
            development=False, ids=None, resume=False, reviewer=None, reviewed_stops=None, transport=None):
    config = tomllib.loads(Path(config_path).read_text()); output = Path(output).resolve()
    manifest, inventory, snapshots, questions, requests = prepare(config,config_path,questions_path,
        public_path,protocol_path,output,mock=mock,development=development,ids=ids)
    if not mock and not os.environ.get('OPENROUTER_API_KEY'):
        raise ValueError('OPENROUTER_API_KEY unavailable; no package or start created')
    if output.exists() and not resume:
        raise ValueError('destination exists; use exact bound --resume or a fresh development output')
    if resume and not output.exists():
        raise ValueError('resume destination missing')
    with registry_claim(config,manifest,output):
        if not resume:
            output.mkdir(parents=True,exist_ok=False)
            for name,path in {**{n:ROOT/n for n in SOURCES},**snapshots}.items():
                destination = output/'source'/name
                destination.parent.mkdir(parents=True,exist_ok=True)
                with destination.open('xb') as stream:
                    stream.write(path.read_bytes());stream.flush();os.fsync(stream.fileno())
            write_new(output/'provenance.json',inventory)
            write_new(output/'manifest.json',manifest)
            append_event(output/'events.jsonl',{'schema_version':SCHEMA,'manifest_sha256':fingerprint(manifest),'event':'plan'},None)
        saved, ledger = read_extension_package(output,return_ledger=True)
        if saved != manifest:
            raise ValueError('resume source/config/input/environment/protocol bindings changed')
        if ledger.active:
            raise ValueError('indeterminate starts require evidence-based reconciliation; automatic resend forbidden')
        if ledger.drift:
            raise ValueError('identity/schema drift requires review; no automatic continuation')
        base = {'schema_version':SCHEMA,'manifest_sha256':fingerprint(manifest)}
        def event(data):
            row = {**base,**data,'previous_sha256':ledger.previous}
            row['event_sha256'] = fingerprint(row)
            previous = ledger.previous
            ledger.apply(row,output)  # Reject invalid transitions before writing.
            append_event(output/'events.jsonl',{**base,**data},previous)
        if reviewer:
            event({'event':'review','reviewer':reviewer,'reviewed_at':utc_now(),
                   'reviewed_stop_events':sorted(reviewed_stops or [])})
        elif reviewed_stops:
            raise ValueError('reviewer required with reviewed stops')
        interval = config['execution']['min_interval_seconds']
        gate = StartGate(interval)
        next_start = {}
        # Preserve pacing and Retry-After across process restarts.
        for record in ledger.records:
            provider = record['scheduler']['provider']
            now = datetime.now(timezone.utc)
            age = (now-datetime.fromisoformat(record['timing']['started_at'])).total_seconds()
            next_start[provider] = max(next_start.get(provider,0),time.monotonic()+max(0,interval-age))
            rawref = record['artifacts']['raw_response']
            if rawref:
                raw = strict_json((output/rawref['path']).read_text())
                elapsed = (now-datetime.fromisoformat(record['timing']['finished_at'])).total_seconds()
                next_start[provider] = max(next_start[provider],time.monotonic()+_retry_after_seconds(raw,elapsed_seconds=elapsed))
        gate.next.update(next_start)
        qmap = {q.item_id:q for q in questions}; unexpected = None; stop = False
        queued = {}
        with _graceful_signals() as signals, ThreadPoolExecutor(max_workers=10) as pool:
            inflight = {}; next_report = 0
            while True:
                queue_phase(ledger, queued, time.monotonic())
                if signals:
                    stop = True
                if time.monotonic() >= next_report:
                    print(f'{len(ledger.selected)}/{len(ledger.targets)} outcomes; {sum(ledger.counts.values())} starts; '
                          f'{len(inflight)} active; paused={list(ledger.paused)}; drift={sorted(ledger.drift)}',flush=True)
                    next_report = time.monotonic()+30
                candidates = ledger.candidates()
                while not stop and len(inflight) < 10:
                    ready = next((tid for tid in candidates if next_start.get(ledger.provider(tid),0) <= time.monotonic()),None)
                    if ready is None:
                        break
                    try:
                        target = ledger.targets[ready]; cid = target['identity']['config_id']
                        model = config['models'][cid]; provider = model['route']
                        index = ledger.counts.get(ready,0)
                        lineage = {'original_query_id':ready,'attempt_index':index,'attempt_id':attempt_id(ready,index),
                                   'parent_attempt_id':attempt_id(ready,index-1) if index else None}
                        adir = output/'artifacts'/ready/str(index)
                        request = requests[cid,target['identity']['item_id']]
                        write_new(adir/'request.json',request)
                        ref = artifact_ref(adir/'request.json',output)
                        event({'event':'start',**lineage,'request':ref,'provider':provider,'started_at':utc_now()})
                        future = pool.submit(worker,request,model,qmap[target['identity']['item_id']],mock,transport,queued[ready],gate)
                        inflight[future] = (ready,lineage,adir,ref)
                        next_start[provider] = time.monotonic()+interval
                    except BaseException as exc:
                        unexpected = exc;stop = True;break
                    candidates = ledger.candidates()
                if not inflight:
                    if stop or not candidates:
                        break
                    time.sleep(.1);continue
                done,_ = wait(inflight,timeout=.25,return_when=FIRST_COMPLETED)
                completed = []
                for future in done:
                    details = inflight.pop(future)
                    try:
                        result = future.result()
                        completed.append((result['finished_at'],details,result))
                    except BaseException as exc:
                        unexpected = unexpected or exc;stop = True
                for _,details,result in sorted(completed,key=lambda row:row[0]):
                    tid,lineage,adir,ref = details
                    target = ledger.targets[tid]; model = config['models'][target['identity']['config_id']]
                    provider = model['route']
                    try:
                        record = save_response(output,adir,ref,result,model,target,lineage)
                        raw = result['raw']; status = raw.get('status') if raw else None
                        streak = ledger.streaks.get(provider,0)+1 if status == 429 else 0
                        record['scheduler'] = {'provider':provider,'response_status':status,
                            'stop_reason':'provider_rate_limit_stop' if streak >= 3 else None}
                        event({'event':'finish',**lineage,'record':record})
                        queued[tid] = time.monotonic()
                        next_start[provider] = max(next_start.get(provider,0),result['started_monotonic']+interval)
                        if raw:
                            next_start[provider] = max(next_start[provider],time.monotonic()+_retry_after_seconds(raw))
                        with gate.lock:
                            gate.next[provider] = max(gate.next.get(provider,0),next_start[provider])
                        if development and record['status'] != 'valid':
                            stop = True
                    except BaseException as exc:
                        unexpected = unexpected or exc;stop = True
        if unexpected:
            raise unexpected
        return read_extension_package(output)[2]


def input_audit_template(config, cid, questions):
    """Produce request-bound audit rows; never invent tokenizer counts or limits."""
    model = config['models'][cid]
    return {'config_id':cid,'model_id':model['model_id'],'provider_slug':model['provider_slug'],
        'endpoint':model['endpoint'],'tokenizer_evidence':'','effective_limit_evidence':'',
        'effective_input_limit_tokens':None,'includes_instruction_and_schema':False,
        'silent_truncation_excluded':False,
        'questions':[{'item_id':q.item_id,'question_sha256':question_hash(q),
            'request_fingerprint':fingerprint(native_decisions.build_request(q,model,config['prompt'])),
            'input_tokens':None} for q in questions]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('readiness','collect','inspect','input-audit-template'))
    parser.add_argument('--config',type=Path,default=CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--live',action='store_true');mode.add_argument('--mock',action='store_true')
    parser.add_argument('--questions',type=Path)
    parser.add_argument('--input-manifest',type=Path)
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--development',action='store_true')
    parser.add_argument('--configs',nargs='+')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--reviewer')
    parser.add_argument('--reviewed-stops',nargs='*')
    args = parser.parse_args()
    if args.command == 'collect' and not (args.live or args.mock):
        parser.error('collect requires explicit --live or --mock before reading config/input')
    if args.command == 'input-audit-template':
        if not args.questions or not args.configs or len(args.configs) != 1:
            parser.error('--questions and exactly one --configs ID required')
        config = tomllib.loads(args.config.read_text())
        print(json.dumps(input_audit_template(config,args.configs[0],load_questions(args.questions)),indent=2,ensure_ascii=False))
        return 0
    if args.command == 'inspect':
        if not args.output: parser.error('--output required')
        coverage = read_extension_package(args.output)[2]
    elif args.command == 'readiness':
        config = tomllib.loads(args.config.read_text())
        questions = load_questions(args.questions) if args.questions else None
        blockers = readiness(config,ids=args.configs,questions=questions,development=args.development)
        semantics = {cid:model['probability_semantics'] for cid,model in config['models'].items()}
        field_blockers = []
        for cid in args.configs or config['model_roster']:
            model = config['models'][cid]
            if not model['probability_semantics']['reviewed']:
                field_blockers.append(cid+': probability confidence/repeatability field semantics unreviewed')
            field_blockers.extend(cid+': scalar '+path+' semantics unreviewed'
                for path,meaning in model['scalar_semantics'].items() if meaning == 'unreviewed')
        print(json.dumps({'collection_blockers':blockers,'confidence_readiness_blockers':field_blockers,
                          'probability_semantics':semantics},indent=2))
        return 1 if blockers else 0
    else:
        if not all((args.questions,args.input_manifest,args.protocol,args.output)):
            parser.error('--questions, --input-manifest, --protocol and --output required')
        coverage = collect(args.config,args.questions,args.input_manifest,args.protocol,args.output,
            mock=args.mock,development=args.development,ids=args.configs,resume=args.resume,
            reviewer=args.reviewer,reviewed_stops=args.reviewed_stops)
    print(json.dumps({k:v for k,v in coverage.items() if k != 'selected_final_records'},indent=2))
    return 0 if coverage['complete'] and coverage.get('development_passed', True) else 1


if __name__ == '__main__':
    raise SystemExit(main())
