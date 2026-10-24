import argparse
import base64
from contextlib import ExitStack, contextmanager
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import fcntl
import json
import os
from pathlib import Path
import random
import re
import signal
import socket
import sys
import threading
import time
import urllib.error

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from benchmarking.common import (ROOT, CONFIG_PATH, WORKFLOW, budget_roster_decision, SCHEMA_VERSION, is_decisions, package_version, decisions_tolerance, evaluation_blockers,
    scheduler_policy, artifact_ref, code_record, config_binding, config_errors, run_environment_record,
    expected_identities, file_hash, fingerprint, load_config, load_questions, measurement,
    question_hash, read_package, runtime_blockers, strict_json, utc_now,
    validate_finish, validate_manifest, manifest_configurations, manifest_runs, validate_manifest_config,
    write_new, run_inventory, save_run_inventory, empty_probability, native_probability,
    continuation_parent, read_jsonl, CAPACITY_AMENDMENT, DEEPSEEK_TEN_AMENDMENT, validate_continuation_settings, verify_run_inventory)
from benchmarking import adapters
from benchmarking.study import locked_study, StudyRegistry


def build_prompt(q, config):
    values={'{stem}':q.stem,'{options}':'\n'.join(f'{o.label}. {o.text}' for o in q.options)}
    return re.sub(r'\{stem\}|\{options\}',lambda match:values[match.group()],config['prompt']['text'])

def plan_counts(config, experiment='E1+E2', count=None):
    errors=config_errors(config)
    if errors: raise ValueError('; '.join(errors))
    if experiment not in ('E1','E2','E1+E2'): raise ValueError('unsupported experiment')
    count=count if count is not None else 1273
    if type(count) is not int or count<1: raise ValueError('positive item count required')
    e1=count*len(config['model_roster']) if experiment in ('E1','E1+E2') else 0
    e2=count*2 if experiment in ('E2','E1+E2') else 0
    preview={'schema_version':package_version(config),'preview_only':True,'experiment':experiment,'items':count,'primary':e1,('secondary_decisions' if is_decisions(config) else 'secondary_jev'):e2,'total':e1+e2,
        'runtime_blockers':runtime_blockers(config)}
    if is_decisions(config): preview['evaluation_blockers']=evaluation_blockers(config)
    return preview

def make_manifest(config, questions_path, *, study_id, experiment, provenance, seed, keys_sha256, preparation_sha256, inventory, content_sha256=None):
    if config_errors(config): raise ValueError('; '.join(config_errors(config)))
    if type(seed) is not int or seed<0: raise ValueError('explicit nonnegative seed required')
    if not study_id or any(x not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for x in study_id): raise ValueError('simple stable study_id required')
    questions=load_questions(questions_path)
    benchmark=questions[0].benchmark;split=questions[0].split
    if any(q.benchmark!=benchmark or q.split!=split for q in questions): raise ValueError('one benchmark/split per package')
    if split!='evaluation': raise ValueError('collection requires evaluation split')
    ordered=list(questions);random.Random(seed).shuffle(ordered)
    if experiment not in ('E1','E2','E1+E2'): raise ValueError('unsupported experiment')
    configs=manifest_configurations(config)
    runs=manifest_runs(study_id,configs,set(experiment.split('+')))
    bindings={'code':fingerprint(code_record(config=config)),'environment':fingerprint(run_environment_record(config)),'config':config_binding(config),
        'prompt':fingerprint(config['prompt']),'questions':file_hash(questions_path),'keys':keys_sha256,'protocol_record':inventory['readme_sha256'],'execution_record':fingerprint(inventory)}
    manifest={'schema_version':package_version(config),'execution_workflow':config.get('execution_workflow',WORKFLOW),'study_id':study_id,'provenance':provenance,'benchmark':benchmark,'split':split,
        'items':[{'item_id':q.item_id,'content_sha256':question_hash(q),'labels':list(q.labels)} for q in questions],
        'configurations':configs,'runs':runs,'bindings':bindings,
        'key_manifest':{'sha256':keys_sha256,'preparation_sha256':preparation_sha256},'schedule':{'kind':'round_robin','seed':seed,'item_order':[q.item_id for q in ordered]}}
    if is_decisions(config):
        manifest['probability_tolerance']=decisions_tolerance(config['models']['D01'].get('sum_tolerance'))
        bindings['study_protocol']=inventory['study_protocol_sha256']
    policy=scheduler_policy(config)
    if policy is not None: manifest['schedule']['policy']=policy
    decision=budget_roster_decision(config)
    if decision is not None: manifest['roster_decision']=decision
    if content_sha256 is not None:
        if not isinstance(content_sha256,str) or not re.fullmatch('[0-9a-f]{64}',content_sha256):
            raise ValueError('invalid preparation content digest')
        manifest['key_manifest']['content_sha256']=content_sha256
    return validate_manifest(manifest)

def append_event(path, event, previous):
    event={**event,'previous_sha256':previous};digest=fingerprint(event);event['event_sha256']=digest
    with Path(path).open('a',encoding='utf-8') as f:
        f.write(json.dumps(event,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    return digest

def reviewed_continuation(manifest, parent, output, *, reviewer, reviewed_stops, amendment=None):
    """Bind a new segment to immutable reviewed parent evidence; never rebind old calls."""
    parent=Path(parent).resolve();output=Path(output).resolve()
    prior,_,coverage=read_package(parent)
    validate_continuation_settings(prior,manifest,amendment,parent)
    if (output/'manifest.json').exists():
        saved,_,_=read_package(output)
        link=saved.get('continuation',{})
        if (output/link.get('parent','')).resolve()!=parent or link.get('reviewer')!=reviewer or link.get('reviewed_stop_events')!=reviewed_stops or link.get('execution_amendment')!=amendment:
            raise ValueError('resume continuation review changed')
        if {k:v for k,v in saved.items() if k!='continuation'}!=manifest: raise ValueError('resume manifest/bindings changed')
        return saved
    manifest={**manifest,'continuation':{
        'parent':os.path.relpath(parent,output),'manifest_sha256':fingerprint(prior),
        'provenance_sha256':file_hash(parent/'provenance.json'),'events_sha256':file_hash(parent/'events.jsonl'),
        'last_event_sha256':coverage['last_event_sha256'],'reviewed_stop_events':reviewed_stops,
        'reviewer':reviewer,'reviewed_at':utc_now(),
        'root_bindings':prior.get('continuation',{}).get('root_bindings',prior['bindings'])}}
    if amendment or 'root_schedule' in prior.get('continuation',{}):
        manifest['continuation']['root_schedule']=prior.get('continuation',{}).get('root_schedule',prior['schedule'])
    if amendment: manifest['continuation']['execution_amendment']=amendment
    continuation_parent(output,manifest,seen={output})
    return manifest

def resolve_incomplete_read(package, *, start_event, reviewer, partial_bytes):
    """Append one human-reviewed failure; retain every original byte and never dispatch."""
    package=Path(package).resolve()
    if not isinstance(reviewer,str) or not reviewer.strip() or type(partial_bytes) is not int or partial_bytes<0:
        raise ValueError('reviewer and nonnegative partial-byte count required')
    manifest=strict_json((package/'manifest.json').read_text())
    config=verify_run_inventory(package,manifest)['config']
    registry_root=Path(config['execution']['study_registry'])
    if not registry_root.is_absolute(): registry_root=ROOT/registry_root
    if not (registry_root/fingerprint(manifest['study_id'])/'study.json').is_file():
        raise ValueError('registered interrupted study required')
    with locked_study(config,manifest) as registry, (package/'.lock').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if any(e.get('parent_manifest_sha256')==fingerprint(manifest) for e in registry.entries):
            raise ValueError('parent already has a continuation; preserve its bound journal')
        if not any(e['package']==str(package) and e['manifest_sha256']==fingerprint(manifest) for e in registry.entries):
            raise ValueError('interrupted package is not registered')
        _,_,coverage=read_package(package)
        events=read_jsonl(package/'events.jsonl')
        start=next((e for e in events if e['event']=='start' and e['event_sha256']==start_event),None)
        fields=('benchmark','split','item_id','config_id','run_id')
        if start is None or coverage['indeterminate']!=[[start['identity'][k] for k in fields]]:
            raise ValueError('exact single unresolved start-event review required')
        request_path=package/start['request']['path']
        if set(request_path.parent.iterdir())!={request_path}:
            raise ValueError('interrupted attempt has additional artifacts; review them before recovery')
        model=config['models'][start['identity']['config_id']];reviewed_at=utc_now()
        usage=adapters.usage_record({},model)
        record={'identity':start['identity'],'status':'transport_error','selected_label':None,'failure_reason':'IncompleteRead',
            'probability':empty_probability(),'usage':usage,'billing':adapters.billing_record({},usage,model),
            'timing':{'started_at':start['started_at'],'finished_at':reviewed_at,
                'duration_seconds':measurement(reason='interrupted_response_timing_unavailable')},
            'route':{'requested_model':strict_json(request_path.read_text())['model'],'returned_model':None,
                'requested_route':_provider(model),'returned_route':None,'effective_controls':None,
                'provider_request_id':None,'finish_reason':None},
            'artifacts':{'request':start['request'],'raw_response':None,'final':None,'reasoning':None},
            'recovery_review':{'reviewer':reviewer,'reviewed_at':reviewed_at,'exception':'http.client.IncompleteRead',
                'partial_bytes':partial_bytes,'source_sha256':fingerprint(code_record()),
                'timing_basis':'review_timestamp_not_response_completion'}}
        if manifest['schedule'].get('policy'):
            record['scheduler']={'provider':_provider(model),'response_status':None,'stop_reason':None}
        if is_decisions(manifest): record['provider_confidence']=measurement()
        validate_finish(record,manifest)
        event={k:start[k] for k in ('schema_version','bindings','manifest_sha256','identity')}
        digest=append_event(package/'events.jsonl',{**event,'event':'finish','record':record},coverage['last_event_sha256'])
        _,_,after=read_package(package)
        return {'reviewed_event':digest,'terminal':after['terminal'],'unstarted':len(after['unstarted']),
            'indeterminate':after['indeterminate']}

def mock_dispatch(request, model, q):
    label=q.labels[int(fingerprint([q.item_id,model['model_id']])[:8],16)%4]
    if model['adapter'] in {'jev','jev_gateway','openai_decisions_gateway'}:
        body={'model':model['version'] or model['model_id'],'answers':{'answer':{'type':'choice','choice':label,
            'probabilities':{x:('0.70' if x==label else '0.10') for x in q.labels},'confidence':'0.42'}},
            'usage':{'input_tokens':100,'output_tokens':1,'cost':'0.00001'}}
        if model['adapter'] in {'jev_gateway','openai_decisions_gateway'}:
            body.update(id='fictional-decision-request',provider=model['route'] or 'fictional-provider')
    else:
        body={'id':'fictional-request','model':model['version'] or model['model_id'],'provider':model['route'] or 'fictional-provider',
            'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'answer':label}),'reasoning':'fictional reasoning, never parsed for answer'}}],
            'usage':{'prompt_tokens':100,'completion_tokens':20,'completion_tokens_details':{'reasoning_tokens':10},'cost':'0.0001'}}
    return {'status':200,'body_base64':base64.b64encode(json.dumps(body).encode()).decode(),'headers':{}}

def _provider(model):
    """Returned provider identity groups routes, pair arms and Mistral variants."""
    return model['route'] or ('typesafe-direct' if model['adapter']=='jev' else 'fictional-provider')


def _retry_after_seconds(raw, *, elapsed_seconds=0.0):
    """Provider hint only paces later unstarted requests; never retries a finish."""
    headers=raw.get('headers',{}) if isinstance(raw,dict) else {}
    value=next((v for k,v in headers.items() if k.lower()=='retry-after'),None)
    if value is None: return 0.0
    try:
        seconds=float(value)
        if not __import__('math').isfinite(seconds): return 0.0
        return max(0.0,seconds-max(0.0,elapsed_seconds))
    except (ValueError,TypeError):
        try: return max(0.0,(parsedate_to_datetime(str(value))-datetime.now(timezone.utc)).total_seconds())
        except (ValueError,TypeError,OverflowError): return 0.0


def _dispatch_worker(request, model, q, config, mock, transport, queued_at):
    started=utc_now();start_time=time.monotonic();raw=None;failure=None;error=None
    try:
        raw=(transport or mock_dispatch)(request,model,q) if mock else (transport(request,model,q) if transport else adapters.dispatch(request,model,config['execution']['timeout_seconds']))
    except (TimeoutError,socket.timeout) as exc: failure=('timeout',type(exc).__name__)
    except (urllib.error.URLError,OSError) as exc: failure=('transport_error',type(exc).__name__)
    except BaseException as exc: error=exc
    return {'raw':raw,'failure':failure,'error':error,'started_at':started,'finished_at':utc_now(),
            'started_monotonic':start_time,'duration_seconds':time.monotonic()-start_time,'queue_wait_seconds':max(0.0,start_time-queued_at)}


@contextmanager
def _graceful_signals():
    requested=[];previous={}
    def request_stop(signum, frame):
        if not requested: requested.append(signal.Signals(signum).name)
    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT,signal.SIGTERM):
                previous[signum]=signal.signal(signum,request_stop)
        yield requested
    finally:
        for signum,handler in previous.items(): signal.signal(signum,handler)


def _save_response(config, q, model, ident, request, request_ref, adir, output, result):
    """Persist one response using the same parsing rules for originals and retries."""
    if result['error'] is not None: raise result['error']
    raw=result['raw'];failure=result['failure'];rawref=None
    if raw is not None:
        rawpath=adir/'raw_response.json';write_new(rawpath,raw);rawref=artifact_ref(rawpath,output)
        parsed=adapters.parse(raw,model,q.labels)
        if parsed['effective_controls'] is not None and parsed['effective_controls']!=model.get('request_settings',{}):
            parsed.update(status='drift',selected_label=None,failure_reason='effective_controls_mismatch')
            if parsed['probability']['native_vector'] is not None:
                p=parsed['probability'];reason=p['concentration_reason']
                parsed['probability']=native_probability(p['native_vector'],None,q.labels,p['concentration'],p['semantics'],
                    sum_tolerance=model.get('sum_tolerance'),decisions=True) if is_decisions(config) else native_probability(p['native_vector'],None,q.labels,p['concentration'],p['semantics'])
                parsed['probability']['concentration_reason']=reason
    else:
        parsed={'status':failure[0],'failure_reason':failure[1],'selected_label':None,'probability':empty_probability(),
            'usage':{},'observed_usd':None,'final':None,'reasoning':None,'returned_model':None,'returned_route':None,'provider_request_id':None,'finish_reason':None,'effective_controls':None}
    refs={'request':request_ref,'raw_response':rawref,'final':None,'reasoning':None}
    for channel in ('final','reasoning'):
        value=parsed.get(channel)
        if value is not None:
            path=adir/(channel+'.txt')
            with path.open('x') as f:
                f.write(value if isinstance(value,str) else adapters.json_text(value));f.flush();os.fsync(f.fileno())
            refs[channel]=artifact_ref(path,output)
    usage=adapters.usage_record(parsed['usage'],model);billing=adapters.billing_record(parsed,usage,model)
    record={'identity':ident,'status':parsed['status'],'selected_label':parsed['selected_label'],'failure_reason':parsed['failure_reason'],
        'probability':parsed['probability'],'usage':usage,'billing':billing,
        'timing':{'started_at':result['started_at'],'finished_at':result['finished_at'],'duration_seconds':measurement(result['duration_seconds'])},
        'route':{'requested_model':request['model'],'returned_model':parsed['returned_model'],'requested_route':_provider(model),
            'returned_route':parsed['returned_route'],'effective_controls':parsed['effective_controls'],'provider_request_id':parsed['provider_request_id'],'finish_reason':parsed['finish_reason']},
        'artifacts':refs}
    if is_decisions(config): record['provider_confidence']=parsed.get('provider_confidence',measurement())
    return record


def run_package(manifest, questions_path, config, output, *, mock, transport=None, inventory=None, input_manifest_path=None, config_path=CONFIG_PATH,
                progress_interval=30.0, progress_stream=None, check_only=False):
    """Resume only exact bindings; terminal failures never retry, open starts never resend."""
    validate_manifest(manifest);output=Path(output)
    questions=load_questions(questions_path);qmap={q.item_id:q for q in questions}
    if file_hash(questions_path)!=manifest['bindings']['questions'] or config_binding(config)!=manifest['bindings']['config'] or fingerprint(code_record(config=config))!=manifest['bindings']['code'] or fingerprint(run_environment_record(config))!=manifest['bindings']['environment']: raise ValueError('input/config/code/environment binding mismatch')
    if set(qmap)!={x['item_id'] for x in manifest['items']} or any(question_hash(qmap[x['item_id']])!=x['content_sha256'] for x in manifest['items']): raise ValueError('question content/IDs mismatch')
    validate_manifest_config(manifest,config,questions)
    if mock and manifest['provenance']!='mock' or not mock and manifest['provenance']=='mock': raise ValueError('execution provenance mismatch')
    if input_manifest_path is None: raise ValueError('public input manifest required')
    public=validate_input_manifest(input_manifest_path,questions_path)
    if public['keys_sha256']!=manifest['bindings']['keys'] or public['preparation_manifest_sha256']!=manifest['key_manifest']['preparation_sha256']:
        raise ValueError('public input/key digest mismatch')
    if not mock and public['provenance']!='prepared': raise ValueError('live inference requires prepared research inputs')
    errors=runtime_blockers(config)
    if errors: raise ValueError('; '.join(errors))
    if not mock and is_decisions(config):
        pending=evaluation_blockers(config)
        if pending: raise ValueError('; '.join(pending))
        if inventory is not None and inventory['git']['dirty'] is not False:
            raise ValueError('commit clean D01 execution source/configuration before evaluation')
    if inventory is None or fingerprint(inventory)!=manifest['bindings']['execution_record']:
        raise ValueError('matching automatic run inventory required')
    if not mock and not check_only:
        for cid in {r['config_id'] for r in manifest['runs']}:
            model=config['models'][cid]
            if not os.environ.get('TYPESAFE_API_KEY' if model['adapter']=='jev' else 'OPENROUTER_API_KEY'):
                raise ValueError(cid+': credential unavailable')
    # Validate every distinct request before claiming a study or starting a call.
    for q in questions:
        for cid in {r['config_id'] for r in manifest['runs']}:
            adapters.build_request(q,config['models'][cid],build_prompt(q,config),mock=mock)
    inherited=[]
    if check_only and 'continuation' not in manifest: raise ValueError('continuation required for no-call check')
    if 'continuation' in manifest:
        _,inherited,parent_coverage=continuation_parent(output,manifest,seen={output.resolve()})
        if not mock and inventory['git']['dirty'] is not False: raise ValueError('commit clean recovery source before continuation')
    if check_only:
        registry_root=Path(config['execution']['study_registry'])
        if not registry_root.is_absolute(): registry_root=ROOT/registry_root
        directory=registry_root/fingerprint(manifest['study_id'])
        if not (directory/'study.json').exists(): raise ValueError('registered parent study required for continuation check')
        registry=StudyRegistry(directory,manifest);registry.load();registry.register(output,check_only=True)
        coverage=read_package(output)[2] if (output/'manifest.json').exists() else parent_coverage
        primary_ids={r['run_id'] for r in manifest['runs'] if r['role']=='primary'}
        return {'check_only':True,'inherited_terminal':len(inherited),'terminal':coverage['terminal'],
            'remaining_E1':sum(i[-1] in primary_ids for i in coverage['unstarted']),
            'remaining_E2':sum(i[-1] not in primary_ids for i in coverage['unstarted']),
            'reviewed_stop_events':manifest['continuation']['reviewed_stop_events'],
            'execution_amendment':manifest['continuation'].get('execution_amendment'),
            'schedule_policy':manifest['schedule'].get('policy')}
    if {r['experiment_id'] for r in manifest['runs']}=={'E2'}:
        registry_root=Path(config['execution']['study_registry'])
        if not registry_root.is_absolute(): registry_root=ROOT/registry_root
        if not (registry_root/fingerprint(manifest['study_id'])/'packages.jsonl').is_file():
            raise ValueError('E2 requires complete registered E1 terminal coverage in the same study')
    with locked_study(config,manifest) as registry, ExitStack() as resources:
        if manifest['schedule'].get('policy') and {r['experiment_id'] for r in manifest['runs']}=={'E2'}:
            primary_plan={**manifest,'runs':manifest_runs(manifest['study_id'],manifest['configurations'],{'E1'})}
            primary_done={tuple(r['identity'][k] for k in ('benchmark','split','item_id','config_id','run_id')) for r in registry.records()}
            if not set(expected_identities(primary_plan)).issubset(primary_done):
                raise ValueError('E2 requires complete registered E1 terminal coverage in the same study')
        fresh=registry.register(output)
        output.mkdir(parents=True,exist_ok=True)
        lock=resources.enter_context((output/'.lock').open('a'))
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/'manifest.json').exists():
            stored=strict_json((output/'manifest.json').read_text())
            if stored!=manifest: raise ValueError('resume manifest/bindings changed')
        else:
            if (output/'events.jsonl').exists(): raise ValueError('journal without manifest')
            if not fresh: raise ValueError('registered package missing; human review required')
            if input_manifest_path is None: raise ValueError('public input manifest required')
            save_run_inventory(output,inventory,questions_path,input_manifest_path,config_path=config_path)
            write_new(output/'manifest.json',manifest)
            plan={'schema_version':manifest['schema_version'],'event':'plan','bindings':manifest['bindings'],
                  'manifest_sha256':fingerprint(manifest),'created_at':utc_now(),'previous_sha256':None}
            plan['event_sha256']=fingerprint(plan)
            write_new(output/'events.jsonl',[plan],jsonl=True)
        _,local_records,coverage=read_package(output,include_parents=False)
        records=inherited+local_records
        if coverage['indeterminate']: raise ValueError('indeterminate started attempts require human review; automatic resend prohibited')
        if any(x['status']=='drift' for x in records): raise ValueError('route/version/control/schema drift remains unresolved; preserve package and review')
        if any(x.get('scheduler',{}).get('stop_reason') for x in local_records): raise ValueError('provider rate-limit stop remains unresolved; preserve package and obtain human access review')
        requested_stop=resources.enter_context(_graceful_signals())
        allowed_ids=set(expected_identities(manifest));label_map={q.item_id:list(q.labels) for q in questions}
        previous=coverage['last_event_sha256'];done={tuple(x['identity'][k] for k in ('benchmark','split','item_id','config_id','run_id')) for x in records}
        policy=manifest['schedule'].get('policy')
        global_limit=policy['concurrency'] if policy else 1
        provider_limit=policy['provider_concurrency'] if policy else 1
        provider_overrides=policy.get('provider_concurrency_overrides',{}) if policy else {}
        def limit_for(provider):
            return provider_overrides.get(provider,provider_limit)
        intervals=policy['provider_min_interval_seconds'] if policy else {}
        rate_stop=policy['rate_limit_stop_after'] if policy else None
        counts={};next_start={};consecutive_429={};paused_providers=set();stop=False;unexpected=None
        provider_pause=bool(policy and policy['version']==3)
        progress_stream=progress_stream if progress_stream is not None else (None if mock else sys.stderr)
        next_report=time.monotonic();valid=sum(r['status']=='valid' for r in records)
        def report(active=0, *, state='Running', force=False):
            nonlocal next_report
            now=time.monotonic()
            if progress_stream is not None and (force or now>=next_report):
                print(f'[{utc_now()}] {state}: {len(done)}/{coverage["expected"]} finished '
                      f'({valid} valid, {len(done)-valid} failed), {active} active'
                      + (f', paused providers: {", ".join(sorted(paused_providers))}' if paused_providers else ''),
                      file=progress_stream,flush=True)
                next_report=now+progress_interval
        report(force=True)
        if policy:
            raw_packages=[output];ancestor_manifest=manifest
            while 'continuation' in ancestor_manifest:
                ancestor=raw_packages[-1]/ancestor_manifest['continuation']['parent']
                raw_packages.append(ancestor)
                ancestor_manifest=strict_json((ancestor/'manifest.json').read_text())
            for package in reversed(raw_packages):
                segment=strict_json((package/'manifest.json').read_text())
                if 'continuation' in segment:
                    link=segment['continuation']
                    parent_events=read_jsonl(package/link['parent']/'events.jsonl')
                    for event in parent_events:
                        if event['event_sha256'] in link['reviewed_stop_events']:
                            if event['record'].get('scheduler',{}).get('stop_reason')=='provider_rate_limit_stop':
                                consecutive_429[event['record']['scheduler']['provider']]=0
                segment_records=read_jsonl(package/'events.jsonl')
                for event in segment_records:
                    if event['event']!='finish': continue
                    record=event['record']
                    provider=record.get('scheduler',{}).get('provider',record['route']['requested_route'])
                    status=record.get('scheduler',{}).get('response_status')
                    consecutive_429[provider]=(consecutive_429.get(provider,0)+1) if status==429 else 0
                    try:
                        request_age=(datetime.now(timezone.utc)-datetime.fromisoformat(record['timing']['started_at'])).total_seconds()
                        finish_age=(datetime.now(timezone.utc)-datetime.fromisoformat(record['timing']['finished_at'])).total_seconds()
                        next_start[provider]=max(next_start.get(provider,0),time.monotonic()+max(0.0,float(intervals.get(provider,0))-request_age))
                    except (ValueError,TypeError):
                        raise ValueError('saved request time cannot restore provider pacing') from None
                    if status==429 and record['artifacts']['raw_response']:
                        raw=strict_json((package/record['artifacts']['raw_response']['path']).read_text())
                        delay=_retry_after_seconds(raw,elapsed_seconds=finish_age)
                        next_start[provider]=max(next_start.get(provider,0),time.monotonic()+delay)
        primary=[r for r in manifest['runs'] if r['role']=='primary'];secondary=[r for r in manifest['runs'] if r['role']=='secondary']
        items_by_id={x['item_id']:x for x in manifest['items']}
        scheduled_items=[items_by_id[x] for x in manifest['schedule']['item_order']]
        primary_tasks=[]
        for n,item in enumerate(scheduled_items):
            rotation=n%len(primary) if primary else 0
            primary_tasks += [(item,r) for r in primary[rotation:]+primary[:rotation]]
        secondary_tasks=[(item,r) for r in secondary for item in scheduled_items]
        event_base={'schema_version':manifest['schema_version'],'bindings':manifest['bindings'],'manifest_sha256':fingerprint(manifest)}

        def persist_finish(context,result):
            nonlocal previous,stop,valid
            item,run,ident,key,q,model,request,request_ref,adir,provider=context
            record=_save_response(config,q,model,ident,request,request_ref,adir,output,result)
            raw=result['raw']
            if policy:
                response_status=raw.get('status') if raw else None
                consecutive_429[provider]=(consecutive_429.get(provider,0)+1) if response_status==429 else 0
                stop_reason='provider_rate_limit_stop' if consecutive_429[provider]>=rate_stop else None
                record['scheduler']={'provider':provider,'response_status':response_status,'stop_reason':stop_reason}
                record['timing']['queue_wait_seconds']=measurement(result['queue_wait_seconds'])
                if provider_pause:
                    # The single-provider slot becomes eligible from the actual
                    # worker start, even if thread scheduling delayed dispatch.
                    next_start[provider]=max(next_start.get(provider,0),
                        result['started_monotonic']+float(intervals.get(provider,0)))
                if response_status==429:
                    next_start[provider]=max(next_start.get(provider,0),time.monotonic()+_retry_after_seconds(raw))
                if stop_reason:
                    if provider_pause:
                        paused_providers.add(provider)
                    else: stop=True
            validate_finish(record,manifest,allowed_ids=allowed_ids,item_labels=label_map)
            previous=append_event(output/'events.jsonl',{**event_base,'identity':ident,'event':'finish','record':record},previous)
            done.add(key)
            valid+=record['status']=='valid'
            if policy and stop_reason and provider_pause:
                report(sum(counts.values()),state=f'Provider paused ({provider}); other providers continue',force=True)
            if record['status']=='drift': stop=True

        submission_sequences={};submission_sequence=len(done)
        with ThreadPoolExecutor(max_workers=global_limit) as workers:
            for phase in (primary_tasks,secondary_tasks):
                if stop: break
                queued_at=time.monotonic();pending=[]
                for item,run in phase:
                    ident={'benchmark':manifest['benchmark'],'split':manifest['split'],'item_id':item['item_id'],'config_id':run['config_id'],'run_id':run['run_id']}
                    key=tuple(ident[k] for k in ('benchmark','split','item_id','config_id','run_id'))
                    if key not in done: pending.append((item,run,ident,key))
                inflight={}
                while (pending and not stop) or inflight:
                    if requested_stop and not stop:
                        stop=True
                        report(len(inflight),state=f'Stopping ({requested_stop[0]}); waiting for active requests',force=True)
                    report(len(inflight),state='Stopping; waiting for active requests' if stop else 'Running')
                    while pending and not stop and not requested_stop and len(inflight)<global_limit:
                        now=time.monotonic()
                        chosen=next((i for i,(_,run,_,_) in enumerate(pending)
                            if _provider(config['models'][run['config_id']]) not in paused_providers
                            and counts.get(_provider(config['models'][run['config_id']]),0)<limit_for(_provider(config['models'][run['config_id']]))
                            and next_start.get(_provider(config['models'][run['config_id']]),0)<=now),None)
                        if chosen is None: break
                        item,run,ident,key=pending.pop(chosen);q=qmap[item['item_id']];model=config['models'][run['config_id']];provider=_provider(model)
                        try:
                            request=adapters.build_request(q,model,build_prompt(q,config),mock=mock)
                            token=fingerprint(ident);adir=output/'artifacts'/token;adir.mkdir(parents=True,exist_ok=True)
                            request_path=adir/'request.json';write_new(request_path,request);request_ref=artifact_ref(request_path,output)
                            previous=append_event(output/'events.jsonl',{**event_base,'identity':ident,'event':'start','request':request_ref,
                                'started_at':utc_now()},previous)
                            future=workers.submit(_dispatch_worker,request,model,q,config,mock,transport,queued_at)
                            inflight[future]=(item,run,ident,key,q,model,request,request_ref,adir,provider)
                            submission_sequences[future]=submission_sequence;submission_sequence+=1
                            counts[provider]=counts.get(provider,0)+1
                            next_start[provider]=time.monotonic()+float(intervals.get(provider,0))
                        except BaseException as exc:
                            unexpected=unexpected or exc;stop=True;break
                    if not inflight:
                        if stop or not pending: break
                        eligible_times=[next_start.get(_provider(config['models'][run['config_id']]),0) for _,run,_,_ in pending
                            if _provider(config['models'][run['config_id']]) not in paused_providers]
                        if not eligible_times: break
                        time.sleep(min(1.0,max(.001,min(eligible_times)-time.monotonic())))
                        continue
                    delays=[next_start.get(_provider(config['models'][run['config_id']]),0)-time.monotonic() for _,run,_,_ in pending
                            if _provider(config['models'][run['config_id']]) not in paused_providers
                            and counts.get(_provider(config['models'][run['config_id']]),0)<limit_for(_provider(config['models'][run['config_id']]))]
                    future_timeout=max(.001,min(delays)) if delays and len(inflight)<global_limit and not stop else None
                    finished,_=wait(inflight,timeout=min(1.0,future_timeout) if future_timeout is not None else 1.0,return_when=FIRST_COMPLETED)
                    completed=[]
                    for future in finished:
                        context=inflight.pop(future);provider=context[-1];counts[provider]-=1
                        sequence=submission_sequences.pop(future)
                        try:
                            result=future.result()
                            completed.append((result['finished_at'],sequence,context,result))
                        except BaseException as exc:
                            unexpected=unexpected or exc;stop=True
                    # A wait result is a set. Its iteration order must not decide
                    # provider streaks or whether further identities can start.
                    for _,_,context,result in sorted(completed,key=lambda row:(row[0],row[1])):
                        try: persist_finish(context,result)
                        except BaseException as exc:
                            unexpected=unexpected or exc;stop=True
                if unexpected is not None: raise unexpected
                # Missing E1 work on paused providers must never release E2.
                if pending: break
        result=read_package(output)[2]
        if provider_pause: result['paused_providers']=sorted(paused_providers)
        report(state='Complete' if result['complete'] else ('Paused; active requests saved' if requested_stop else ('Provider review required; active requests saved' if paused_providers else 'Stopped')),force=True)
        return result


def validate_input_manifest(path, questions_path):
    record=strict_json(Path(path).read_text())
    fields={'schema_version','type','provenance','benchmark','split','count','questions_sha256','keys_sha256','preparation_manifest_sha256','ordered_ids','source_rows'}
    if set(record) not in (fields,fields|{'content_manifest_sha256'}) or record['schema_version']!=2 or record['type']!='inference_input': raise ValueError('public answer-free preparation manifest required')
    if 'content_manifest_sha256' in record and (not isinstance(record['content_manifest_sha256'],str) or not re.fullmatch('[0-9a-f]{64}',record['content_manifest_sha256'])):
        raise ValueError('invalid preparation content digest')
    for name in ('questions_sha256','keys_sha256','preparation_manifest_sha256'):
        if not isinstance(record[name],str) or not re.fullmatch('[0-9a-f]{64}',record[name]): raise ValueError('invalid public digest: '+name)
    if record['provenance'] not in ('mock','prepared'): raise ValueError('unsupported input provenance')
    rows=record['source_rows']
    if not isinstance(rows,list) or any(type(row) is not int or row<1 for row in rows) or len(set(rows))!=len(rows):
        raise ValueError('source rows must be unique positive integers')
    questions=load_questions(questions_path)
    if record['questions_sha256']!=file_hash(questions_path) or record['ordered_ids']!=[q.item_id for q in questions] or record['count']!=len(questions) or len(record['source_rows'])!=len(questions): raise ValueError('public manifest coverage/hash mismatch')
    if any(q.benchmark!=record['benchmark'] or q.split!=record['split'] for q in questions): raise ValueError('public manifest split mismatch')
    return record

def smoke_test(config, questions_path, input_manifest, output, *, mock, transport=None, config_path=CONFIG_PATH, protocol_path=None):
    """One development item per configuration; save receipts and stop at the first failure."""
    prep=validate_input_manifest(input_manifest,questions_path)
    questions=load_questions(questions_path)
    if len(questions)!=1 or questions[0].benchmark!='medqa' or questions[0].split!='development':
        raise ValueError('smoke requires exactly one prepared MedQA development question')
    errors=config_errors(config)
    if errors: raise ValueError('; '.join(errors))
    output=Path(output)
    if output.exists(): raise ValueError('smoke output already exists; no overwrite, resume or resend')
    if not mock and prep['provenance']!='prepared': raise ValueError('live smoke requires prepared research inputs')
    q=questions[0];roster=config['model_roster']
    requests={cid:adapters.build_request(q,config['models'][cid],build_prompt(q,config),mock=mock) for cid in roster}
    errors=runtime_blockers(config)
    if errors: raise ValueError('; '.join(errors))
    inventory=run_inventory(config,questions_path,input_manifest,config_path=config_path,protocol_path=protocol_path)
    git=inventory['git']
    if not mock and is_decisions(config) and git['dirty'] is not False: raise ValueError('commit clean D01 source before live smoke')
    if not mock:
        required={'TYPESAFE_API_KEY' if config['models'][cid]['adapter']=='jev' else 'OPENROUTER_API_KEY' for cid in roster}
        if any(not os.environ.get(name) for name in required): raise ValueError('required smoke credential unavailable')
    inventory={**inventory,'purpose':'deployment_smoke','created_at':utc_now(),'mode':'mock' if mock else 'live',
               'item_id':q.item_id,'configurations':roster,'expected_calls':len(roster),'no_scoring':True}
    output.mkdir(parents=True,exist_ok=False)
    save_run_inventory(output,inventory,questions_path,input_manifest,config_path=config_path)
    outcomes=[{'configuration':cid,'status':'unstarted','http_status':None} for cid in roster]
    interrupted=None;last_start={}
    def save_report():
        valid=sum(row['status']=='valid' for row in outcomes)
        result={'type':'deployment_smoke','mode':inventory['mode'],'created_at':inventory['created_at'],
            'passed':valid==len(roster),'expected':len(roster),'valid':valid,'outcomes':outcomes}
        write_new(output/'result.json',result)
        lines=[f"# Smoke test: {'PASS' if result['passed'] else 'STOPPED'} ({inventory['mode']})",'',
            f"Date: {inventory['created_at']}. Commit: `{git['commit']}`. Valid: {valid}/{len(roster)}.",
            '','One development question per configuration. No scoring or retries.','',
            '| Configuration | Result | HTTP |','| --- | --- | --- |']
        lines.extend(f"| {row['configuration']} | {row['status']} | {row['http_status'] or ''} |" for row in outcomes)
        with (output/'Smoke_Test.md').open('x',encoding='utf-8') as f:
            f.write('\n'.join(lines)+'\n');f.flush();os.fsync(f.fileno())
        return result
    with _graceful_signals() as requested_stop:
        for row in outcomes:
            if requested_stop: break
            cid=row['configuration'];model=config['models'][cid];request=requests[cid]
            try:
                if not mock:
                    route=model['route'];interval=config['execution'].get('provider_min_interval_seconds',{}).get(route,0)
                    delay=last_start.get(route,0)+interval-time.monotonic()
                    if delay>0: time.sleep(delay)
                if requested_stop: break
                write_new(output/cid/'request.json',request)
                write_new(output/cid/'start.json',{'started_at':utc_now(),'request_sha256':file_hash(output/cid/'request.json')})
                row['status']='indeterminate';last_start[model['route']]=time.monotonic()
                receipt=_dispatch_worker(request,model,q,config,mock,transport,time.monotonic())
                if receipt['raw'] is not None:
                    write_new(output/cid/'raw_response.json',receipt['raw'])
                    parsed=adapters.parse(receipt['raw'],model,q.labels)
                    if parsed['effective_controls'] is not None and parsed['effective_controls']!=model.get('request_settings',{}):
                        parsed.update(status='drift',failure_reason='effective_controls_mismatch')
                    row.update(status=parsed['status'],http_status=receipt['raw']['status'],failure_reason=parsed['failure_reason'])
                    if is_decisions(config):
                        row.update(probability=parsed['probability'],provider_confidence=parsed['provider_confidence'],
                            returned_model=parsed['returned_model'],returned_route=parsed['returned_route'],
                            usage=adapters.usage_record(parsed['usage'],model),
                            billing=adapters.billing_record(parsed,adapters.usage_record(parsed['usage'],model),model))
                elif receipt['failure']:
                    row.update(status=receipt['failure'][0],failure_reason=receipt['failure'][1])
                elif receipt['error']:
                    row['failure_reason']=type(receipt['error']).__name__
                    interrupted=receipt['error']
                row.update(started_at=receipt['started_at'],finished_at=receipt['finished_at'],duration_seconds=receipt['duration_seconds'])
                write_new(output/cid/'finish.json',row)
            except BaseException as exc:
                interrupted=exc
            if interrupted is not None or requested_stop or row['status']!='valid': break
    result=save_report()
    if interrupted is not None: raise interrupted
    return result


def main():
    p=argparse.ArgumentParser(description='Manual MedQA collection and separately documented failure retries; no scoring.')
    p.add_argument('--config',type=Path,default=CONFIG_PATH)
    p.add_argument('--protocol',type=Path,help='actual vault protocol, required for Luna Decisions and failure retries')
    sub=p.add_subparsers(dest='command',required=True)
    plan=sub.add_parser('plan',help='no-call count preview')
    plan.add_argument('--experiment',choices=['E1','E2','E1+E2'],default='E1+E2');plan.add_argument('--count',type=int)
    resolve=sub.add_parser('resolve-incomplete-read',help='append one reviewed interrupted-response failure; no calls')
    resolve.add_argument('--package',type=Path,required=True)
    resolve.add_argument('--start-event',required=True)
    resolve.add_argument('--reviewer',required=True)
    resolve.add_argument('--partial-bytes',type=int,required=True)
    retry=sub.add_parser('retry-failures',help='supplementary first-valid recovery; up to ten additional calls per original failure')
    retry.add_argument('--source',type=Path,action='append',required=True,help='complete immutable original package; repeat for Luna')
    retry.add_argument('--decisions-config',type=Path,default=ROOT/'benchmarking/decisions.toml')
    for name in ('questions','input-manifest','output'): retry.add_argument('--'+name,type=Path,required=True)
    retry_mode=retry.add_mutually_exclusive_group(required=True)
    retry_mode.add_argument('--live',action='store_true')
    retry_mode.add_argument('--mock',action='store_true')
    retry.add_argument('--mock-registry',type=Path)
    retry.add_argument('--check-only',action='store_true',help='read-only candidate check; no credentials, registry writes or calls')
    retry.add_argument('--reviewer')
    retry.add_argument('--reviewed-stop',action='append',help='exact paused-provider finish-event SHA-256; repeat for multiple providers')
    run=sub.add_parser('run',help='explicit --live collection or --mock fictional test')
    for name in ('questions','input-manifest','output'): run.add_argument('--'+name,type=Path,required=True)
    run.add_argument('--study-id',required=True)
    run.add_argument('--seed',type=int,help='defaults to the configured schedule seed')
    run.add_argument('--experiment',choices=['E1','E2','E1+E2'],default='E1+E2')
    run_mode=run.add_mutually_exclusive_group(required=True)
    run_mode.add_argument('--live',action='store_true',help='intentionally start paid requests')
    run_mode.add_argument('--mock',action='store_true',help='explicit fictional test; no API calls')
    run.add_argument('--mock-registry',type=Path,help='isolated fictional registry; forbidden with --live')
    run.add_argument('--continue-from',type=Path,help='immutable stopped parent package; start only its unstarted identities')
    run.add_argument('--reviewer',help='person accepting the recorded provider stop or reviewed interruption')
    run.add_argument('--reviewed-stop',action='append',help='exact provider-stop or interruption-review event SHA-256; repeat for multiple events')
    run.add_argument('--execution-amendment',choices=[CAPACITY_AMENDMENT,DEEPSEEK_TEN_AMENDMENT],help='explicitly apply a bounded accepted scheduling amendment to this continuation')
    run.add_argument('--check-only',action='store_true',help='validate continuation without files, credentials or calls')
    smoke=sub.add_parser('smoke',help='one development item per configuration; explicit mode, no scoring')
    for name in ('questions','input-manifest','output'): smoke.add_argument('--'+name,type=Path,required=True)
    smoke_mode=smoke.add_mutually_exclusive_group(required=True)
    smoke_mode.add_argument('--live',action='store_true',help='intentionally start paid requests')
    smoke_mode.add_argument('--mock',action='store_true',help='explicit fictional test; no API calls')
    args=p.parse_args()
    try:
        if args.command=='resolve-incomplete-read':
            print(json.dumps(resolve_incomplete_read(args.package,start_event=args.start_event,
                reviewer=args.reviewer,partial_bytes=args.partial_bytes),indent=2));return 0
        config=load_config(args.config)
        if args.command=='retry-failures':
            if args.protocol is None: raise ValueError('failure retries require actual amended --protocol')
            from benchmarking.retries import run_retries
            result=run_retries(args.source,{'original':config,'decisions':load_config(args.decisions_config)},
                {'original':args.config,'decisions':args.decisions_config},args.questions,args.input_manifest,
                args.protocol,args.output,mock=args.mock,mock_registry=args.mock_registry,check_only=args.check_only,
                reviewer=args.reviewer,reviewed_stops=args.reviewed_stop)
            print(json.dumps(result,indent=2))
            if not args.check_only and not result['complete']: return 2
        elif args.command=='plan':
            print(json.dumps(plan_counts(config,args.experiment,args.count),indent=2))
        elif args.command=='smoke':
            result=smoke_test(config,args.questions,args.input_manifest,args.output,mock=args.mock,config_path=args.config,protocol_path=args.protocol)
            print(str(args.output/'Smoke_Test.md'))
            if not result['passed']: return 2
        else:
            if args.continue_from and (not args.reviewer or (not args.reviewed_stop and args.execution_amendment!=DEEPSEEK_TEN_AMENDMENT)):
                raise ValueError('--continue-from requires --reviewer and exact --reviewed-stop events, except the ten-slot amendment of a drained parent')
            if args.check_only and not args.continue_from:
                raise ValueError('--check-only requires continuation')
            if not args.continue_from and (args.reviewer or args.reviewed_stop or args.execution_amendment): raise ValueError('review metadata requires --continue-from')
            if args.mock_registry:
                if args.live: raise ValueError('--mock-registry is forbidden with --live')
                config['execution']['study_registry']=str(args.mock_registry.resolve())
            prep=validate_input_manifest(args.input_manifest,args.questions)
            if args.live and prep['provenance']!='prepared': raise ValueError('live inference requires prepared research inputs')
            seed=config['execution']['seed'] if args.seed is None else args.seed
            if args.live and seed!=config['execution']['seed']: raise ValueError('seed differs from configured schedule')
            inventory=run_inventory(config,args.questions,args.input_manifest,study_id=args.study_id,config_path=args.config,
                resume_output=args.output,continuation=bool(args.continue_from),protocol_path=args.protocol)
            manifest=make_manifest(config,args.questions,study_id=args.study_id,experiment=args.experiment,
                provenance='definitive' if args.live else 'mock',seed=seed,
                keys_sha256=prep['keys_sha256'],preparation_sha256=prep['preparation_manifest_sha256'],inventory=inventory,
                content_sha256=prep.get('content_manifest_sha256'))
            if args.continue_from:
                manifest=reviewed_continuation(manifest,args.continue_from,args.output,
                    reviewer=args.reviewer,reviewed_stops=args.reviewed_stop or [],amendment=args.execution_amendment)
            result=run_package(manifest,args.questions,config,args.output,mock=args.mock,
                inventory=inventory,input_manifest_path=args.input_manifest,config_path=args.config,check_only=args.check_only)
            print(json.dumps(result,indent=2))
            if not args.check_only and not result['complete']: return 2
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print('Blocked: '+str(exc),file=sys.stderr);return 1
    return 0

if __name__=='__main__': raise SystemExit(main())
