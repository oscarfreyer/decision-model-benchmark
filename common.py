"""Pure v3 package contract. Preparation/configuration/review schemas remain v2."""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import random
import re
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parent
SCHEMA_VERSION = 3
LEGACY_ROSTER = tuple([f'C{i:02d}' for i in range(1,25)]+['J01'])
PREPRINT_ROSTER = ('C03','C05','C06','C09','C11','C13','C14','C15','C17','C18','C19','C21','C22','C23','C24','J01')
ROSTER_DECISION = 'preprint-budget-roster-20261006'
KIMI_INSTANT_DECISION = 'preprint-kimi-instant-replacement-20261006'
KIMI_INSTANT_ROSTER = tuple(cid for cid in PREPRINT_ROSTER if cid != 'C21')[:-1] + ('C25', 'J01')
BUDGET_ROSTERS = {ROSTER_DECISION: PREPRINT_ROSTER, KIMI_INSTANT_DECISION: KIMI_INSTANT_ROSTER}
OBSERVED_COST_DECISION = 'R5-observed-cost-unverified-tokens-20261006'
EXECUTION_FREEZE_DECISION = 'separate-execution-analysis-freezes-20261006'

def execution_freeze_enabled(config):
    return EXECUTION_FREEZE_DECISION in config.get('accepted_decisions', [])

def budget_roster_decision(config):
    """Keep completed Mistral packages distinct from the prospective replacement."""
    accepted = config.get('accepted_decisions', [])
    return next((d for d in (KIMI_INSTANT_DECISION, ROSTER_DECISION) if d in accepted), None)

TERMINAL_STATUSES = frozenset({'valid', 'invalid', 'refused', 'truncated', 'http_error', 'timeout', 'transport_error', 'drift'})
BINDING_NAMES = frozenset({'code', 'environment', 'config', 'prompt', 'questions', 'keys', 'protocol_record'})

def strict_json(text):
    def pairs(items):
        obj = {}
        for k, v in items:
            if k in obj: raise ValueError('duplicate JSON key: ' + k)
            obj[k] = v
        return obj
    def invalid(value): raise ValueError('nonfinite JSON number: ' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)

def fingerprint(value):
    # Canonical JSON hashes bind object content independent of dictionary order.
    # file_hash instead binds original bytes, including formatting and comments.
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def file_hash(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def utc_now(): return datetime.now(timezone.utc).isoformat()
def read_jsonl(path): return [strict_json(x) for x in Path(path).read_text().splitlines() if x.strip()]
def write_new(path, value, *, jsonl=False):
    # Exclusive creation preserves prior artifacts; fsync finishes persistence
    # before a caller records the file's digest in the attempt journal.
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = ''.join(json.dumps(x, ensure_ascii=False, allow_nan=False) + '\n' for x in value) if jsonl else json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8') as f:
        import os
        f.write(text);f.flush();os.fsync(f.fileno())

def _text(value, name):
    if not isinstance(value, str) or not value.strip(): raise ValueError(name + ' must be nonempty text')

@dataclass(frozen=True)
class Option:
    label: str
    text: str

@dataclass(frozen=True)
class Question:
    benchmark: str
    item_id: str
    split: str
    stem: str
    options: tuple[Option, ...]
    subject: str | None = None
    @property
    def labels(self): return tuple(x.label for x in self.options)
    def validate(self):
        for name in ('benchmark', 'item_id', 'split', 'stem'): _text(getattr(self, name), name)
        if len(self.options) != 4 or set(self.labels) != set('ABCD'): raise ValueError('exactly four supplied labels A-D required')
        for opt in self.options: _text(opt.text, 'option text')

def identity(item):
    if isinstance(item, Question): return item.benchmark, item.split, item.item_id
    return item['benchmark'], item['split'], item['item_id']

def question_record(q):
    return {'benchmark': q.benchmark, 'split': q.split, 'item_id': q.item_id, 'stem': q.stem,
            'options': [asdict(o) for o in q.options]}

def question_hash(q): return fingerprint(question_record(q))
def content_hash(q):
    # Exact question hashes bind identity/text/order; this normalized content hash
    # detects whitespace/case duplicates across IDs and splits without editing text.
    norm = lambda s: re.sub(r'\s+', ' ', s.strip()).casefold()
    return fingerprint([norm(q.stem), [[o.label, norm(o.text)] for o in q.options]])

def load_questions(path):
    # An exact field allowlist rejects answer-bearing metadata, not just known
    # answer field names. Source/key handling belongs to preparation and analysis.
    result = []; ids = set()
    for row in read_jsonl(path):
        if set(row) != {'benchmark','split','item_id','stem','options'}: raise ValueError('inference question fields must be answer-free and exact')
        if not isinstance(row['options'], list) or any(set(x) != {'label', 'text'} for x in row['options']): raise ValueError('invalid options')
        q = Question(row['benchmark'], row['item_id'], row['split'], row['stem'], tuple(Option(**x) for x in row['options'])); q.validate()
        if identity(q) in ids: raise ValueError('duplicate question identity')
        ids.add(identity(q)); result.append(q)
    if not result: raise ValueError('empty question file')
    return result

def load_keys(path, questions):
    keys = {}
    for row in read_jsonl(path):
        key = identity(row)
        if key in keys: raise ValueError('duplicate key identity')
        if row['reference_answer'] not in 'ABCD' or len(row['reference_answer']) != 1: raise ValueError('invalid key label')
        keys[key] = row
    if set(keys) != {identity(q) for q in questions}: raise ValueError('exact question/key coverage required')
    return keys

def dataset_audit(paths, keys_path, expected, provenance):
    questions = [q for path in paths for q in load_questions(path)]
    if len({identity(q) for q in questions}) != len(questions): raise ValueError('duplicate IDs across files')
    if len({content_hash(q) for q in questions}) != len(questions): raise ValueError('normalized duplicate/overlap')
    load_keys(keys_path, questions)
    counts = {}
    for q in questions:
        k = q.benchmark + '/' + q.split; counts[k] = counts.get(k, 0) + 1
    if counts != expected: raise ValueError('split counts mismatch')
    return {'schema_version':2,'ready':True,'counts':counts,'provenance':provenance,
            'files':{str(p):file_hash(p) for p in paths}, 'keys_hash':file_hash(keys_path),
            'items':[{'id': list(identity(q)), 'hash':question_hash(q)} for q in questions]}

def native_probability(vector, selected_label, labels=('A','B','C','D'), concentration=None, semantics='unverified'):
    # Retain the provider vector and selected decision. Validation derives metadata
    # without normalization, clipping, argmax substitution or concentration scaling.
    concentration_reason='provider_unavailable' if concentration is None else None
    if concentration is not None:
        try:
            if isinstance(concentration,bool) or not isinstance(concentration,(str,int,float,Decimal)) or not Decimal(str(concentration)).is_finite(): raise ValueError()
        except (ValueError,InvalidOperation,TypeError): concentration=None;concentration_reason='nonfinite_or_invalid'
        else:
            # Provider JSON decimals remain exact and serializable in both adapters.
            if isinstance(concentration, Decimal): concentration=str(concentration)
    result = {'native_vector':vector,'sum':None,'deviation':None,'valid':False,'reason':None,
              'selected_probability':None,'concentration':concentration,'concentration_reason':concentration_reason,'semantics':semantics}
    if not isinstance(vector, dict) or set(vector) != set(labels): result['reason']='label_coverage'; return result
    try:
        if any(isinstance(x,bool) or not isinstance(x,(str,int,float,Decimal)) for x in vector.values()): raise ValueError()
        values = {k:Decimal(str(v)) for k,v in vector.items()}
        if any(not v.is_finite() or not 0 <= v <= 1 for v in values.values()): raise ValueError()
        total = sum(values.values()); deviation = abs(total - 1)
        result.update(sum=str(total), deviation=str(deviation))
        # Inclusive 0.02 tolerance reflects four values rounded to two decimals.
        # Passing tolerance does not imply unit sum or accepted probability semantics.
        if deviation > Decimal('0.02'): result['reason']='sum_outside_tolerance'; return result
        if selected_label not in values: result['reason']='selected_label_unavailable'; return result
        result.update(valid=True, reason=None, selected_probability=str(values[selected_label]))
    except (InvalidOperation,ValueError,TypeError): result['reason']='nonfinite_or_range'
    return result

def empty_probability(reason='unavailable'):
    return {'native_vector':None,'sum':None,'deviation':None,'valid':False,'reason':reason,
            'selected_probability':None,'concentration':None,'concentration_reason':'provider_unavailable','semantics':'unavailable'}

def unavailable(reason='provider_unavailable'): return {'value':None,'reason':reason}
def measurement(value=None, reason='provider_unavailable'):
    # Zero is an observed measurement; only None means unavailable, with a reason.
    return {'value':value,'reason':None if value is not None else reason}

def validate_measurement(record, name, *, integer=False):
    if not isinstance(record,dict) or set(record) != {'value','reason'}: raise ValueError('invalid measurement: '+name)
    value = record['value']
    if value is None:
        _text(record['reason'],name+' missing reason')
    elif isinstance(value,bool) or not isinstance(value,(str,int,float)) or not math.isfinite(float(value)) or float(value)<0 or record['reason'] is not None or (integer and (type(value) is not int)):
        raise ValueError('invalid measured value: '+name)

def attempt_identity(row):
    r = row.get('identity',row)
    return r['benchmark'],r['split'],r['item_id'],r['config_id'],r['run_id']

def validate_manifest(manifest):
    if manifest.get('schema_version') != SCHEMA_VERSION: raise ValueError('unsupported package schema; historical v2 requires archived v2 readers')
    if manifest.get('provenance') not in {'mock','development','definitive'}: raise ValueError('explicit provenance required')
    _text(manifest.get('study_id'),'study_id')
    for key in ('benchmark','split'): _text(manifest.get(key),key)
    if not BINDING_NAMES.issubset(manifest.get('bindings',{})): raise ValueError('missing code/environment/config/prompt/questions/keys/protocol_record bindings')
    for value in manifest['bindings'].values():
        if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value): raise ValueError('invalid binding digest')
    items=manifest['items']; configs=manifest['configurations']; runs=manifest['runs']
    for sequence, key in ((items,'item_id'),(configs,'config_id'),(runs,'run_id')):
        if not sequence or len({r[key] for r in sequence}) != len(sequence): raise ValueError('empty/duplicate manifest identities')
    for item in items:
        if set(item)!= {'item_id','content_sha256','labels'} or set(item['labels'])!=set('ABCD') or len(item['labels'])!=4 or not re.fullmatch('[0-9a-f]{64}',item['content_sha256']): raise ValueError('invalid manifest question')
    if 'schedule' in manifest:
        order=manifest['schedule'].get('item_order')
        if order is not None and (len(order)!=len(items) or set(order)!={x['item_id'] for x in items}): raise ValueError('scheduling item order mismatch')
    ids={r['config_id'] for r in configs}
    for c in configs:
        for k in ('config_id','model_id','base_model','effort','adapter'): _text(c.get(k),k)
        if 'pair_id' not in c: raise ValueError('pair_id required (null for singles)')
    for r in runs:
        if r['config_id'] not in ids or type(r['replicate']) is not int or r['replicate']<1: raise ValueError('invalid run configuration/replicate')
        if r['role'] not in {'primary','secondary'} or r['experiment_id'] not in {'E0','E1','E2'}: raise ValueError('invalid run role/experiment')
        if r['experiment_id']=='E2' and (r['config_id']!='J01' or r['replicate'] not in (2,3) or r['role']!='secondary'): raise ValueError('E2 must be secondary JEV runs2/3')
        if r['experiment_id']=='E0' and (r['role']!='primary' or r['replicate']!=1): raise ValueError('E0 must be primary run1')
        if r['experiment_id']=='E1' and (r['role']!='primary' or r['replicate']!=1): raise ValueError('E1 must be primary run1')
        if r['role']=='primary' and r['replicate']!=1: raise ValueError('primary must use run1')
    if len({(r['config_id'],r['replicate']) for r in runs})!=len(runs): raise ValueError('duplicate run replicate')
    if manifest['provenance']=='definitive':
        if len(items)!=1273 or ids not in (set(LEGACY_ROSTER), *(set(r) for r in BUDGET_ROSTERS.values())): raise ValueError('definitive requires 1273 items and exact reviewed roster')
        if manifest['split']!='evaluation': raise ValueError('definitive requires evaluation split')
    decision=manifest.get('roster_decision')
    matching_decision=next((d for d,r in BUDGET_ROSTERS.items() if ids==set(r)),None)
    if matching_decision is not None and decision!=matching_decision: raise ValueError('budget roster requires its explicit decision identity')
    if decision is not None and (decision not in BUDGET_ROSTERS or ids!=set(BUDGET_ROSTERS[decision])): raise ValueError('roster decision/configuration mismatch')
    if 'policy' in manifest.get('schedule',{}): validate_schedule_policy(manifest['schedule']['policy'])
    if manifest.get('key_manifest',{}).get('sha256')!=manifest['bindings']['keys']: raise ValueError('key manifest/binding mismatch')
    return manifest

def manifest_configurations(config):
    return [{'config_id':cid, **{k:config['models'][cid][k] for k in ('model_id','base_model','effort','adapter')},
             'pair_id':config['models'][cid]['pair_id'] or None} for cid in config['model_roster']]

def manifest_runs(study_id, configurations, experiments):
    runs=[]
    for eid in ('E0','E1'):
        if eid in experiments:
            runs += [{'run_id':f'{study_id}-{eid}-{c["config_id"]}-r1','experiment_id':eid,
                      'config_id':c['config_id'],'replicate':1,'role':'primary'} for c in configurations]
    if 'E2' in experiments:
        runs += [{'run_id':f'{study_id}-E2-J01-r{n}','experiment_id':'E2','config_id':'J01',
                  'replicate':n,'role':'secondary'} for n in (2,3)]
    return runs

def validate_schedule_policy(policy):
    """Exact, versioned scheduler settings; no unknown runtime knobs."""
    required={'version','concurrency','provider_concurrency','provider_min_interval_seconds','rate_limit_stop_after'}
    if not isinstance(policy,dict) or type(policy.get('version')) is not int or policy['version'] not in (1,2): raise ValueError('unsupported scheduler policy')
    if policy['version']==2: required.add('provider_concurrency_overrides')
    if set(policy)!=required: raise ValueError('unsupported scheduler policy')
    ceiling=4 if policy['version']==1 else None
    for field,maximum in (('concurrency',ceiling),('provider_concurrency',ceiling),('rate_limit_stop_after',100)):
        value=policy[field]
        if type(value) is not int or value<1 or (maximum is not None and value>maximum): raise ValueError('invalid scheduler '+field)
    if policy['provider_concurrency']>policy['concurrency']: raise ValueError('provider concurrency exceeds global limit')
    overrides=policy.get('provider_concurrency_overrides',{})
    if not isinstance(overrides,dict): raise ValueError('provider concurrency override map required')
    for provider,limit in overrides.items():
        if not isinstance(provider,str) or not provider.strip() or type(limit) is not int or not 1<=limit<=policy['concurrency']:
            raise ValueError('invalid provider concurrency override')
    intervals=policy['provider_min_interval_seconds']
    if not isinstance(intervals,dict): raise ValueError('provider pacing map required')
    for provider,interval in intervals.items():
        if not isinstance(provider,str) or not provider or type(interval) not in (int,float) or not math.isfinite(interval) or interval<0:
            raise ValueError('invalid provider pacing interval')
    return policy


def scheduler_policy(config):
    execution=config.get('execution',{})
    if execution.get('scheduler_version') is None: return None
    policy={'version':execution['scheduler_version'],
        **{key:execution.get(key) for key in ('concurrency','provider_concurrency','provider_min_interval_seconds','rate_limit_stop_after')}}
    if execution['scheduler_version']==2:
        policy['provider_concurrency_overrides']=execution.get('provider_concurrency_overrides',{})
    elif execution.get('provider_concurrency_overrides'):
        raise ValueError('provider concurrency overrides require scheduler version 2')
    policy=validate_schedule_policy(policy)
    providers={m.get('route') or ('typesafe-direct' if m.get('adapter')=='jev' else 'fictional-provider')
               for m in config.get('models',{}).values()}
    if set(policy.get('provider_concurrency_overrides',{}))-providers:
        raise ValueError('provider concurrency override names an unconfigured provider')
    return policy


def validate_manifest_config(manifest, config, questions):
    """Bind human-facing identities, roles and schedule to executable settings/input order."""
    # A matching config digest alone does not validate separately stored roster,
    # role or schedule metadata; reconstruct each from executable configuration.
    if manifest['configurations']!=manifest_configurations(config): raise ValueError('manifest configuration metadata mismatch')
    experiments={r['experiment_id'] for r in manifest['runs']}
    if experiments not in ({'E0'},{'E1'},{'E2'},{'E1','E2'}): raise ValueError('unsupported experiment combination')
    if manifest['provenance']=='development' and experiments!={'E0'}: raise ValueError('development provenance requires E0; evaluation requires definitive freeze')
    if manifest['runs']!=manifest_runs(manifest['study_id'],manifest['configurations'],experiments): raise ValueError('manifest run metadata mismatch')
    expected=[{'item_id':q.item_id,'content_sha256':question_hash(q),'labels':list(q.labels)} for q in questions]
    if manifest['items']!=expected: raise ValueError('manifest preparation order/content mismatch')
    if any(q.benchmark!=manifest['benchmark'] or q.split!=manifest['split'] for q in questions): raise ValueError('manifest benchmark/split mismatch')
    if manifest['split']!=('development' if experiments=={'E0'} else 'evaluation'): raise ValueError('manifest experiment/split mismatch')
    schedule=manifest.get('schedule',{});seed=schedule.get('seed')
    if type(seed) is not int or seed<0: raise ValueError('manifest scheduling seed required')
    if manifest['provenance']!='mock' and seed!=config['execution']['seed']: raise ValueError('manifest seed differs from accepted runtime')
    order=[q.item_id for q in questions];random.Random(seed).shuffle(order)
    expected_schedule={'kind':'round_robin','seed':seed,'item_order':order}
    policy=scheduler_policy(config)
    if policy is not None: expected_schedule['policy']=policy
    if schedule!=expected_schedule: raise ValueError('manifest schedule mismatch')
    expected_decision=budget_roster_decision(config)
    if manifest.get('roster_decision')!=expected_decision: raise ValueError('manifest roster decision mismatch')

def study_context(manifest):
    """The shared E1/E2 study identity; runs are separately registered packages."""
    return {k:v for k,v in manifest.items() if k!='runs'}

def expected_identities(manifest):
    validate_manifest(manifest)
    return [(manifest['benchmark'],manifest['split'],q['item_id'],r['config_id'],r['run_id']) for r in manifest['runs'] for q in manifest['items']]

def validate_finish(record, manifest, *, allowed_ids=None, item_labels=None):
    # Recompute native-vector derived fields rather than trusting saved validity.
    # Failed final answers must not retain a label that offline scoring could use.
    ident=attempt_identity(record)
    if ident not in (allowed_ids if allowed_ids is not None else set(expected_identities(manifest))): raise ValueError('unexpected attempt')
    if record.get('status') not in TERMINAL_STATUSES: raise ValueError('unknown terminal status')
    labels=item_labels[ident[2]] if item_labels is not None else next(x['labels'] for x in manifest['items'] if x['item_id']==ident[2])
    if record['status']=='valid':
        if record.get('selected_label') not in labels: raise ValueError('valid answer requires supplied label')
    elif record.get('selected_label') is not None: raise ValueError('failed answer cannot be selected for scoring')
    if record['status']=='valid' and (record['artifacts']['raw_response'] is None or record['artifacts']['final'] is None): raise ValueError('valid answer requires raw/final artifacts')
    if record['status']!='valid': _text(record.get('failure_reason'),'failure reason')
    p=record['probability']
    if p.get('concentration') is not None:
        try:
            scalar=p['concentration']
            if isinstance(scalar,bool) or not isinstance(scalar,(str,int,float)) or not Decimal(str(scalar)).is_finite(): raise ValueError()
        except (ValueError,InvalidOperation,TypeError): raise ValueError('nonfinite concentration') from None
    if set(p)!=set(empty_probability()): raise ValueError('probability fields mismatch')
    if p['native_vector'] is not None:
        checked=native_probability(p['native_vector'], record['selected_label'], labels,p['concentration'],p['semantics'])
        if p['concentration'] is None:
            if p['concentration_reason'] not in {'provider_unavailable','nonfinite_or_invalid'}: raise ValueError('invalid concentration missing reason')
            checked['concentration_reason']=p['concentration_reason']
        elif p['concentration_reason'] is not None: raise ValueError('finite concentration cannot have missing reason')
        if checked!=p: raise ValueError('probability validity/derived fields mismatch')
    elif p['valid'] or p['selected_probability'] is not None or not p['reason']: raise ValueError('unavailable probability mismatch')
    for k in ('input_tokens','output_tokens','reasoning_tokens','final_tokens','billed_tokens'): validate_measurement(record['usage'][k],k,integer=True)
    if record['usage']['reasoning_in_output'] not in (True,False,None): raise ValueError('invalid accounting flag')
    _text(record['usage']['accounting_definition'],'accounting definition')
    for k in ('observed_usd','estimated_usd'): validate_measurement(record['billing'][k],k)
    if record['billing']['estimated_usd']['value'] is not None:
        _text(record['billing'].get('rate_provenance'),'estimated rate provenance'); _text(record['billing'].get('rate_date'),'estimated rate date')
    for k in ('duration_seconds',): validate_measurement(record['timing'][k],k)
    if 'queue_wait_seconds' in record['timing']:
        validate_measurement(record['timing']['queue_wait_seconds'],'queue_wait_seconds')
    if 'scheduler' in record:
        metadata=record['scheduler']
        if not manifest['schedule'].get('policy') or not isinstance(metadata,dict) or set(metadata)!={'provider','response_status','stop_reason'}:
            raise ValueError('invalid scheduler finish metadata')
        _text(metadata['provider'],'scheduler provider')
        status=metadata['response_status']
        if status is not None and (type(status) is not int or not 100<=status<=599): raise ValueError('invalid scheduler response status')
        if metadata['stop_reason'] not in (None,'provider_rate_limit_stop') or metadata['stop_reason'] and status!=429:
            raise ValueError('invalid provider stop reason')
    for k in ('started_at','finished_at'):
        if datetime.fromisoformat(record['timing'][k]).tzinfo is None: raise ValueError('timezone required')
    if datetime.fromisoformat(record['timing']['finished_at']) < datetime.fromisoformat(record['timing']['started_at']): raise ValueError('negative wall time')
    if record['billing']['currency']!='USD': raise ValueError('USD accounting required')
    if 'rate_provenance' not in record['billing'] or 'rate_date' not in record['billing']: raise ValueError('billing provenance required')
    for k in ('requested_model','returned_model','requested_route','returned_route','effective_controls','provider_request_id','finish_reason'):
        if k not in record['route']: raise ValueError('missing route field: '+k)
    for k in ('request','raw_response','final','reasoning'):
        if k not in record['artifacts']: raise ValueError('missing artifact reference')
    if record['artifacts']['request'] is None: raise ValueError('request artifact required')
    return record

def artifact_ref(path, package): return {'path':str(Path(path).relative_to(package)),'sha256':file_hash(path)}
def check_artifact(ref, package):
    if ref is None: return
    if set(ref)!={'path','sha256'}: raise ValueError('invalid artifact reference')
    rel=Path(ref['path']); root=Path(package).resolve(); path=(root/rel).resolve()
    if rel.is_absolute() or not path.is_relative_to(root) or path.is_symlink(): raise ValueError('artifact path escapes package')
    if file_hash(path)!=ref['sha256']: raise ValueError('immutable artifact hash mismatch')

def read_package(package):
    """Return manifest, terminal records, coverage. Verify journal chain/raw hashes."""
    package=Path(package)
    manifest=validate_manifest(strict_json((package/'manifest.json').read_text()))
    expected=set(expected_identities(manifest)); item_labels={x['item_id']:x['labels'] for x in manifest['items']}; starts={}; finishes={}; previous=None
    plan_seen=False;manifest_sha256=fingerprint(manifest)
    # Plan -> unique start -> unique finish is the only accepted progression.
    # A start without finish is indeterminate; no start is unstarted. Neither is
    # a completed failure, even though a started request may already be charged.
    for event in read_jsonl(package/'events.jsonl'):
        digest=event.pop('event_sha256',None)
        if event.get('previous_sha256')!=previous or fingerprint(event)!=digest: raise ValueError('journal chain/hash mismatch')
        previous=digest
        if event.get('schema_version')!=SCHEMA_VERSION or event.get('bindings')!=manifest['bindings']: raise ValueError('event binding mismatch')
        if event.get('manifest_sha256')!=manifest_sha256: raise ValueError('event manifest binding mismatch')
        if event['event']=='plan':
            if plan_seen or starts or finishes: raise ValueError('unique initial plan event required')
            plan_seen=True;continue
        if not plan_seen: raise ValueError('initial manifest-bound plan event required')
        ident=attempt_identity(event)
        if ident not in expected: raise ValueError('unexpected journal identity')
        if event['event']=='start':
            if ident in starts: raise ValueError('duplicate start')
            check_artifact(event['request'],package); starts[ident]=event
        elif event['event']=='finish':
            if ident not in starts or ident in finishes: raise ValueError('finish without unique start')
            record=validate_finish(event['record'],manifest,allowed_ids=expected,item_labels=item_labels)
            if attempt_identity(record)!=ident or record['artifacts']['request']!=starts[ident]['request']: raise ValueError('start/finish identity or request mismatch')
            for ref in record['artifacts'].values(): check_artifact(ref,package)
            finishes[ident]=record
        else: raise ValueError('unknown journal event')
    if not plan_seen: raise ValueError('initial manifest-bound plan event required')
    return manifest,list(finishes.values()),{'expected':len(expected),'terminal':len(finishes),
        'unstarted':[list(x) for x in sorted(expected-set(starts))],
        'indeterminate':[list(x) for x in sorted(set(starts)-set(finishes))],
        'complete':set(finishes)==expected,'last_event_sha256':previous}

def environment_record():
    return {'python':platform.python_version(),'platform':platform.platform(),
            'packages':dict(sorted((d.metadata['Name'].lower(),d.version) for d in importlib.metadata.distributions()))}

ANALYSIS_SOURCES = ('common.py', 'requirements.txt', 'requirements.lock',
    'analysis/__init__.py', 'analysis/pipeline.py', 'analysis/readiness.py',
    'analysis/reproduction.py', 'analysis/statistics.py', 'analysis/figures.py')
PRODUCTION_SOURCES = ANALYSIS_SOURCES + (
    'preparation/__init__.py', 'preparation/prepare_medqa.py', 'preparation/subset.py',
    'benchmarking/__init__.py', 'benchmarking/run.py', 'benchmarking/adapters.py',
    'benchmarking/readiness.py', 'benchmarking/study.py', 'benchmarking/e0_launch.py')

def analysis_code_record(root=ROOT):
    """Scientific consumer dependency closure; development files are separate evidence."""
    return {name:file_hash(Path(root)/name) for name in sorted(ANALYSIS_SOURCES)}

EXECUTION_SOURCES = tuple(name for name in PRODUCTION_SOURCES
    if not name.startswith('analysis/') and name not in {'requirements.txt', 'requirements.lock'})

def historical_source_path(root, name):
    """Retired launcher/readiness source stays in the exact private reader snapshot."""
    path=Path(root)/name
    if not path.exists() and name in {'benchmarking/readiness.py','benchmarking/e0_launch.py'}:
        return Path(root)/'benchmarking/legacy'/Path(name).name
    return path

def execution_code_record(root=ROOT):
    return {name:file_hash(historical_source_path(root,name)) for name in sorted(EXECUTION_SOURCES)}

def execution_environment_record():
    # Preparation and inference use only CPython's standard library.
    return {'python':platform.python_version(),'platform':platform.platform(),'packages':{}}

def run_environment_record(config):
    return execution_environment_record() if execution_freeze_enabled(config) else environment_record()

def execution_git_record():
    # Analysis work and later commits do not change the accepted execution bytes.
    result=subprocess.run(['git','status','--porcelain','--',*EXECUTION_SOURCES,'config.toml'],
        cwd=ROOT,capture_output=True,text=True,check=True)
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True)
    return {'commit':commit.stdout.strip(),'dirty':bool(result.stdout.strip())}

def code_record(root=ROOT, *, config=None):
    # Explicit production files make a code-only release reproduce the same binding.
    # Missing required files fail closed. Historical manifests/readers stay unchanged.
    if config is not None and execution_freeze_enabled(config): return execution_code_record(root)
    return {name:file_hash(historical_source_path(root,name)) for name in sorted(PRODUCTION_SOURCES)}

def git_record():
    def run(*args):
        r=subprocess.run(['git',*args],cwd=ROOT,capture_output=True,text=True,check=True); return r.stdout.strip()
    return {'commit':run('rev-parse','HEAD'),'dirty':bool(run('status','--porcelain'))}

def load_config(path=ROOT/'config.toml'):
    with Path(path).open('rb') as f: return tomllib.load(f)

FREEZE_ROLES = frozenset({'questions_medqa','keys_medqa','audit_medqa','protocol_record','environment_lock',
    'method_review','route_control_evidence','data_permissions'})
REVIEW_TOPICS = ('protocol','model_routes_controls','data_permissions','runtime','statistics',
                'confidence_policy','reference_adjudication','preregistration_disposition','milestone_m01')

EXECUTION_FREEZE_ROLES = FREEZE_ROLES - {'method_review'}
EXECUTION_REVIEW_TOPICS = ('protocol','model_routes_controls','data_permissions','runtime',
    'preregistration_disposition','milestone_m01')

def freeze_requirements(candidate, config):
    if candidate.get('schema_version') == 3 and candidate.get('freeze_scope') == 'execution' and execution_freeze_enabled(config):
        return EXECUTION_FREEZE_ROLES, EXECUTION_REVIEW_TOPICS
    if candidate.get('schema_version') == 2 and not execution_freeze_enabled(config):
        return FREEZE_ROLES, REVIEW_TOPICS
    raise ValueError('freeze schema/scope does not match accepted configuration')

def config_binding(config):
    # Analysis settings and review locators do not change a new execution freeze.
    omitted={'freeze','analysis'} if execution_freeze_enabled(config) else {'freeze'}
    return fingerprint({k:v for k,v in config.items() if k not in omitted})

# Native effort labels selected in Oscar's R1 review; historical v3 manifests
# retain their original low/high labels. No serialized field or schema changes.
R1_EFFORT_DECISION = 'R1-native-pairs-20261005'
EFFORT_PAIRS = {f'P{i:02d}': ('low', 'high') for i in range(1, 11)}
EFFORT_PAIRS.update(P08=('low', 'xhigh'), P09=('low', 'xhigh'), P10=('none', 'high'))


def effort_pair_levels(pair_id, lower_effort, upper_effort):
    levels = (lower_effort, upper_effort)
    if pair_id not in EFFORT_PAIRS or levels not in {('low', 'high'), EFFORT_PAIRS[pair_id]}:
        raise ValueError('Unapproved native effort pair')
    return levels


def native_effort_request_valid(model):
    reasoning = model.get('request_settings', {}).get('reasoning', {})
    return (isinstance(reasoning, dict) and reasoning.get('effort') == model.get('effort')
            and (reasoning.get('enabled') is False if model.get('effort') == 'none'
                 else reasoning.get('enabled') is not False))


REGION_DISCLOSURE_DECISION = 'R1-unknown-region-20261005'
TOKEN_TARGET_DECISION = 'R1-requested-token-target-20261005'


def token_limit_request_valid(config, model, limit):
    """Validate the requested target; keep backend accounting evidence separate."""
    settings = model.get('request_settings', {})
    if limit.get('value') != 16384 or settings.get(limit.get('field')) != 16384:
        return False
    if TOKEN_TARGET_DECISION in config.get('accepted_decisions', []):
        return limit.get('field') == 'max_completion_tokens' and 'max_tokens' not in settings
    # Historical configurations retain their reviewed combined-ceiling rule.
    return limit.get('counts_reasoning_and_final') is True


def unknown_region_accepted(config, model, evidence):
    """Human exception for this exact route; never manufacture geographic evidence."""
    approval=evidence.get('region_exception',{})
    settings=fingerprint({k:v for k,v in model.items() if k not in
        {'control_evidence','control_evidence_sha256','display_name','effort','request_settings'}})
    protocol=evidence.get('protocol_sha256')
    if not isinstance(approval,dict): return False
    try:
        datetime.fromisoformat(approval['approved_at'])
    except (ValueError,KeyError,TypeError): return False
    return (REGION_DISCLOSURE_DECISION in config.get('accepted_decisions',[])
            and model.get('region')=='' and evidence.get('region_disclosed') is False
            and approval.get('decision')==REGION_DISCLOSURE_DECISION
            and approval.get('actor')=='human' and approval.get('approved') is True
            and isinstance(approval.get('reviewer'),str) and bool(approval['reviewer'].strip())
            and approval.get('scope')=='fixed_provider_snapshot_variant'
            and isinstance(protocol,str) and bool(re.fullmatch('[0-9a-f]{64}',protocol))
            and approval.get('protocol_sha256')==protocol
            and approval.get('settings_sha256')==settings
            and all(isinstance(model.get(k),str) and bool(model[k].strip())
                    for k in ('provider_slug','route','version','endpoint')))


def config_errors(config):
    errors=[]
    if config.get('schema_version')!=2: errors.append('unsupported configuration version')
    decision=budget_roster_decision(config)
    current=decision is not None
    expected=list(BUDGET_ROSTERS[decision] if current else LEGACY_ROSTER)
    if config.get('model_roster')!=expected or set(config.get('models',{}))!=set(expected): errors.append('exact reviewed model roster required')
    execution=config.get('execution',{})
    if execution.get('max_attempts')!=1 or execution.get('retries') is not False or execution.get('fallbacks') is not False: errors.append('one attempt, no retries/fallbacks required')
    if execution.get('total_generated_token_allowance')!=16384: errors.append('accepted total token allowance must be 16384')
    if config.get('benchmark',{}).get('expected_evaluation_items')!=1273 or config.get('benchmark',{}).get('closed_book') is not True: errors.append('1273 MedQA closed-book items required')
    models=config.get('models',{})
    for i in range(1,21,2):
        if f'C{i:02d}' not in expected or f'C{i+1:02d}' not in expected: continue
        lo,hi=models.get(f'C{i:02d}',{}),models.get(f'C{i+1:02d}',{})
        pair_id=f'P{(i+1)//2:02d}'
        levels=EFFORT_PAIRS[pair_id] if R1_EFFORT_DECISION in config.get('accepted_decisions',[]) else ('low','high')
        if (lo.get('effort'),hi.get('effort'))!=levels or lo.get('pair_id')!=pair_id or hi.get('pair_id')!=pair_id: errors.append('incorrect effort pair identity')
        # Native effort payload is the sole request setting allowed to differ.
        omit={'display_name','effort','proposed_route'}
        a={k:v for k,v in lo.items() if k not in omit};b={k:v for k,v in hi.items() if k not in omit}
        for d in (a,b):
            controls=dict(d.get('request_settings',{})); controls.pop('reasoning',None);d['request_settings']=controls
        if a!=b: errors.append(f'C{i:02d}/C{i+1:02d}: pair controls/route/version mismatch apart from effort')
        reasoning_lo=lo.get('request_settings',{}).get('reasoning',{});reasoning_hi=hi.get('request_settings',{}).get('reasoning',{})
        if reasoning_lo or reasoning_hi:
            # Only None/High may change enabled state; no other controls differ.
            omit_reasoning={'effort','enabled'} if levels[0]=='none' else {'effort'}
            left={k:v for k,v in reasoning_lo.items() if k not in omit_reasoning};right={k:v for k,v in reasoning_hi.items() if k not in omit_reasoning}
            if left!=right or not native_effort_request_valid(lo) or not native_effort_request_valid(hi): errors.append('native pair request must match declared effort; only none may disable reasoning')
    if current:
        singles=set(expected)-{'C05','C06','C13','C14','C17','C18'}
        if any(models.get(cid,{}).get('pair_id') not in ('',None) for cid in singles): errors.append('retained single arms must be unpaired')
        if (config.get('analysis',{}).get('s1_count'),config.get('analysis',{}).get('s2_count'))!=(15,3): errors.append('budget roster requires S1=15/S2=3')
    if decision==KIMI_INSTANT_DECISION:
        instant=models.get('C25',{})
        if (instant.get('model_id')!='moonshotai/kimi-k2.5' or instant.get('base_model')!='moonshotai/kimi-k2.5'
                or instant.get('effort')!='none' or instant.get('adapter')!='openrouter'
                or instant.get('request_settings',{}).get('reasoning')!={'enabled':False}):
            errors.append('C25 Kimi Instant requires K2.5 with reasoning explicitly disabled')
        native=models.get('C23',{})
        if any(instant.get(k)!=native.get(k) for k in ('route','version','endpoint','provider_slug','provider_ignore')):
            errors.append('C25 Kimi Instant must retain the selected C23 endpoint')
    return errors

def operational_blockers(config):
    """Shared request-count and schedule prerequisites for every live run."""
    errors=config_errors(config);e=config.get('execution',{})
    if type(e.get('timeout_seconds')) not in (int,float) or not math.isfinite(float(e.get('timeout_seconds',0))) or e.get('timeout_seconds',0)<=0: errors.append('positive timeout_seconds pending')
    try:
        policy=scheduler_policy(config)
        if policy is None and e.get('concurrency')!=1: errors.append('serial concurrency=1 or reviewed versioned scheduler policy required')
        if budget_roster_decision(config) is not None and policy is None: errors.append('budget roster requires explicit scheduler policy')
    except ValueError as exc: errors.append(str(exc))
    if e.get('schedule')!='round_robin': errors.append('accepted interleaved schedule pending; supported round_robin')
    if type(e.get('seed')) is not int: errors.append('accepted scheduling seed pending')
    if not isinstance(e.get('study_registry'),str) or not e['study_registry'].strip(): errors.append('persistent study_registry required')
    return errors


def runtime_blockers(config, *, evidence_resolver=None, expected_protocol_sha256=None):
    """Observed endpoint evidence required by definitive execution and freeze."""
    errors=operational_blockers(config)
    for cid,m in config.get('models',{}).items():
        if m.get('adapter')=='jev_gateway' and (not isinstance(m.get('provider_slug'),str) or not m['provider_slug'].strip()):
            errors.append(cid+': JEV OpenRouter provider_slug pending')
        for k in ('route','version','endpoint','control_evidence','control_evidence_sha256','output_accounting'):
            if not m.get(k): errors.append(cid+': '+k+' pending')
        if type(m.get('reasoning_in_output')) is not bool and OBSERVED_COST_DECISION not in config.get('accepted_decisions',[]): errors.append(cid+': endpoint token accounting pending')
        if m.get('pair_id'):
            if not native_effort_request_valid(m): errors.append(cid+': explicit native effort forwarding pending')
        if not m.get('region') and not m.get('control_evidence'): errors.append(cid+': region pending')
        if m.get('control_evidence'):
            try:
                path=(Path(evidence_resolver(m['control_evidence'],m['control_evidence_sha256']))
                      if evidence_resolver is not None else Path(m['control_evidence']))
                evidence=strict_json(path.read_text())
                if file_hash(path)!=m['control_evidence_sha256']: raise ValueError('evidence hash mismatch')
                if not m.get('region'):
                    if (not unknown_region_accepted(config,m,evidence) or expected_protocol_sha256 is None
                            or evidence.get('protocol_sha256')!=expected_protocol_sha256):
                        raise ValueError('region pending; bound human unknown-region exception and actual protocol required')
                if evidence.get('evidence_stage') not in (None,'live_verified'):
                    raise ValueError('documentation-stage prerequisites cannot establish definitive live verification')
                # The evidence binds BOTH pair arms; acceptance never comes from config strings alone.
                if evidence.get('actor')!='human' or evidence.get('approved') is not True or evidence.get('settings_sha256')!=fingerprint({k:v for k,v in m.items() if k not in {'control_evidence','control_evidence_sha256','display_name','effort','request_settings'}}): raise ValueError('unbound/unaccepted endpoint evidence')
                required=('route_version_verified','schema_verified','no_fallback_verified','no_retry_verified','completion_verified','data_permissions_verified')
                if any(evidence.get(k) is not True for k in required) or evidence.get('total_generated_token_allowance')!=16384: raise ValueError('endpoint controls incomplete')
                limit=evidence.get('token_limit_control',{})
                if m['adapter'] in {'jev','jev_gateway'}:
                    # This branch becomes eligible only with explicit reviewed native-output evidence.
                    if m.get('request_settings') or limit!={'kind':'native_choice','generated_text_limit_applicable':False} or evidence.get('native_choice_output_verified') is not True or evidence.get('no_generated_text_verified') is not True:
                        raise ValueError('reviewed bounded native-Choice output evidence required; no decoding overrides')
                    if m['adapter']=='jev_gateway' and evidence.get('decisions_mapping_verified') is not True:
                        raise ValueError('reviewed OpenRouter Decisions label/vector/confidence mapping required')
                else:
                    if evidence.get('truncation_verified') is not True or not token_limit_request_valid(config,m,limit): raise ValueError('endpoint total generated-token request or truncation review incomplete')
                    if m.get('pair_id') and evidence.get('native_effort_verified') is not True:
                        if OBSERVED_COST_DECISION not in config.get('accepted_decisions',[]) or evidence.get('requested_effort_only_accepted') is not True: raise ValueError('native effort forwarding unverified; bound human limitation acceptance required')
                if type(m.get('reasoning_in_output')) is not bool and evidence.get('token_accounting_limitations_accepted') is not True:
                    raise ValueError('bound human acceptance of unavailable token accounting required')
                if evidence.get('request_settings_by_effort',{}).get(m['effort'])!=m.get('request_settings',{}): raise ValueError('native effort/settings not bound to review')
                if not execution_freeze_enabled(config) and m['adapter'] in {'jev','jev_gateway'} and (m.get('probability_semantics')!='native_choice_probability' or evidence.get('jev_route_and_vector_semantics_verified') is not True): raise ValueError('JEV route/vector semantics unverified')
            except (OSError,ValueError,TypeError) as exc: errors.append(cid+': '+str(exc))
    return errors

def freeze_candidate(config, artifacts):
    """Candidate inventory only; pending/dirty fields remain explicit blockers."""
    roles=EXECUTION_FREEZE_ROLES if execution_freeze_enabled(config) else FREEZE_ROLES
    if not set(artifacts).issubset(roles) or 'protocol_record' not in artifacts: raise ValueError('actual protocol_record and only preprint roles required')
    protocol=Path(artifacts['protocol_record'])
    if protocol.name!='Preprint_Protocol.md' or protocol.parent.name!='protocols' or not protocol.is_file(): raise ValueError('bind actual vault protocols/Preprint_Protocol.md')
    return {'schema_version':3 if execution_freeze_enabled(config) else 2,
        **({'freeze_scope':'execution'} if execution_freeze_enabled(config) else {}),
        'created_at':utc_now(),'status':'candidate','config_sha256':config_binding(config),
        'prompt_sha256':fingerprint(config['prompt']),'code':code_record(config=config),
        'environment':run_environment_record(config),'git':execution_git_record() if execution_freeze_enabled(config) else git_record(),
        'artifacts':{role:{'path':str(Path(p).resolve()),'sha256':file_hash(p)} for role,p in artifacts.items()},
        'missing_artifacts':sorted(roles-set(artifacts)),
        'blockers':runtime_blockers(config,expected_protocol_sha256=file_hash(protocol))+['missing freeze artifact: '+x for x in sorted(roles-set(artifacts))]}

def definitive_blockers(config=None, *, inference=False):
    # Validate current bytes plus dated human approvals, not candidate status text.
    # Inference validates solution-free questions without opening reference keys.
    # Offline analysis rehashes keys against the same accepted freeze inventory.
    config=config or load_config(); errors=[]; runtime_checked=False
    try:
        path=Path(config['freeze']['manifest']);review_path=Path(config['freeze']['human_review_record'])
        candidate=strict_json(path.read_text());review=strict_json(review_path.read_text())
        errors+=runtime_blockers(config,expected_protocol_sha256=candidate['artifacts']['protocol_record']['sha256']);runtime_checked=True
        roles,topics=freeze_requirements(candidate,config)
        if set(candidate['artifacts'])!=roles: raise ValueError('preprint freeze inventory incompatible')
        if candidate.get('missing_artifacts') or candidate.get('blockers'): errors.append('freeze candidate has unresolved blockers')
        if review.get('actor')!='human' or not review.get('reviewer') or not review.get('approved_at') or review.get('manifest_sha256')!=file_hash(path): errors.append('separate dated human freeze review required')
        if any(review.get('approvals',{}).get(t,review.get('approvals',{}).get('runtime_budget') if t=='runtime' else None) is not True for t in topics): errors.append('required preprint approvals pending')
        if candidate['config_sha256']!=config_binding(config) or candidate['prompt_sha256']!=fingerprint(config['prompt']): errors.append('config/prompt freeze mismatch')
        if candidate['code']!=code_record(config=config) or candidate['environment']!=run_environment_record(config): errors.append('code/environment freeze mismatch')
        current_git=execution_git_record() if execution_freeze_enabled(config) else git_record()
        if current_git['dirty'] or candidate['git']['dirty'] or (not execution_freeze_enabled(config) and current_git['commit']!=candidate['git']['commit']): errors.append('freeze requires clean accepted versioned code')
        if not re.fullmatch('[0-9a-f]{40,64}',candidate['git'].get('commit','')): errors.append('accepted source commit provenance required')
        for role,ref in candidate['artifacts'].items():
            p=Path(ref['path'])
            if inference and role=='keys_medqa':
                continue # Reference keys belong to offline scoring, not inference.
            elif file_hash(p)!=ref['sha256']: errors.append('freeze artifact mismatch: '+role)
        protocol=Path(candidate['artifacts']['protocol_record']['path'])
        if protocol.name!='Preprint_Protocol.md' or protocol.parent.name!='protocols': errors.append('actual vault protocol required')
        qpath=Path(candidate['artifacts']['questions_medqa']['path']);questions=load_questions(qpath)
        if len(questions)!=1273 or any(q.benchmark!='medqa' or q.split!='evaluation' for q in questions): errors.append('definitive requires complete official MedQA evaluation package')
        audit=strict_json(Path(candidate['artifacts']['audit_medqa']['path']).read_text())
        if audit.get('ready') is not True or audit.get('keys_sha256',audit.get('keys_hash'))!=candidate['artifacts']['keys_medqa']['sha256'] or audit.get('questions_sha256',audit.get('files',{}).get(str(qpath)))!=file_hash(qpath): errors.append('audit/question/key freeze mismatch')
        if not execution_freeze_enabled(config):
            method=strict_json(Path(candidate['artifacts']['method_review']['path']).read_text())
            if method.get('actor')!='human' or method.get('approved') is not True or method.get('analysis_sha256')!=fingerprint(config['analysis']): errors.append('accepted SAP method binding required')
            if method.get('schema_version')!=2 or method.get('type')!='analysis_methods' or method.get('accepted') is not True or not method.get('reviewer') or not method.get('accepted_at') or method.get('protocol_sha256')!=candidate['artifacts']['protocol_record']['sha256']: errors.append('dated protocol-bound method acceptance required')
            if type(method.get('bootstrap_seed')) is not int or method.get('bootstrap_seed',-1)<0: errors.append('accepted bootstrap seed pending')
            if method.get('normalization') not in {'preserve_native','divide_by_sum','native_selected_normalized_distribution'} or method.get('normalization')!=config['analysis']['normalization'] or type(method.get('probability_semantics_verified')) is not bool: errors.append('accepted probability normalization/semantics policy pending')
            for name in ('review_evidence','shared_stem_review'):
                evidence=method.get(name)
                if not isinstance(evidence,dict) or set(evidence)!={'path','sha256'}: errors.append('method evidence required: '+name)
                elif file_hash(Path(candidate['artifacts']['method_review']['path']).parent/evidence['path'])!=evidence['sha256']: errors.append('method evidence hash mismatch: '+name)
        route=strict_json(Path(candidate['artifacts']['route_control_evidence']['path']).read_text())
        if route.get('actor')!='human' or route.get('approved') is not True or route.get('config_sha256')!=config_binding(config): errors.append('accepted route/control inventory required')
        permissions=strict_json(Path(candidate['artifacts']['data_permissions']['path']).read_text())
        if permissions.get('actor')!='human' or permissions.get('approved') is not True or permissions.get('config_sha256')!=config_binding(config): errors.append('accepted data-use permissions required')
        lock=strict_json(Path(candidate['artifacts']['environment_lock']['path']).read_text())
        if lock!=run_environment_record(config): errors.append('environment lock mismatch')
    except (OSError,ValueError,KeyError,TypeError) as exc:
        if not runtime_checked: errors+=runtime_blockers(config)
        errors.append('human-reviewed artifact-bound freeze pending: '+str(exc))
    return errors

def validate_frozen_package(manifest, config, *, inference=False):
    """Pure producer/consumer check: accepted freeze must match this definitive package."""
    if manifest['provenance']!='definitive': raise ValueError('definitive freeze validator requires definitive provenance')
    errors=definitive_blockers(config,inference=inference)
    if errors: raise ValueError('; '.join(errors))
    candidate=strict_json(Path(config['freeze']['manifest']).read_text())
    validate_manifest_config(manifest,config,load_questions(candidate['artifacts']['questions_medqa']['path']))
    expected={'config':candidate['config_sha256'],'prompt':candidate['prompt_sha256'],'code':fingerprint(candidate['code']),
        'environment':fingerprint(candidate['environment']),'questions':candidate['artifacts']['questions_medqa']['sha256'],
        'keys':candidate['artifacts']['keys_medqa']['sha256'],'protocol_record':candidate['artifacts']['protocol_record']['sha256']}
    if any(manifest['bindings'].get(k)!=v for k,v in expected.items()): raise ValueError('attempt package does not match accepted freeze bindings')
    return candidate
