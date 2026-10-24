"""Solution-free preparation and v3 attempts with automatic lean execution provenance."""
import copy
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import subprocess
import tomllib

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "benchmarking/config.toml"
WORKFLOW = "lean-v1"
SCHEMA_VERSION = 3
DECISIONS_WORKFLOW = "lean-decisions-v1"
DECISIONS_SCHEMA_VERSION = 4
DECISIONS_MODEL = "openai/gpt-6-luna-decisions"
DECISIONS_MODEL_VERSION = "openai/gpt-6-luna-decisions-20261006"

def is_decisions(record):
    return record.get('execution_workflow', record.get('workflow')) == DECISIONS_WORKFLOW

def package_version(record):
    return DECISIONS_SCHEMA_VERSION if is_decisions(record) else SCHEMA_VERSION

def decisions_tolerance(value):
    if value in (None, ''): return None
    try:
        if isinstance(value,bool): raise ValueError()
        result=Decimal(str(value))
        if not result.is_finite() or not 0 <= result < 1: raise ValueError()
    except (ValueError,InvalidOperation,TypeError):
        raise ValueError('D01 sum tolerance must be finite and in [0,1)') from None
    return str(result)

LEGACY_ROSTER = tuple([f'C{i:02d}' for i in range(1,25)]+['J01'])
PREPRINT_ROSTER = ('C03','C05','C06','C09','C11','C13','C14','C15','C17','C18','C19','C21','C22','C23','C24','J01')
ROSTER_DECISION = 'preprint-budget-roster-20261006'
KIMI_INSTANT_DECISION = 'preprint-kimi-instant-replacement-20261006'
KIMI_INSTANT_ROSTER = tuple(cid for cid in PREPRINT_ROSTER if cid != 'C21')[:-1] + ('C25', 'J01')
BUDGET_ROSTERS = {ROSTER_DECISION: PREPRINT_ROSTER, KIMI_INSTANT_DECISION: KIMI_INSTANT_ROSTER}
def budget_roster_decision(config):
    """Keep completed Mistral packages distinct from the prospective replacement."""
    accepted = config.get('accepted_decisions', [])
    return next((d for d in (KIMI_INSTANT_DECISION, ROSTER_DECISION) if d in accepted), None)

TERMINAL_STATUSES = frozenset({'valid', 'invalid', 'refused', 'truncated', 'http_error', 'timeout', 'transport_error', 'drift'})
BINDING_NAMES = frozenset({'code', 'environment', 'config', 'prompt', 'questions', 'keys', 'protocol_record', 'execution_record'})

OPENROUTER_ENDPOINTS = frozenset({'https://openrouter.ai/api/v1/chat/completions',
    'https://eu.openrouter.ai/api/v1/chat/completions','https://us.openrouter.ai/api/v1/chat/completions'})
TYPESAFE_ENDPOINT = 'https://api.typesafe.ai/v1/systemone'
JEV_DECISIONS_ENDPOINT = 'https://openrouter.ai/api/alpha/decisions'

def validate_transport(model):
    """Apply the same adapter-specific endpoint allowlist before starts and HTTP."""
    adapter=model.get('adapter');endpoint=model.get('endpoint')
    allowed={'jev':{TYPESAFE_ENDPOINT},'jev_gateway':{JEV_DECISIONS_ENDPOINT},'openai_decisions_gateway':{JEV_DECISIONS_ENDPOINT},'openrouter':OPENROUTER_ENDPOINTS}
    if not isinstance(adapter,str) or not isinstance(endpoint,str) or endpoint not in allowed.get(adapter,()):
        raise ValueError('unapproved transport endpoint')

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

def native_probability(vector, selected_label, labels=('A','B','C','D'), concentration=None, semantics='unverified', *, sum_tolerance='0.02', decisions=False):
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
        if decisions:
            with localcontext() as context:
                context.prec=max(28,max(len(v.as_tuple().digits)+abs(v.as_tuple().exponent) for v in values.values())+10)
                total=sum(values.values());deviation=abs(total-1)
        else: total = sum(values.values()); deviation = abs(total - 1)
        result.update(sum=str(total), deviation=str(deviation))
        # Inclusive 0.02 tolerance reflects four values rounded to two decimals.
        # Passing tolerance does not imply unit sum or accepted probability semantics.
        if decisions:
            if total <= 0: result['reason']='nonpositive_sum'; return result
            if selected_label in values: result['selected_probability']=str(values[selected_label])
            tolerance=decisions_tolerance(sum_tolerance)
            if tolerance is None: result['reason']='sum_tolerance_unreviewed'; return result
        else: tolerance='0.02'
        if deviation > Decimal(tolerance): result['reason']='sum_outside_tolerance'; return result
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

def validate_provider_confidence(record):
    """Preserve a native finite scalar without imposing probability semantics."""
    if not isinstance(record,dict) or set(record)!={'value','reason'}: raise ValueError('invalid provider confidence measurement')
    value=record['value']
    if value is None:
        _text(record['reason'],'provider confidence missing reason');return
    try:
        if isinstance(value,bool) or not isinstance(value,(str,int,float)) or not Decimal(str(value)).is_finite(): raise ValueError()
    except (ValueError,InvalidOperation,TypeError): raise ValueError('nonfinite provider confidence') from None
    if record['reason'] is not None: raise ValueError('finite provider confidence cannot have missing reason')


def attempt_identity(row):
    r = row.get('identity',row)
    return r['benchmark'],r['split'],r['item_id'],r['config_id'],r['run_id']

def validate_manifest(manifest):
    if manifest.get('execution_workflow') not in (WORKFLOW,DECISIONS_WORKFLOW): raise ValueError('historical packages require their preserved readers; no rebinding')
    decisions=is_decisions(manifest)
    if manifest.get('schema_version') != package_version(manifest): raise ValueError('unsupported package schema; historical v2 requires archived v2 readers')
    if manifest.get('provenance') not in {'mock','definitive'}: raise ValueError('explicit provenance required')
    _text(manifest.get('study_id'),'study_id')
    for key in ('benchmark','split'): _text(manifest.get(key),key)
    if not BINDING_NAMES.issubset(manifest.get('bindings',{})): raise ValueError('missing code/environment/config/prompt/questions/keys/protocol_record bindings')
    for value in manifest['bindings'].values():
        if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value): raise ValueError('invalid binding digest')
    if decisions:
        if 'study_protocol' not in manifest['bindings']: raise ValueError('D01 actual protocol binding required')
        if 'probability_tolerance' not in manifest: raise ValueError('D01 precision policy required')
        if decisions_tolerance(manifest['probability_tolerance']) != manifest['probability_tolerance']:
            raise ValueError('D01 canonical tolerance required')
    items=manifest['items']; configs=manifest['configurations']; runs=manifest['runs']
    for sequence, key in ((items,'item_id'),(configs,'config_id'),(runs,'run_id')):
        if not sequence or len({r[key] for r in sequence}) != len(sequence): raise ValueError('empty/duplicate manifest identities')
    for item in items:
        if set(item)!= {'item_id','content_sha256','labels'} or set(item['labels'])!=set('ABCD') or len(item['labels'])!=4 or not re.fullmatch('[0-9a-f]{64}',item['content_sha256']): raise ValueError('invalid manifest question')
    if 'schedule' in manifest:
        order=manifest['schedule'].get('item_order')
        if order is not None and (len(order)!=len(items) or set(order)!={x['item_id'] for x in items}): raise ValueError('scheduling item order mismatch')
    ids={r['config_id'] for r in configs}
    if decisions and (ids!={'D01'} or any(c.get('model_id')!=DECISIONS_MODEL or c.get('adapter')!='openai_decisions_gateway'
            or c.get('effort')!='native' or c.get('pair_id') is not None for c in configs)):
        raise ValueError('D01-only native Luna Decisions package required')
    for c in configs:
        for k in ('config_id','model_id','base_model','effort','adapter'): _text(c.get(k),k)
        if 'pair_id' not in c: raise ValueError('pair_id required (null for singles)')
    for r in runs:
        if r['config_id'] not in ids or type(r['replicate']) is not int or r['replicate']<1: raise ValueError('invalid run configuration/replicate')
        if r['role'] not in {'primary','secondary'} or r['experiment_id'] not in {'E1','E2'}: raise ValueError('invalid run role/experiment')
        if r['experiment_id']=='E2' and (r['config_id']!=('D01' if decisions else 'J01') or r['replicate'] not in (2,3) or r['role']!='secondary'): raise ValueError('E2 must be secondary JEV runs2/3')
        if r['experiment_id']=='E1' and (r['role']!='primary' or r['replicate']!=1): raise ValueError('E1 must be primary run1')
        if r['role']=='primary' and r['replicate']!=1: raise ValueError('primary must use run1')
    if len({(r['config_id'],r['replicate']) for r in runs})!=len(runs): raise ValueError('duplicate run replicate')
    if manifest['provenance']=='definitive':
        if len(items)!=1273 or ids not in (({'D01'},) if decisions else (set(LEGACY_ROSTER), *(set(r) for r in BUDGET_ROSTERS.values()))): raise ValueError('definitive requires 1273 items and exact reviewed roster')
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
    if 'E1' in experiments:
        runs += [{'run_id':f'{study_id}-E1-{c["config_id"]}-r1','experiment_id':'E1',
                  'config_id':c['config_id'],'replicate':1,'role':'primary'} for c in configurations]
    if 'E2' in experiments:
        repeat='D01' if {c['config_id'] for c in configurations}=={'D01'} else 'J01'
        runs += [{'run_id':f'{study_id}-E2-{repeat}-r{n}','experiment_id':'E2','config_id':repeat,
                  'replicate':n,'role':'secondary'} for n in (2,3)]
    return runs

def validate_schedule_policy(policy):
    """Exact, versioned scheduler settings; no unknown runtime knobs."""
    required={'version','concurrency','provider_concurrency','provider_min_interval_seconds','rate_limit_stop_after'}
    if not isinstance(policy,dict) or type(policy.get('version')) is not int or policy['version'] not in (1,2,3): raise ValueError('unsupported scheduler policy')
    if policy['version'] in (2,3): required.add('provider_concurrency_overrides')
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
    if execution['scheduler_version'] in (2,3):
        policy['provider_concurrency_overrides']=execution.get('provider_concurrency_overrides',{})
    elif execution.get('provider_concurrency_overrides'):
        raise ValueError('provider concurrency overrides require scheduler version 2 or 3')
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
    if manifest.get('execution_workflow')!=config.get('execution_workflow',WORKFLOW): raise ValueError('workflow/configuration mismatch')
    if is_decisions(manifest) and manifest['probability_tolerance']!=decisions_tolerance(config['models']['D01'].get('sum_tolerance')): raise ValueError('D01 tolerance/configuration mismatch')
    if manifest['configurations']!=manifest_configurations(config): raise ValueError('manifest configuration metadata mismatch')
    experiments={r['experiment_id'] for r in manifest['runs']}
    if experiments not in ({'E1'},{'E2'},{'E1','E2'}): raise ValueError('unsupported experiment combination')
    if manifest['runs']!=manifest_runs(manifest['study_id'],manifest['configurations'],experiments): raise ValueError('manifest run metadata mismatch')
    expected=[{'item_id':q.item_id,'content_sha256':question_hash(q),'labels':list(q.labels)} for q in questions]
    if manifest['items']!=expected: raise ValueError('manifest preparation order/content mismatch')
    if any(q.benchmark!=manifest['benchmark'] or q.split!=manifest['split'] for q in questions): raise ValueError('manifest benchmark/split mismatch')
    if manifest['split']!='evaluation': raise ValueError('manifest experiment/split mismatch')
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
    context={k:v for k,v in manifest.items() if k not in ('runs','continuation')}
    if 'continuation' in manifest:
        context['bindings']=manifest['continuation']['root_bindings']
        context['schedule']=manifest['continuation'].get('root_schedule',manifest['schedule'])
    return context

def scientific_context(manifest):
    """Preserved context for exact continuation or the explicitly bounded amendment."""
    context={k:v for k,v in manifest.items() if k!='continuation'}
    context['bindings']={k:v for k,v in manifest['bindings'].items()
                         if k not in ('code','environment','protocol_record','execution_record')}
    return context

CAPACITY_AMENDMENT = 'inference-net-capacity-20261007'
DEEPSEEK_TEN_AMENDMENT = 'inference-net-concurrency-10-20261007'


def capacity_amended_config(config):
    """The accepted amendment changes pacing and provider-stop scope only."""
    amended=copy.deepcopy(config);execution=amended['execution']
    if execution.get('scheduler_version')!=2:
        raise ValueError('capacity amendment requires scheduler version 2 parent')
    execution['scheduler_version']=3
    execution['provider_concurrency_overrides']['InferenceNet']=1
    execution['provider_min_interval_seconds']['InferenceNet']=10.0
    return amended


def deepseek_ten_amended_config(config):
    """Apply ten shared DeepSeek slots and five-second start spacing."""
    amended=copy.deepcopy(config);execution=amended['execution']
    if execution.get('scheduler_version')!=3 or execution.get('provider_concurrency_overrides',{}).get('InferenceNet')!=1:
        raise ValueError('ten-slot amendment requires scheduler version 3 and InferenceNet limit 1 parent')
    if execution.get('concurrency',0)<10:
        raise ValueError('ten-slot amendment requires global concurrency at least 10')
    if execution.get('provider_min_interval_seconds',{}).get('InferenceNet')!=10.0:
        raise ValueError('ten-slot amendment requires ten-second InferenceNet spacing parent')
    execution['provider_concurrency_overrides']['InferenceNet']=10
    execution['provider_min_interval_seconds']['InferenceNet']=5.0
    return amended


def validate_continuation_settings(prior, manifest, amendment, parent):
    if amendment is None:
        if scientific_context(prior)!=scientific_context(manifest):
            raise ValueError('continuation scientific settings changed')
        return
    if amendment not in (CAPACITY_AMENDMENT,DEEPSEEK_TEN_AMENDMENT):
        raise ValueError('unsupported execution amendment')
    # Validate the full saved configuration, including transport, payloads and
    # timeout. Removing a config digest from context alone would permit drift.
    config=strict_json((Path(parent)/'source/config.json').read_text())
    amended=(capacity_amended_config if amendment==CAPACITY_AMENDMENT else deepseek_ten_amended_config)(config)
    if manifest['bindings']['config']!=config_binding(amended):
        raise ValueError('capacity amendment changed unapproved configuration settings')
    expected=copy.deepcopy(scientific_context(prior))
    expected['bindings']['config']=config_binding(amended)
    expected['schedule']['policy']=scheduler_policy(amended)
    if expected!=scientific_context(manifest):
        raise ValueError('capacity amendment changed unapproved scientific settings')


def stop_events(package):
    """Exact provider-stop or reviewed interruption events requiring continuation review."""
    return [event['event_sha256'] for event in read_jsonl(Path(package)/'events.jsonl')
            if event.get('event')=='finish' and
            (event['record'].get('scheduler',{}).get('stop_reason')=='provider_rate_limit_stop'
             or event['record'].get('recovery_review') is not None)]

def continuation_parent(package, manifest, *, seen=None):
    link=manifest['continuation']
    fields={'parent','manifest_sha256','provenance_sha256','events_sha256','last_event_sha256',
            'reviewed_stop_events','reviewer','reviewed_at','root_bindings'}
    if set(link) not in (fields,fields|{'root_schedule'},fields|{'root_schedule','execution_amendment'}):
        raise ValueError('invalid continuation metadata')
    _text(link['parent'],'continuation parent');_text(link['reviewer'],'reviewer')
    if Path(link['parent']).is_absolute(): raise ValueError('continuation parent must be a relative package path')
    try:
        if datetime.fromisoformat(link['reviewed_at']).utcoffset() is None: raise ValueError()
    except (TypeError,ValueError): raise ValueError('timezone-aware review timestamp required') from None
    parent=(Path(package)/link['parent']).resolve()
    prior,records,coverage=read_package(parent,_seen=seen)
    validate_continuation_settings(prior,manifest,link.get('execution_amendment'),parent)
    root_schedule=prior.get('continuation',{}).get('root_schedule',prior['schedule'])
    if link.get('root_schedule',manifest['schedule'])!=root_schedule:
        raise ValueError('continuation original schedule changed')
    if fingerprint(prior)!=link['manifest_sha256'] or file_hash(parent/'provenance.json')!=link['provenance_sha256'] or file_hash(parent/'events.jsonl')!=link['events_sha256'] or coverage['last_event_sha256']!=link['last_event_sha256']:
        raise ValueError('continuation parent evidence changed')
    if link['root_bindings']!=prior.get('continuation',{}).get('root_bindings',prior['bindings']):
        raise ValueError('continuation original bindings changed')
    stops=stop_events(parent)
    # The named ten-slot amendment also permits an operator-drained parent with
    # no provider-stop event. Its exact journal is still bound and reviewed.
    if (not stops and link.get('execution_amendment')!=DEEPSEEK_TEN_AMENDMENT) or link['reviewed_stop_events']!=stops:
        raise ValueError('exact parent provider-stop review required')
    if coverage['indeterminate'] or any(r['status']=='drift' for r in records):
        raise ValueError('unresolved parent starts/drift prohibit continuation')
    if not coverage['unstarted']: raise ValueError('parent has no unstarted identities')
    return prior,records,coverage

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
    decisions=is_decisions(manifest)
    if decisions:
        validate_provider_confidence(record.get('provider_confidence'))
        if p.get('concentration') is not None or p.get('concentration_reason')!='provider_unavailable': raise ValueError('D01 confidence must remain separate from JEV concentration')
    if set(p)!=set(empty_probability()): raise ValueError('probability fields mismatch')
    if p['native_vector'] is not None:
        checked=native_probability(p['native_vector'], record['selected_label'], labels,p['concentration'],p['semantics'],
            sum_tolerance=manifest.get('probability_tolerance'),decisions=True) if decisions else native_probability(p['native_vector'], record['selected_label'], labels,p['concentration'],p['semantics'])
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
    if 'recovery_review' in record:
        review=record['recovery_review']
        if not isinstance(review,dict) or set(review)!={'reviewer','reviewed_at','exception','partial_bytes','source_sha256','timing_basis'}:
            raise ValueError('invalid interruption review')
        _text(review['reviewer'],'interruption reviewer')
        if not isinstance(review['source_sha256'],str) or not re.fullmatch('[0-9a-f]{64}',review['source_sha256']):
            raise ValueError('invalid recovery source hash')
        if (record['status']!='transport_error' or record['failure_reason']!='IncompleteRead'
                or review['exception']!='http.client.IncompleteRead'
                or type(review['partial_bytes']) is not int or review['partial_bytes']<0
                or review['timing_basis']!='review_timestamp_not_response_completion'
                or record['timing']['finished_at']!=review['reviewed_at']
                or record['timing']['duration_seconds']['value'] is not None
                or record['artifacts']['raw_response'] is not None):
            raise ValueError('invalid reviewed interruption outcome')
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

def read_package(package, *, include_parents=True, _seen=None):
    """Return manifest, terminal records, coverage. Verify journal chain/raw hashes."""
    package=Path(package).resolve()
    raw_manifest = strict_json((package/'manifest.json').read_text())
    if raw_manifest.get('execution_workflow') == 'decision-extension-v1':
        from benchmarking.extension_contract import read_extension_package
        return read_extension_package(package)
    seen=set() if _seen is None else set(_seen)
    if package in seen: raise ValueError('continuation parent cycle')
    seen.add(package)
    manifest=validate_manifest(strict_json((package/'manifest.json').read_text()))
    verify_run_inventory(package, manifest)
    inherited=[];parent_segments=[]
    if 'continuation' in manifest:
        _,inherited,parent_coverage=continuation_parent(package,manifest,seen=seen)
        parent_segments=parent_coverage['segments']
    expected=set(expected_identities(manifest)); inherited_ids={attempt_identity(r) for r in inherited}
    item_labels={x['item_id']:x['labels'] for x in manifest['items']}; starts={}; finishes={}; previous=None
    plan_seen=False;manifest_sha256=fingerprint(manifest)
    primary_runs={r['run_id'] for r in manifest['runs'] if r['role']=='primary'}
    primary_ids={i for i in expected if i[-1] in primary_runs}
    # Plan -> unique start -> unique finish is the only accepted progression.
    # A start without finish is indeterminate; no start is unstarted. Neither is
    # a completed failure, even though a started request may already be charged.
    for event in read_jsonl(package/'events.jsonl'):
        digest=event.pop('event_sha256',None)
        if event.get('previous_sha256')!=previous or fingerprint(event)!=digest: raise ValueError('journal chain/hash mismatch')
        previous=digest
        if event.get('schema_version')!=manifest['schema_version'] or event.get('bindings')!=manifest['bindings']: raise ValueError('event binding mismatch')
        if event.get('manifest_sha256')!=manifest_sha256: raise ValueError('event manifest binding mismatch')
        if event['event']=='plan':
            if plan_seen or starts or finishes: raise ValueError('unique initial plan event required')
            plan_seen=True;continue
        if not plan_seen: raise ValueError('initial manifest-bound plan event required')
        ident=attempt_identity(event)
        if ident not in expected: raise ValueError('unexpected journal identity')
        if ident in inherited_ids: raise ValueError('continuation resends a previously started identity')
        if event['event']=='start':
            if ident in starts: raise ValueError('duplicate start')
            if ident[-1] not in primary_runs and not primary_ids.issubset(inherited_ids|set(finishes)):
                raise ValueError('E2 start before complete E1 terminal coverage')
            check_artifact(event['request'],package); starts[ident]=event
        elif event['event']=='finish':
            if ident not in starts or ident in finishes: raise ValueError('finish without unique start')
            record=validate_finish(event['record'],manifest,allowed_ids=expected,item_labels=item_labels)
            if attempt_identity(record)!=ident or record['artifacts']['request']!=starts[ident]['request']: raise ValueError('start/finish identity or request mismatch')
            for ref in record['artifacts'].values(): check_artifact(ref,package)
            finishes[ident]=record
        else: raise ValueError('unknown journal event')
    if not plan_seen: raise ValueError('initial manifest-bound plan event required')
    records=(inherited if include_parents else [])+list(finishes.values())
    return manifest,records,{'expected':len(expected),'terminal':len(inherited)+len(finishes),
        'inherited_terminal':len(inherited),'segment_terminal':len(finishes),
        'segments':parent_segments+[{'manifest_sha256':manifest_sha256,'events_sha256':file_hash(package/'events.jsonl'),
                                    'provenance_sha256':file_hash(package/'provenance.json'),'bindings':manifest['bindings']}],
        'unstarted':[list(x) for x in sorted(expected-inherited_ids-set(starts))],
        'indeterminate':[list(x) for x in sorted(set(starts)-set(finishes))],
        'complete':inherited_ids|set(finishes)==expected,'last_event_sha256':previous}

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


def config_errors(config):
    errors=[]
    if config.get('schema_version')!=2: errors.append('unsupported configuration version')
    if is_decisions(config):
        if config.get('model_roster')!=['D01'] or set(config.get('models',{}))!={'D01'}: errors.append('D01-only configuration required')
        model=config.get('models',{}).get('D01',{})
        if any(model.get(k)!=v for k,v in {'model_id':DECISIONS_MODEL,'base_model':DECISIONS_MODEL,
                'adapter':'openai_decisions_gateway','effort':'native','route':'OpenAI','provider_slug':'openai','pair_id':''}.items()):
            errors.append('exact native Luna Decisions identity and OpenAI route required')
        if model.get('version') not in {DECISIONS_MODEL, DECISIONS_MODEL_VERSION}:
            errors.append('exact reviewed Luna Decisions response version required')
        if model.get('request_settings'): errors.append('D01 generative controls prohibited')
        try: decisions_tolerance(model.get('sum_tolerance'))
        except ValueError as exc: errors.append(str(exc))
        e=config.get('execution',{})
        if e.get('max_attempts')!=1 or e.get('retries') is not False or e.get('fallbacks') is not False: errors.append('one attempt, no retries/fallbacks required')
        if config.get('benchmark',{}).get('expected_evaluation_items')!=1273 or config.get('benchmark',{}).get('closed_book') is not True: errors.append('1273 MedQA closed-book items required')
        if any(key in e for key in ('total_generated_token_allowance',)): errors.append('D01 generated-token controls prohibited')
        return errors
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



EXECUTION_SOURCES = (
    'preparation/__init__.py', 'preparation/prepare_medqa.py', 'preparation/subset.py',
    'benchmarking/__init__.py', 'benchmarking/common.py', 'benchmarking/run.py',
    'benchmarking/adapters.py', 'benchmarking/study.py')
CLEAN_REPOSITORY_FILES = EXECUTION_SOURCES + ('benchmarking/config.toml', 'README.md', '.gitignore')

def load_config(path=CONFIG_PATH):
    with Path(path).open('rb') as stream: return tomllib.load(stream)

def config_binding(config):
    return fingerprint(config)

def code_record(root=None, *, config=None):
    root=ROOT if root is None else root
    return {name:file_hash(Path(root)/name) for name in sorted(EXECUTION_SOURCES)}

def run_environment_record(config=None):
    return {'python':platform.python_version(), 'implementation':platform.python_implementation(),
            'platform':platform.platform(), 'packages':{}}

def execution_git_record(config_path=CONFIG_PATH):
    try:
        commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
        try: selected=str(Path(config_path).resolve().relative_to(ROOT))
        except ValueError: return {'commit':commit,'dirty':True}
        status=subprocess.run(['git','status','--porcelain','--',*CLEAN_REPOSITORY_FILES,selected],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
        return {'commit':commit, 'dirty':bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {'commit':None, 'dirty':None}

def evaluation_blockers(config):
    if not is_decisions(config): return []
    model=config.get('models',{}).get('D01',{})
    errors=[]
    try:
        if decisions_tolerance(model.get('sum_tolerance')) is None: errors.append('D01 sum_tolerance requires reviewed development precision')
    except ValueError as exc: errors.append(str(exc))
    if not isinstance(model.get('precision_evidence'),str) or not model['precision_evidence'].strip(): errors.append('D01 precision_evidence required before evaluation')
    return errors


def runtime_blockers(config):
    """Check executable controls, without private evidence or approval files."""
    errors=operational_blockers(config)
    if config.get('execution_workflow') not in (WORKFLOW,DECISIONS_WORKFLOW): errors.append('supported execution configuration required')
    for cid, model in config.get('models',{}).items():
        try: validate_transport(model)
        except ValueError as exc: errors.append(cid+': '+str(exc))
        for field in ('route','version','endpoint','provider_slug','output_accounting'):
            if not isinstance(model.get(field),str) or not model[field].strip(): errors.append(cid+': '+field+' required')
        if model.get('adapter') == 'openrouter':
            settings=model.get('request_settings',{})
            if settings.get('max_completion_tokens') != 16384 or 'max_tokens' in settings:
                errors.append(cid+': requested combined-token target must be 16384')
            if model.get('effort')=='none':
                if settings.get('reasoning',{}).get('enabled') is not False: errors.append(cid+': disabled reasoning required')
            elif model.get('effort') not in ('standard','native') and not native_effort_request_valid(model):
                errors.append(cid+': native requested effort mismatch')
        elif model.get('adapter') not in ('jev_gateway','openai_decisions_gateway') or model.get('request_settings'):
            errors.append(cid+': selected native Decisions adapter without decoding overrides required')
    return errors

def run_inventory(config, questions_path, input_manifest_path, *, study_id=None, config_path=CONFIG_PATH, resume_output=None, continuation=False, protocol_path=None):
    current = {'type':'execution_provenance', 'workflow':config.get('execution_workflow',WORKFLOW), 'code':code_record(),
            'environment':run_environment_record(), 'git':execution_git_record(config_path),
            'config':config, 'config_source_sha256':file_hash(config_path), 'readme_sha256':file_hash(ROOT/'README.md'),
            'questions_sha256':file_hash(questions_path),
            'input_manifest_sha256':file_hash(input_manifest_path)}
    if is_decisions(config):
        if protocol_path is None: raise ValueError('D01 requires the actual vault protocol via --protocol')
        current['study_protocol_sha256']=file_hash(protocol_path)
        current['study_protocol_text']=Path(protocol_path).read_bytes().decode('utf-8')
    # Resume and separately collected E2 retain the original commit provenance.
    # Only matching source/configuration/input/environment bytes may continue.
    if resume_output is not None and (Path(resume_output)/'manifest.json').exists():
        saved,_,_=read_package(resume_output)
        original=verify_run_inventory(resume_output,saved)
        without_git=lambda record:{k:v for k,v in record.items() if k!='git'}
        if without_git(original)!=without_git(current): raise ValueError('study source/configuration/input/environment bindings changed')
        return original
    if continuation: return current
    if study_id is not None:
        registry=Path(config['execution']['study_registry'])
        if not registry.is_absolute(): registry=ROOT/registry
        directory=registry/fingerprint(study_id)
        header=directory/'study.json'
        if header.exists():
            study=strict_json(header.read_text())['study']
            if study.get('execution_workflow')!=config.get('execution_workflow',WORKFLOW): raise ValueError('historical study cannot be rebound')
            entries=read_jsonl(directory/'packages.jsonl')
            if not entries: raise ValueError('registered study has no recoverable package; preserve and review')
            original=strict_json((Path(entries[0]['package'])/'provenance.json').read_text())
            if fingerprint(original)!=study['bindings']['execution_record']: raise ValueError('registered provenance changed')
            without_git=lambda record:{k:v for k,v in record.items() if k!='git'}
            if without_git(original)!=without_git(current): raise ValueError('study source/configuration/input/environment bindings changed')
            return original
    return current

def save_run_inventory(output, inventory, questions_path, input_manifest_path, *, config_path=CONFIG_PATH):
    """Keep exact source/config/input bytes alongside hashes, with no keys or secrets."""
    output=Path(output)
    for name, expected in inventory['code'].items():
        data=(ROOT/name).read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected: raise ValueError('source changed before run: '+name)
        target=output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream: stream.write(data)
    for name, source, expected in (
        ('benchmarking/config.toml',Path(config_path),inventory['config_source_sha256']),
        ('README.md',ROOT/'README.md',inventory['readme_sha256']),
        ('questions.jsonl',Path(questions_path),inventory['questions_sha256']),
        ('inference-manifest.json',Path(input_manifest_path),inventory['input_manifest_sha256'])):
        data=source.read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected: raise ValueError('input/README changed before run: '+name)
        target=output/'source'/name
        with target.open('xb') as stream: stream.write(data)
    if is_decisions(inventory):
        data=inventory['study_protocol_text'].encode('utf-8')
        if hashlib.sha256(data).hexdigest()!=inventory['study_protocol_sha256']: raise ValueError('protocol snapshot hash mismatch')
        with (output/'source/study-protocol.md').open('xb') as stream: stream.write(data)
    write_new(output/'source/config.json',inventory['config'])
    write_new(output/'provenance.json',inventory)

def verify_run_inventory(output, manifest):
    """Verify saved provenance, never require the original checkout or private vault."""
    output=Path(output)
    inventory=strict_json((output/'provenance.json').read_text())
    if inventory.get('type')!='execution_provenance' or inventory.get('workflow')!=manifest['execution_workflow']:
        raise ValueError('automatic execution provenance missing/unsupported')
    if fingerprint(inventory)!=manifest['bindings'].get('execution_record'): raise ValueError('execution provenance binding mismatch')
    if set(inventory['code'])!=set(EXECUTION_SOURCES): raise ValueError('execution source inventory incomplete')
    for name, expected in inventory['code'].items():
        if file_hash(output/'source'/name)!=expected: raise ValueError('saved execution source mismatch: '+name)
    config=strict_json((output/'source/config.json').read_text())
    if config!=inventory['config'] or config_binding(config)!=manifest['bindings']['config']:
        raise ValueError('saved configuration mismatch')
    errors=runtime_blockers(config)
    if errors: raise ValueError('; '.join(errors))
    expected={'code':fingerprint(inventory['code']), 'environment':fingerprint(inventory['environment']),
              'prompt':fingerprint(config['prompt']), 'questions':inventory['questions_sha256'],
              'protocol_record':inventory['readme_sha256']}
    if is_decisions(manifest):
        expected['study_protocol']=inventory['study_protocol_sha256']
        if file_hash(output/'source/study-protocol.md')!=inventory['study_protocol_sha256'] or (output/'source/study-protocol.md').read_bytes().decode('utf-8')!=inventory['study_protocol_text']:
            raise ValueError('saved actual study protocol mismatch')
    if any(manifest['bindings'][name]!=value for name,value in expected.items()): raise ValueError('saved execution bindings mismatch')
    for name,expected in (('benchmarking/config.toml',inventory['config_source_sha256']),('README.md',inventory['readme_sha256']),('questions.jsonl',inventory['questions_sha256']),
                          ('inference-manifest.json',inventory['input_manifest_sha256'])):
        if file_hash(output/'source'/name)!=expected: raise ValueError('saved input/README mismatch: '+name)
    public=strict_json((output/'source/inference-manifest.json').read_text())
    if public['keys_sha256']!=manifest['bindings']['keys'] or public['preparation_manifest_sha256']!=manifest['key_manifest']['preparation_sha256']:
        raise ValueError('saved public handoff/key digest mismatch')
    validate_manifest_config(manifest,config,load_questions(output/'source/questions.jsonl'))
    return inventory
