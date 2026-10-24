"""Pure schema-6 collection reader and attempt state machine. No network imports."""
from datetime import datetime
from pathlib import Path
import random
import re
import tomllib
from benchmarking.common import (ROOT, check_artifact, empty_probability, file_hash,
    fingerprint, load_questions, measurement, native_probability, question_hash,
    read_jsonl, strict_json, validate_measurement, validate_provider_confidence,
    TERMINAL_STATUSES)
from benchmarking.configuration_ids import (CURRENT_NAMESPACE, EXTENSION_IDS,
    canonical_configuration, configuration_mapping)

WORKFLOW = 'decision-extension-v1'
SCHEMA = 6
SOURCES = ('benchmarking/__init__.py', 'benchmarking/common.py', 'benchmarking/adapters.py',
           'benchmarking/run.py', 'benchmarking/study.py', 'benchmarking/configuration_ids.py',
           'benchmarking/native_decisions.py', 'benchmarking/extension_contract.py',
           'benchmarking/extension.py')


def request_payload(q, model, prompt):
    return {'model': model['model_id'],
            'state': prompt.format(stem=q.stem, options='\n'.join(o.label+': '+o.text for o in q.options)),
            'questions': {'answer': {'type': 'choice',
                'instructions': 'Select exactly one supplied option label.',
                'criteria': {o.label: o.text for o in q.options}}},
            'provider': {'order': [model['provider_slug']], 'only': [model['provider_slug']],
                         'allow_fallbacks': False, 'require_parameters': True}}


def config_errors(config):
    errors = []
    if (config.get('schema_version') != SCHEMA or config.get('execution_workflow') != WORKFLOW
            or config.get('config_namespace') != CURRENT_NAMESPACE):
        errors.append('unsupported extension configuration')
    if tuple(config.get('model_roster', [])) != EXTENSION_IDS or set(config.get('models', {})) != set(EXTENSION_IDS):
        errors.append('exact D03-D13 roster required')
    expected = {'concurrency': 10, 'provider_concurrency': 10, 'timeout_seconds': 600,
                'seed': 20261006, 'max_retries': 10, 'rate_limit_stop_after': 3}
    for key, value in expected.items():
        if type(config.get('execution', {}).get(key)) is not int or config['execution'][key] != value:
            errors.append('accepted runtime required: '+key)
    interval = config.get('execution', {}).get('min_interval_seconds')
    if type(interval) not in (int, float) or interval != 5:
        errors.append('accepted five-second per-provider start spacing required')
    for cid, model in config.get('models', {}).items():
        try:
            canonical_configuration(CURRENT_NAMESPACE, cid, model_id=model['model_id'], effort=model['effort'])
        except (KeyError, ValueError) as exc:
            errors.append(cid+': '+str(exc))
        semantics = model.get('probability_semantics', {})
        if (set(semantics) != {'reviewed','meaning','evidence'} or type(semantics.get('reviewed')) is not bool
            or not isinstance(semantics.get('meaning'), str) or not semantics.get('meaning')
            or not isinstance(semantics.get('evidence'), str)
            or semantics.get('reviewed') and not semantics.get('evidence', '').strip()):
            errors.append(cid+': explicit probability field-review evidence required')
        if model.get('request_settings') or model.get('adapter') != 'native_decision_gateway':
            errors.append(cid+': native adapter without decoding controls required')
    return errors


def configurations(config, ids):
    return [{**configuration_mapping(CURRENT_NAMESPACE, cid),
             'base_model': config['models'][cid]['base_model'], 'pair_id': None,
             'adapter': 'native_decision_gateway',
             'probability_semantics': config['models'][cid]['probability_semantics'],
             'scalar_semantics': config['models'][cid]['scalar_semantics']}
            for cid in ids]


def runs(study, ids, development=False):
    return [{'run_id': f'{study}-'+('E1' if n == 1 else 'E2')+f'-{cid}-r{n}',
             'config_id': cid, 'replicate': n, 'role': 'primary' if n == 1 else 'secondary',
             'experiment_id': 'E1' if n == 1 else 'E2'}
            for n in ((1,) if development else (1, 2, 3)) for cid in ids]


def targets(manifest):
    """Seeded question order with rotating model order; independent of correctness."""
    out = {}
    for n in sorted({r['replicate'] for r in manifest['runs']}):
        group = [r for r in manifest['runs'] if r['replicate'] == n]
        for position, item in enumerate(manifest['schedule']['item_order']):
            offset = position % len(group)
            for run in group[offset:]+group[:offset]:
                ident = {'benchmark': manifest['benchmark'], 'split': manifest['split'],
                         'item_id': item, 'config_id': run['config_id'], 'run_id': run['run_id']}
                tid = fingerprint([CURRENT_NAMESPACE, manifest['benchmark'], manifest['split'],
                                   item, run['config_id'], n])
                out[tid] = {'identity': ident, **{k: run[k] for k in ('replicate','role','experiment_id')}}
    return out


def attempt_id(tid, index):
    return fingerprint([tid, index])


class Ledger:
    """Replayable single-active identity, first-valid and persisted-cap invariants."""
    def __init__(self, manifest):
        self.manifest = manifest
        self.targets = targets(manifest)
        self.counts = {}
        self.active = {}
        self.records = []
        self.last = {}
        self.selected = {}
        self.paused = {}
        self.streaks = {}
        self.drift = set()
        self.previous = None
        self.plan_seen = False
        self.starts = {}
        self.max_index = 0 if manifest['purpose'] == 'development' else 10

    def resolved(self, tid):
        return tid in self.selected

    def candidates(self):
        pending = [t for t in self.targets if not self.resolved(t)]
        primary = [t for t in pending if self.targets[t]['role'] == 'primary']
        pool = primary or pending
        # Complete the initial round before recovery; active calls hold the barrier.
        initial = [t for t in pool if self.counts.get(t, 0) == 0 or
                   t in self.active and self.active[t] == 0]
        pool = initial or pool
        eligible = [t for t in pool if self.provider(t) not in self.paused
                    and self.provider(t) not in self.drift]
        if not eligible:
            return []
        round_index = min(self.counts.get(t, 0) - (t in self.active) for t in eligible)
        return [t for t in eligible if t not in self.active and self.counts.get(t, 0) == round_index]

    def provider(self, tid):
        cid = self.targets[tid]['identity']['config_id']
        return self.manifest['models'][cid]['route']

    def apply(self, event, package=None):
        row = dict(event)
        digest = row.pop('event_sha256', None)
        if (row.get('previous_sha256') != self.previous or fingerprint(row) != digest or
            row.get('schema_version') != SCHEMA or row.get('manifest_sha256') != fingerprint(self.manifest)):
            raise ValueError('extension journal chain/binding mismatch')
        kind = row['event']
        if kind == 'plan':
            if self.plan_seen or self.previous is not None:
                raise ValueError('duplicate/noninitial plan')
            self.plan_seen = True
        elif not self.plan_seen:
            raise ValueError('initial plan required')
        elif kind == 'review':
            wanted = sorted(h for hashes in self.paused.values() for h in hashes)
            if (not wanted or row.get('reviewed_stop_events') != wanted or
                not row.get('reviewer', '').strip() or self.active):
                raise ValueError('exact drained provider-pause review required')
            for provider in self.paused:
                self.streaks[provider] = 0
            self.paused.clear()
        elif kind in ('start', 'finish'):
            tid, index = row['original_query_id'], row['attempt_index']
            if tid not in self.targets or type(index) is not int or not 0 <= index <= self.max_index:
                raise ValueError('unexpected target/index')
            target = self.targets[tid]
            if row.get('attempt_id') != attempt_id(tid, index):
                raise ValueError('attempt ID mismatch')
            parent = attempt_id(tid, index-1) if index else None
            if row.get('parent_attempt_id') != parent:
                raise ValueError('attempt parent mismatch')
            if kind == 'start':
                if (tid in self.active or tid in self.selected or index != self.counts.get(tid, 0)
                    or tid not in self.candidates() or len(self.active) >= 10):
                    raise ValueError('duplicate, invalid-phase, paused or exhausted start')
                if sum(self.provider(t) == self.provider(tid) for t in self.active) >= 10:
                    raise ValueError('provider concurrency exceeded')
                if row['provider'] != self.provider(tid):
                    raise ValueError('provider mismatch')
                if index and row['request']['sha256'] != self.starts[(tid, 0)]['request']['sha256']:
                    raise ValueError('retry request changed')
                if package:
                    check_artifact(row['request'], package)
                self.starts[tid, index] = row
                self.counts[tid] = index+1
                self.active[tid] = index
            else:
                if self.active.get(tid) != index:
                    raise ValueError('finish without unique active start')
                record = row['record']
                if (record['identity'] != target['identity'] or
                    any(record.get(k) != row[k] for k in ('original_query_id','attempt_id','attempt_index','parent_attempt_id')) or
                    record['artifacts']['request'] != self.starts[tid,index]['request']):
                    raise ValueError('finish identity/lineage mismatch')
                validate_record(record, self.manifest, target)
                if package:
                    for ref in record['artifacts'].values():
                        check_artifact(ref, package)
                provider = self.provider(tid)
                status = record['scheduler']['response_status']
                self.streaks[provider] = self.streaks.get(provider, 0)+1 if status == 429 else 0
                stop = 'provider_rate_limit_stop' if self.streaks[provider] >= 3 else None
                if record['scheduler']['stop_reason'] != stop:
                    raise ValueError('pause threshold mismatch')
                if stop:
                    self.paused.setdefault(provider, []).append(digest)
                if record['status'] == 'drift':
                    self.drift.add(provider)
                self.records.append(record)
                self.last[tid] = record
                del self.active[tid]
                if record['status'] == 'valid' or index == self.max_index:
                    self.selected[tid] = record
        else:
            raise ValueError('unsupported extension event')
        self.previous = digest

    def coverage(self):
        development = self.manifest['purpose'] == 'development'
        passed = len(self.selected) == len(self.targets) and all(
            r['status'] == 'valid' and r['probability']['valid'] for r in self.selected.values())
        return {**({'development_passed': passed} if development else {}),'expected': len(self.targets), 'terminal': len(self.selected),
                'attempt_starts': sum(self.counts.values()), 'attempt_terminal': len(self.records),
                'unstarted': [t for t in self.targets if t not in self.counts],
                'indeterminate': list(self.active),
                'pending': [t for t in self.targets if t not in self.selected],
                'complete': len(self.selected) == len(self.targets) and not self.active and not self.drift,
                'paused_providers': self.paused, 'drift': sorted(self.drift),
                'selected_final_records': list(self.selected.values()), 'last_event_sha256': self.previous}


def validate_record(record, manifest, target):
    cid = target['identity']['config_id']
    model = manifest['models'][cid]
    if (record.get('source_namespace') != CURRENT_NAMESPACE or record.get('source_config_id') != cid
            or record.get('current_config_id') != cid):
        raise ValueError('source/current configuration mismatch')
    if record.get('probability_semantics') != model['probability_semantics']:
        raise ValueError('probability field-review evidence mismatch')
    if record['status'] not in TERMINAL_STATUSES:
        raise ValueError('unknown status')
    if record['status'] == 'valid':
        if record['selected_label'] not in tuple('ABCD') or not record['artifacts']['raw_response'] or not record['artifacts']['final']:
            raise ValueError('valid answer requires native label/raw/final')
        if (record['route']['returned_model'] != model['expected_returned_model'] or
            record['route']['returned_route'] != model['route']):
            raise ValueError('valid answer identity drift')
    elif record['selected_label'] is not None or not record['failure_reason']:
        raise ValueError('failure cannot carry selected label')
    probability = record['probability']
    if probability['native_vector'] is not None:
        checked = native_probability(probability['native_vector'], record['selected_label'],
            semantics=model['probability_semantics']['meaning'], sum_tolerance='0.02', decisions=True)
        if probability != checked:
            raise ValueError('probability metadata mismatch')
    elif probability['valid'] or probability['selected_probability'] is not None or not probability['reason']:
        raise ValueError('missing-vector metadata mismatch')
    if probability['concentration'] is not None:
        raise ValueError('provider scalars must remain separate')
    for path, scalar in record['provider_scalars'].items():
        validate_provider_confidence({k: scalar[k] for k in ('value','reason')})
        if scalar['semantics'] != model['scalar_semantics'].get(path, 'unreviewed'):
            raise ValueError('scalar semantics changed')
    if not set(model['scalar_semantics']).issubset(record['provider_scalars']):
        raise ValueError('missing provider scalar absence records')
    for key in ('input_tokens','output_tokens','reasoning_tokens','final_tokens','billed_tokens'):
        validate_measurement(record['usage'][key], key, integer=True)
    for key in ('observed_usd','estimated_usd'):
        validate_measurement(record['billing'][key], key)
    http_status = record['scheduler']['response_status']
    if http_status is not None and (type(http_status) is not int or not 100 <= http_status <= 599):
        raise ValueError('invalid HTTP response status')
    if record['billing']['currency'] != 'USD':
        raise ValueError('USD required')
    for key in ('duration_seconds','queue_wait_seconds'):
        validate_measurement(record['timing'][key], key)
    start, finish = (datetime.fromisoformat(record['timing'][k]) for k in ('started_at','finished_at'))
    if start.tzinfo is None or finish.tzinfo is None or finish < start:
        raise ValueError('invalid request timing')
    if (record['route']['requested_model'] != model['model_id'] or
        record['route']['requested_route'] != model['route'] or
        record['scheduler']['provider'] != model['route']):
        raise ValueError('requested route mismatch')


def read_extension_package(package, *, return_ledger=False):
    package = Path(package).resolve()
    manifest = strict_json((package/'manifest.json').read_text())
    if manifest.get('schema_version') != SCHEMA or manifest.get('execution_workflow') != WORKFLOW:
        raise ValueError('unsupported extension schema/workflow')
    if manifest.get('purpose') not in ('development','evaluation'):
        raise ValueError('explicit collection purpose required')
    if ((manifest.get('provenance') == 'development' and manifest['purpose'] != 'development') or
        (manifest.get('provenance') == 'definitive' and manifest['purpose'] != 'evaluation')):
        raise ValueError('provenance/purpose mismatch')
    if manifest.get('config_namespace') != CURRENT_NAMESPACE or manifest.get('provenance') not in ('mock','development','definitive'):
        raise ValueError('invalid namespace/provenance')
    inventory = strict_json((package/'provenance.json').read_text())
    if fingerprint(inventory) != manifest['bindings']['execution_record']:
        raise ValueError('execution inventory binding mismatch')
    if set(inventory['code']) != set(SOURCES):
        raise ValueError('incomplete source inventory')
    for group in ('code','snapshots'):
        for name, digest in inventory[group].items():
            check_artifact({'path': 'source/'+name, 'sha256': digest}, package)
    config = tomllib.loads((package/'source/config.toml').read_text())
    errors = config_errors(config)
    if errors:
        raise ValueError('; '.join(errors))
    if manifest.get('campaign') != config.get('campaign'):
        raise ValueError('campaign mismatch')
    ids = [c['config_id'] for c in manifest['configurations']]
    if len(set(ids)) != len(ids) or not ids or any(cid not in EXTENSION_IDS for cid in ids):
        raise ValueError('invalid selected configuration inventory')
    if manifest['provenance'] == 'definitive' and tuple(ids) != EXTENSION_IDS:
        raise ValueError('complete D03-D13 definitive campaign required')
    if manifest['configurations'] != configurations(config, ids) or manifest['models'] != {cid: config['models'][cid] for cid in ids}:
        raise ValueError('model/configuration inventory changed')
    questions = load_questions(package/'source/questions.jsonl')
    public = strict_json((package/'source/inference-manifest.json').read_text())
    if (public['questions_sha256'] != file_hash(package/'source/questions.jsonl') or
        public['ordered_ids'] != [q.item_id for q in questions] or
        public['count'] != len(questions)):
        raise ValueError('public input mismatch')
    if manifest['provenance'] == 'definitive' and (len(questions) != 1273 or manifest['split'] != 'evaluation' or public['provenance'] != 'prepared'):
        raise ValueError('definitive evaluation input required')
    if manifest['purpose'] == 'development' and (len(questions) != 1 or manifest['split'] != 'development'):
        raise ValueError('one development question required')
    expected_bindings = {'execution_record': fingerprint(inventory), 'code': fingerprint(inventory['code']),
        'environment': fingerprint(inventory['environment']), 'config': fingerprint(config),
        'questions': inventory['snapshots']['questions.jsonl'], 'keys': public['keys_sha256'],
        'input_manifest': inventory['snapshots']['inference-manifest.json'],
        'study_protocol': inventory['snapshots']['study-protocol.md'], 'prompt': fingerprint(config['prompt'])}
    if manifest['bindings'] != expected_bindings:
        raise ValueError('input/source bindings mismatch')
    if manifest['key_manifest'] != {'sha256': public['keys_sha256'], 'preparation_sha256': public['preparation_manifest_sha256']}:
        raise ValueError('separate key binding mismatch')
    expected_items = [{'item_id':q.item_id,'content_sha256':question_hash(q),'labels':list(q.labels)} for q in questions]
    order = [q.item_id for q in questions]
    random.Random(20261006).shuffle(order)
    if (manifest['items'] != expected_items or manifest['schedule']['item_order'] != order or
        manifest['schedule']['policy'] != config['execution'] or
        any(q.benchmark != manifest['benchmark'] or q.split != manifest['split'] for q in questions) or
        manifest['runs'] != runs(manifest['study_id'], ids, manifest['purpose'] == 'development')):
        raise ValueError('run/input/schedule inventory mismatch')
    ledger = Ledger(manifest)
    qmap = {q.item_id:q for q in questions}
    for event in read_jsonl(package/'events.jsonl'):
        ledger.apply(event, package)
        if event['event'] == 'start':
            target = ledger.targets[event['original_query_id']]
            ident = target['identity']
            actual = strict_json((package/event['request']['path']).read_text())
            expected = request_payload(qmap[ident['item_id']], manifest['models'][ident['config_id']], config['prompt'])
            if actual != expected:
                raise ValueError('full native request differs from bound question/configuration')
    # Validate actual transport start spacing independently of dispatch bookkeeping.
    starts_by_provider = {}
    for record in ledger.records:
        stamp = datetime.fromisoformat(record['timing']['started_at'])
        starts_by_provider.setdefault(record['scheduler']['provider'], []).append(stamp)
    spacing = config['execution']['min_interval_seconds']
    for stamps in starts_by_provider.values():
        ordered = sorted(stamps)
        if any((b-a).total_seconds() < spacing-0.001 for a,b in zip(ordered, ordered[1:])):
            raise ValueError('observed provider start spacing violated')
    if not ledger.plan_seen:
        raise ValueError('missing journal plan')
    if return_ledger:
        return manifest, ledger
    return manifest, ledger.records, ledger.coverage()
