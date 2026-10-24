"""Pure supplementary-retry contract. No transport, credentials or scoring keys."""
from pathlib import Path
import os
import re

from benchmarking.common import (EXECUTION_SOURCES, attempt_identity,
    check_artifact, file_hash, fingerprint, load_questions, read_jsonl, read_package,
    strict_json, validate_finish, runtime_blockers, scheduler_policy)

WORKFLOW = 'lean-failure-retries-v1'
SCHEMA = 5
MAX_RETRIES = 10
SOURCES = EXECUTION_SOURCES + ('benchmarking/retry_common.py', 'benchmarking/retries.py')
POLICY = {'max_additional_starts': 10, 'stop': 'first_valid',
          'analysis_role': 'secondary_recovery', 'global_workers': 10,
          'provider_workers': 10, 'timeout_seconds': 600,
          'provider_min_interval_seconds': {'InferenceNet': 5.0, 'Mistral': 2.0, 'OpenAI': 5.0},
          'rate_limit_stop_after': 3, 'one_active_per_target': True,
          'target_order': 'original_seeded_item_order'}
CONFIG_NAMES = ('original', 'decisions')


def relative_path(package, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError('relative source package path required')
    return (Path(package) / name).resolve()


def original_sources(paths):
    """Verify complete immutable sources; retain each artifact's actual segment."""
    sources = []; originals = {}; manifests = {}; segments = {}; targets = []; ordering = {}
    study_ids = set()
    for source_index, path in enumerate(paths):
        path = Path(path).resolve()
        manifest, records, coverage = read_package(path)
        if manifest['study_id'] in study_ids:
            raise ValueError('duplicate original study')
        study_ids.add(manifest['study_id'])
        if not coverage['complete'] or coverage['indeterminate'] or any(r['status'] == 'drift' for r in records):
            raise ValueError('complete original collection without indeterminate starts/drift required')
        if {r['experiment_id'] for r in manifest['runs']} != {'E1', 'E2'}:
            raise ValueError('complete primary and repeat source collection required')
        sources.append({'manifest_sha256': file_hash(path/'manifest.json'),
                        'provenance_sha256': file_hash(path/'provenance.json'),
                        'events_sha256': file_hash(path/'events.jsonl')})
        chain = []; segment = path
        while True:
            local = strict_json((segment/'manifest.json').read_text())
            chain.append((segment, local))
            if 'continuation' not in local:
                break
            segment = relative_path(segment, local['continuation']['parent'])
        locations = {}
        for segment, local in reversed(chain):
            starts = {}
            for event in read_jsonl(segment/'events.jsonl'):
                if event['event'] == 'start':
                    starts[attempt_identity(event)] = event
                elif event['event'] == 'finish':
                    ident = attempt_identity(event)
                    if ident in locations:
                        raise ValueError('duplicate original terminal identity')
                    locations[ident] = (segment, local, starts[ident], event)
        item_positions = {item: i for i, item in enumerate(manifest['schedule']['item_order'])}
        primary_configs = [r['config_id'] for r in manifest['runs'] if r['role'] == 'primary']
        for record in records:
            ident = attempt_identity(record)
            if ident in originals:
                raise ValueError('duplicate original identity across sources')
            target_id = fingerprint(list(ident))
            segment, local, start, finish = locations[ident]
            run = next(r for r in local['runs'] if r['run_id'] == ident[-1])
            originals[target_id] = record
            manifests[target_id] = local
            segments[target_id] = segment
            if record['status'] != 'valid':
                item_position = item_positions[ident[2]]
                rotation = item_position % len(primary_configs)
                config_position = (primary_configs.index(ident[3])-rotation) % len(primary_configs)
                ordering[target_id] = (run['role'] != 'primary', run['replicate'] if run['role'] != 'primary' else 0,
                                       item_position, source_index, config_position)
                targets.append({'target_id': target_id, 'identity': record['identity'],
                    'source_index': source_index,
                    'segment': str(segment.relative_to(path)) if segment.is_relative_to(path) else
                               os.path.relpath(segment, path),
                    'start_event_sha256': start['event_sha256'],
                    'finish_event_sha256': finish['event_sha256'],
                    'request_sha256': record['artifacts']['request']['sha256'],
                    'original_status': record['status'],
                    **{k: run[k] for k in ('role', 'experiment_id', 'replicate')}})
        sources[-1]['study_id'] = manifest['study_id']
        sources[-1]['workflow'] = manifest['execution_workflow']
        sources[-1]['provenance'] = manifest['provenance']
    targets.sort(key=lambda t: ordering[t['target_id']])
    # Claim identity excludes output/campaign names and volatile hashes, so a new
    # destination or altered provenance cannot reset the original-query budget.
    campaign_id = fingerprint(sorted(list(attempt_identity(r)) for r in originals.values()))
    return sources, targets, originals, manifests, segments, campaign_id


def validate_configs(configs):
    if set(configs) != set(CONFIG_NAMES):
        raise ValueError('original and decisions configurations required')
    for name, config in configs.items():
        errors = runtime_blockers(config)
        if errors:
            raise ValueError('; '.join(errors))
        policy = scheduler_policy(config)
        if (policy['version'] != 3 or policy['concurrency'] != 10 or
                policy['provider_concurrency'] != 10 or
                any(n != 10 for n in policy['provider_concurrency_overrides'].values()) or
                policy['rate_limit_stop_after'] != 3 or config['execution']['timeout_seconds'] != 600):
            raise ValueError('accepted ten-worker retry runtime required')
        wanted = {'OpenAI': 5.0} if name == 'decisions' else {'InferenceNet': 5.0, 'Mistral': 2.0, 'OpenAI': 5.0}
        if policy['provider_min_interval_seconds'] != wanted:
            raise ValueError('accepted retry provider pacing required')
        expected = 'lean-decisions-v1' if name == 'decisions' else 'lean-v1'
        if config['execution_workflow'] != expected:
            raise ValueError('retry source/configuration workflow mismatch')


def validate_source_configs(configs, paths):
    """Allow worker ceilings and the exact accepted OpenAI pacing amendment."""
    for path in paths:
        saved = strict_json((Path(path)/'source/config.json').read_text())
        name = 'decisions' if saved['execution_workflow'] == 'lean-decisions-v1' else 'original'
        current = configs[name]
        def scientific(config):
            execution = {k: v for k, v in config['execution'].items()
                         if k not in {'concurrency', 'provider_concurrency', 'provider_concurrency_overrides'}}
            return {**config, 'execution': execution}
        previous = scientific(saved)
        wanted = scientific(current)
        old_intervals = {'OpenAI': 1.0} if name == 'decisions' else {'InferenceNet': 5.0, 'Mistral': 2.0}
        new_intervals = {'OpenAI': 5.0} if name == 'decisions' else {**old_intervals, 'OpenAI': 5.0}
        if (previous['execution']['provider_min_interval_seconds'] == old_intervals and
                wanted['execution']['provider_min_interval_seconds'] == new_intervals):
            previous['execution']['provider_min_interval_seconds'] = new_intervals
        if previous != wanted:
            raise ValueError('retry scientific/source configuration changed beyond accepted workers/OpenAI pacing')


def pause_hashes(pauses):
    return sorted(digest for digests in pauses.values() for digest in digests)


def source_context(package, manifest):
    paths = [relative_path(package, s['path']) for s in manifest['sources']]
    sources, targets, originals, manifests, segments, campaign = original_sources(paths)
    expected_sources = [{**s, 'path': saved['path']} for s, saved in zip(sources, manifest['sources'])]
    if (expected_sources != manifest['sources'] or targets != manifest['targets'] or
            campaign != manifest['campaign_id']):
        raise ValueError('original retry source evidence/eligibility changed')
    if manifest['provenance'] == 'definitive' and any(s['provenance'] != 'definitive' for s in sources):
        raise ValueError('definitive retries cannot use fictional originals')
    configs = strict_json((Path(package)/'provenance.json').read_text())['configs']
    validate_source_configs(configs, paths)
    saved_configs = {segment: strict_json((segment/'source/config.json').read_text())
                     for segment in set(segments.values())}
    for tid, saved_manifest in manifests.items():
        name = 'decisions' if saved_manifest['execution_workflow'] == 'lean-decisions-v1' else 'original'
        saved = saved_configs[segments[tid]]
        cid = originals[tid]['identity']['config_id']
        if saved['prompt'] != configs[name]['prompt'] or saved['models'][cid] != configs[name]['models'][cid]:
            raise ValueError('retry request/parse settings differ from originating segment')
    return originals, manifests, segments


def validate_inventory(package, manifest):
    package = Path(package)
    inventory = strict_json((package/'provenance.json').read_text())
    if inventory.get('workflow') != WORKFLOW or fingerprint(inventory) != manifest['bindings']['execution_record']:
        raise ValueError('retry execution provenance mismatch')
    if set(inventory['code']) != set(SOURCES):
        raise ValueError('retry execution source inventory incomplete')
    for name, digest in inventory['code'].items():
        check_artifact({'path': 'source/'+name, 'sha256': digest}, package)
    for name, digest in inventory['snapshots'].items():
        check_artifact({'path': 'source/'+name, 'sha256': digest}, package)
    configs = {name: strict_json((package/'source'/f'{name}.json').read_text()) for name in CONFIG_NAMES}
    if configs != inventory['configs']:
        raise ValueError('retry configuration snapshot mismatch')
    validate_configs(configs)
    from benchmarking.common import is_decisions, evaluation_blockers, load_config
    for name in CONFIG_NAMES:
        if load_config(package/'source'/f'{name}.toml') != configs[name]:
            raise ValueError('retry TOML/configuration mismatch')
        if manifest['provenance'] != 'mock' and is_decisions(configs[name]):
            if evaluation_blockers(configs[name]):
                raise ValueError('reviewed D01 evaluation precision required')
    public = strict_json((package/'source/inference-manifest.json').read_text())
    questions = load_questions(package/'source/questions.jsonl')
    if (public['questions_sha256'] != file_hash(package/'source/questions.jsonl') or
            public['keys_sha256'] != manifest['bindings']['keys']):
        raise ValueError('retry input/key binding mismatch')
    expected = {'execution_record': fingerprint(inventory), 'code': fingerprint(inventory['code']),
        'environment': fingerprint(inventory['environment']), 'config': fingerprint(configs),
        'questions': inventory['snapshots']['questions.jsonl'],
        'input_manifest': inventory['snapshots']['inference-manifest.json'],
        'study_protocol': inventory['snapshots']['study-protocol.md'],
        'readme': inventory['snapshots']['README.md'], 'keys': public['keys_sha256']}
    if manifest['bindings'] != expected:
        raise ValueError('retry saved execution bindings mismatch')
    if manifest['provenance'] != 'mock' and public['provenance'] != 'prepared':
        raise ValueError('live retry input provenance must be prepared')
    return inventory, questions


def read_retry_package(package):
    """Validate first-valid/cap/identity/source invariants; never merge originals."""
    package = Path(package).resolve()
    manifest = strict_json((package/'manifest.json').read_text())
    if (manifest.get('schema_version') != SCHEMA or manifest.get('execution_workflow') != WORKFLOW or
            manifest.get('policy') != POLICY or manifest.get('provenance') not in {'mock', 'definitive'}):
        raise ValueError('unsupported supplementary retry policy/schema')
    if not re.fullmatch('[0-9a-f]{64}', manifest.get('campaign_id', '')):
        raise ValueError('invalid retry campaign identity')
    inventory, questions = validate_inventory(package, manifest)
    originals, source_manifests, segments = source_context(package, manifest)
    target_map = {t['target_id']: t for t in manifest['targets']}
    if len(target_map) != len(manifest['targets']):
        raise ValueError('duplicate retry target')
    qmap = {q.item_id: q for q in questions}
    for tid in originals:
        original_manifest = source_manifests[tid]
        if (original_manifest['bindings']['questions'] != manifest['bindings']['questions'] or
                original_manifest['bindings']['keys'] != manifest['bindings']['keys']):
            raise ValueError('retry/original question/key mapping mismatch')
        if attempt_identity(originals[tid])[2] not in qmap:
            raise ValueError('retry target absent from original questions')
    previous = None; plan_seen = False; starts = {}; finishes = {}; active = {}
    counts = {}; recovered = {}; streaks = {}; pauses = {}; drift = False
    for event in read_jsonl(package/'events.jsonl'):
        event = dict(event); digest = event.pop('event_sha256', None)
        if (event.get('previous_sha256') != previous or fingerprint(event) != digest or
                event.get('schema_version') != SCHEMA or
                event.get('manifest_sha256') != fingerprint(manifest)):
            raise ValueError('retry journal chain/binding mismatch')
        previous = digest
        kind = event['event']
        if kind == 'plan':
            if plan_seen or starts:
                raise ValueError('unique retry plan required')
            plan_seen = True
            continue
        if not plan_seen:
            raise ValueError('initial retry plan required')
        if kind == 'review':
            stops = event.get('reviewed_stop_events')
            if (not pauses or stops != pause_hashes(pauses) or
                    not isinstance(event.get('reviewer'), str) or not event['reviewer'].strip()):
                raise ValueError('exact retry provider-stop review required')
            if active or drift:
                raise ValueError('drained retry campaign without drift required for review')
            for provider in pauses:
                streaks[provider] = 0
            pauses.clear()
            continue
        tid = event['target_id']; index = event['retry_index']; key = (tid, index)
        if tid not in target_map or type(index) is not int or not 1 <= index <= MAX_RETRIES:
            raise ValueError('unexpected retry identity/index')
        if kind == 'start':
            provider = event['provider']
            if (tid in active or tid in recovered or index != counts.get(tid, 0)+1 or
                    key in starts or provider in pauses or drift):
                raise ValueError('duplicate, paused, drifted, recovered or out-of-order retry start')
            if target_map[tid]['role'] != 'primary':
                for other, target in target_map.items():
                    if target['role'] == 'primary' and other not in recovered and not (counts.get(other) == 10 and other not in active):
                        raise ValueError('secondary retries before primary recovery/exhaustion')
            if len(active) >= 10:
                raise ValueError('retry global worker ceiling exceeded')
            check_artifact(event['request'], package)
            if event['request']['sha256'] != target_map[tid]['request_sha256']:
                raise ValueError('retry request differs from original bytes')
            starts[key] = event
            active[tid] = index; counts[tid] = index
        elif kind == 'finish':
            if key not in starts or key in finishes or active.get(tid) != index:
                raise ValueError('retry finish without unique start')
            record = event['record']
            original = originals[tid]
            if record['identity'] != original['identity'] or record['artifacts']['request'] != starts[key]['request']:
                raise ValueError('retry finish original identity/request mismatch')
            source_manifest = source_manifests[tid]
            validate_finish(record, source_manifest, allowed_ids={attempt_identity(original)})
            for ref in record['artifacts'].values():
                check_artifact(ref, package)
            provider = record['scheduler']['provider']
            expected_provider = original['route']['requested_route']
            if provider != starts[key]['provider'] or provider != expected_provider:
                raise ValueError('retry provider identity mismatch')
            status = record['scheduler']['response_status']
            streaks[provider] = streaks.get(provider, 0)+1 if status == 429 else 0
            stop = 'provider_rate_limit_stop' if streaks[provider] >= 3 else None
            if record['scheduler']['stop_reason'] != stop:
                raise ValueError('retry provider pause threshold mismatch')
            if stop:
                pauses.setdefault(provider, []).append(digest)
            finishes[key] = record; del active[tid]
            if record['status'] == 'valid':
                recovered[tid] = index
            if record['status'] == 'drift':
                drift = True
        else:
            raise ValueError('unsupported retry journal event')
    if not plan_seen:
        raise ValueError('initial retry plan required')
    exhausted = {tid for tid, n in counts.items() if n == 10 and tid not in active and tid not in recovered}
    complete = len(recovered)+len(exhausted) == len(target_map)
    terminal = [{'target_id': tid, 'retry_index': index, 'record': record}
                for (tid, index), record in finishes.items()]
    coverage = {'expected_targets': len(target_map), 'max_additional_calls': len(target_map)*10,
        'starts': len(starts), 'terminal': len(finishes), 'recovered': len(recovered),
        'exhausted': len(exhausted), 'remaining_targets': len(target_map)-len(recovered)-len(exhausted),
        'complete': complete, 'indeterminate': [[tid, index] for tid, index in active.items()],
        'paused_providers': pauses, 'drift': drift, 'last_event_sha256': previous,
        'started_by_target': counts, 'recovered_by_target': recovered}
    return manifest, terminal, coverage
