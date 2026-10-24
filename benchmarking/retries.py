"""Explicit manual collection of failed-query retries with one ten-worker pool."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmarking import adapters
from benchmarking.common import (ROOT, artifact_ref, evaluation_blockers, file_hash,
    fingerprint, load_config, load_questions, read_jsonl, run_environment_record,
    strict_json, utc_now, write_new, measurement, validate_finish, attempt_identity)
from benchmarking.retry_common import (CONFIG_NAMES, POLICY, SCHEMA, SOURCES, WORKFLOW,
    original_sources, read_retry_package, validate_configs, validate_source_configs, pause_hashes)
from benchmarking.run import (_dispatch_worker, _graceful_signals, _provider,
    _retry_after_seconds, _save_response, append_event, build_prompt, validate_input_manifest)


def candidate(source_paths, configs, config_paths, questions_path, public_path, protocol_path,
              output, *, mock):
    """Read-only candidate. Never creates a package, claim or credentials lookup."""
    output = Path(output).resolve()
    validate_configs(configs)
    prep = validate_input_manifest(public_path, questions_path)
    if not mock and prep['provenance'] != 'prepared':
        raise ValueError('live retries require prepared inputs')
    if not mock and evaluation_blockers(configs['decisions']):
        raise ValueError('reviewed D01 precision required')
    sources, targets, originals, source_manifests, segments, campaign = original_sources(source_paths)
    validate_source_configs(configs, source_paths)
    if not mock and ({s['workflow'] for s in sources} != {'lean-v1', 'lean-decisions-v1'} or
                     any(s['provenance'] != 'definitive' for s in sources)):
        raise ValueError('both complete definitive original and Luna studies required')
    questions = load_questions(questions_path); qmap = {q.item_id: q for q in questions}
    questions_sha256 = file_hash(questions_path)
    for tid, original in originals.items():
        saved = source_manifests[tid]
        if (saved['bindings']['questions'] != questions_sha256 or
                saved['bindings']['keys'] != prep['keys_sha256']):
            raise ValueError('source/retry question or original-key binding mismatch')
    requests = {}; models = {}; selected_configs = {}
    for target in targets:
        tid = target['target_id']; original = originals[tid]
        saved = source_manifests[tid]
        name = 'decisions' if saved['execution_workflow'] == 'lean-decisions-v1' else 'original'
        config = configs[name]; model = config['models'][original['identity']['config_id']]
        source_config = strict_json((segments[tid]/'source/config.json').read_text())
        if source_config['prompt'] != config['prompt'] or source_config['models'][original['identity']['config_id']] != model:
            raise ValueError('original scientific request/parse settings changed')
        request_bytes = (segments[tid]/original['artifacts']['request']['path']).read_bytes()
        request = strict_json(request_bytes.decode('utf-8'))
        q = qmap[original['identity']['item_id']]
        if adapters.build_request(q, model, build_prompt(q, config), mock=mock) != request:
            raise ValueError('rebuilt retry request differs from original')
        requests[tid] = (request, request_bytes)
        models[tid] = model; selected_configs[tid] = config
    snapshots = {'questions.jsonl': file_hash(questions_path),
        'inference-manifest.json': file_hash(public_path),
        'study-protocol.md': file_hash(protocol_path), 'README.md': file_hash(ROOT/'README.md')}
    for name in CONFIG_NAMES:
        snapshots[name+'.toml'] = file_hash(config_paths[name])
    code = {name: file_hash(ROOT/name) for name in SOURCES}
    local_configs = [Path(config_paths[n]).resolve() for n in CONFIG_NAMES]
    outside_repository = any(not p.is_relative_to(ROOT) for p in local_configs)
    selected_files = (*SOURCES, *(str(p.relative_to(ROOT)) for p in local_configs if p.is_relative_to(ROOT)),
                      'README.md', '.gitignore')
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = outside_repository or bool(subprocess.run(['git', 'status', '--porcelain', '--', *selected_files], cwd=ROOT,
                                          capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        raise ValueError('accepted Git execution checkout required') from None
    inventory = {'workflow': WORKFLOW, 'type': 'retry_execution_provenance',
        'code': code, 'configs': configs, 'environment': run_environment_record(),
        'git': {'commit': commit, 'dirty': dirty}, 'snapshots': snapshots}
    bindings = {'execution_record': fingerprint(inventory), 'code': fingerprint(code),
        'environment': fingerprint(inventory['environment']), 'config': fingerprint(configs),
        'questions': snapshots['questions.jsonl'], 'input_manifest': snapshots['inference-manifest.json'],
        'study_protocol': snapshots['study-protocol.md'], 'readme': snapshots['README.md'], 'keys': prep['keys_sha256']}
    manifest = {'schema_version': SCHEMA, 'execution_workflow': WORKFLOW,
        'campaign_id': campaign, 'provenance': 'mock' if mock else 'definitive',
        'policy': POLICY, 'bindings': bindings, 'targets': targets,
        'sources': [{**s, 'path': os.path.relpath(Path(path).resolve(), output)}
                    for s, path in zip(sources, source_paths)]}
    return manifest, inventory, (originals, source_manifests, qmap, requests, models, selected_configs)


def save_inventory(output, inventory, configs, config_paths, questions_path, public_path, protocol_path):
    paths = {'questions.jsonl': questions_path, 'inference-manifest.json': public_path,
             'study-protocol.md': protocol_path, 'README.md': ROOT/'README.md',
             **{n+'.toml': config_paths[n] for n in CONFIG_NAMES},
             **{name: ROOT/name for name in SOURCES}}
    for name, source in paths.items():
        expected = inventory['code'].get(name, inventory['snapshots'].get(name))
        data = Path(source).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('retry source/input changed before snapshot: '+name)
        path = output/'source'/name; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
    for name in CONFIG_NAMES:
        write_new(output/'source'/f'{name}.json', configs[name])
    write_new(output/'provenance.json', inventory)


def run_retries(source_paths, configs, config_paths, questions_path, public_path, protocol_path,
                output, *, mock, mock_registry=None, check_only=False, reviewer=None,
                reviewed_stops=None, transport=None, progress_stream=None, progress_interval=30.0):
    """Every call is explicit, uniquely indexed and persisted before transport."""
    if mock and mock_registry is None:
        raise ValueError('isolated --mock-registry required for fictional retries')
    if not mock and (mock_registry is not None or transport is not None):
        raise ValueError('fictional registry/transport prohibited with live retries')
    output = Path(output).resolve()
    manifest, inventory, context = candidate(source_paths, configs, config_paths, questions_path,
        public_path, protocol_path, output, mock=mock)
    current_dirty = inventory['git']['dirty']
    if (output/'manifest.json').exists():
        saved, _, coverage = read_retry_package(output)
        old = strict_json((output/'provenance.json').read_text())
        if {k: v for k, v in old.items() if k != 'git'} != {k: v for k, v in inventory.items() if k != 'git'}:
            raise ValueError('retry source/config/input/protocol/environment bindings changed')
        inventory = old; manifest['bindings']['execution_record'] = fingerprint(old)
        if saved != manifest:
            raise ValueError('retry campaign manifest changed')
    else:
        coverage = {'starts': 0, 'terminal': 0, 'complete': not manifest['targets'],
            'started_by_target': {}, 'recovered_by_target': {}, 'paused_providers': {},
            'drift': False, 'indeterminate': [], 'last_event_sha256': None}
    if coverage['indeterminate'] or coverage['drift']:
        raise ValueError('indeterminate retry starts or drift require human review; resend prohibited')
    stops = pause_hashes(coverage['paused_providers'])
    if reviewed_stops or reviewer:
        if not reviewer or not stops or sorted(reviewed_stops or []) != stops:
            raise ValueError('exact paused-provider stop hashes and reviewer required')
    elif stops and not check_only:
        raise ValueError('paused retry providers require explicit stop-event review')
    registry = Path(mock_registry).resolve() if mock else ROOT/'benchmarking/output/retry-campaigns'
    claim_dir = registry/manifest['campaign_id']; claim_path = claim_dir/'claim.json'
    claim = {'campaign_id': manifest['campaign_id'], 'output': str(output),
             'manifest_sha256': fingerprint(manifest)}
    if claim_path.exists() and strict_json(claim_path.read_text()) != claim:
        raise ValueError('original queries already claimed; resume the existing retry campaign')
    if not claim_path.exists() and output.exists() and any(output.iterdir()):
        raise ValueError('unregistered retry output must be empty')
    preview = {'check_only': check_only, 'targets': len(manifest['targets']),
        'original_attempts': len(context[0]),
        'primary_targets': sum(t['role'] == 'primary' for t in manifest['targets']),
        'secondary_targets': sum(t['role'] == 'secondary' for t in manifest['targets']),
        'max_additional_calls': len(manifest['targets'])*10, 'global_workers': 10,
        'provider_workers': 10, 'policy': POLICY, 'starts': coverage['starts'],
        'terminal': coverage['terminal'], 'pending_stop_review': stops,
        'source_dirty': current_dirty, 'campaign_id': manifest['campaign_id'],
        'manifest_sha256': fingerprint(manifest), 'bindings': manifest['bindings']}
    if check_only or not manifest['targets']:
        return {**preview, 'complete': coverage['complete']}
    if not mock:
        if current_dirty:
            raise ValueError('commit clean retry execution source/configuration before launch')
        if not os.environ.get('OPENROUTER_API_KEY'):
            raise ValueError('credential unavailable')
    with ExitStack() as stack:
        claim_dir.mkdir(parents=True, exist_ok=True)
        lock = stack.enter_context((claim_dir/'.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if claim_path.exists():
            if strict_json(claim_path.read_text()) != claim:
                raise ValueError('duplicate retry campaign claim')
            if not (output/'manifest.json').exists():
                raise ValueError('claimed retry package missing; preserve and review')
        else:
            if output.exists() and any(output.iterdir()):
                raise ValueError('fresh retry output must be empty')
            write_new(claim_path, claim)
            output.mkdir(parents=True, exist_ok=True)
            save_inventory(output, inventory, configs, config_paths, questions_path, public_path, protocol_path)
            write_new(output/'manifest.json', manifest)
            append_event(output/'events.jsonl', {'schema_version': SCHEMA, 'event': 'plan',
                'manifest_sha256': fingerprint(manifest), 'created_at': utc_now()}, None)
        # Re-read inside the exclusive campaign lock, including all source hashes.
        manifest, terminal, coverage = read_retry_package(output)
        return _collect(output, manifest, terminal, coverage, context, mock=mock,
            reviewer=reviewer, reviewed_stops=reviewed_stops, transport=transport,
            progress_stream=progress_stream, progress_interval=progress_interval)


def _collect(output, manifest, terminal, coverage, context, *, mock, reviewer,
             reviewed_stops, transport, progress_stream, progress_interval):
    originals, source_manifests, qmap, requests, models, configs = context
    previous = coverage['last_event_sha256']
    event_base = {'schema_version': SCHEMA, 'manifest_sha256': fingerprint(manifest)}
    counts = dict(coverage['started_by_target']); recovered = dict(coverage['recovered_by_target'])
    paused = dict(coverage['paused_providers']); streaks = {}; next_start = {}
    for event in read_jsonl(output/'events.jsonl'):
        if event['event'] == 'review':
            streaks.clear()
        if event['event'] != 'finish':
            continue
        record = event['record']; provider = record['scheduler']['provider']
        status = record['scheduler']['response_status']
        streaks[provider] = streaks.get(provider, 0)+1 if status == 429 else 0
        age = (datetime.now(timezone.utc)-datetime.fromisoformat(record['timing']['started_at'])).total_seconds()
        next_start[provider] = max(next_start.get(provider, 0), time.monotonic()+max(0, POLICY['provider_min_interval_seconds'].get(provider, 0)-age))
        if record['artifacts']['raw_response']:
            raw = strict_json((output/record['artifacts']['raw_response']['path']).read_text())
            finish_age = (datetime.now(timezone.utc)-datetime.fromisoformat(record['timing']['finished_at'])).total_seconds()
            if status == 429:
                next_start[provider] = max(next_start.get(provider, 0), time.monotonic()+_retry_after_seconds(raw, elapsed_seconds=finish_age))
    if reviewer:
        if sorted(reviewed_stops or []) != pause_hashes(paused):
            raise ValueError('provider-stop review changed while locking')
        previous = append_event(output/'events.jsonl', {**event_base, 'event': 'review',
            'reviewer': reviewer, 'reviewed_at': utc_now(),
            'reviewed_stop_events': sorted(reviewed_stops)}, previous)
        for provider in paused:
            streaks[provider] = 0
        paused.clear()
    targets = manifest['targets']; active = set(); stop = False; unexpected = None
    queued = {t['target_id']: time.monotonic() for t in targets}
    stream = progress_stream if progress_stream is not None else (None if mock else sys.stderr)
    next_report = 0.0

    def finished(tid):
        return tid in recovered or counts.get(tid, 0) == 10 and tid not in active

    def report(force=False):
        nonlocal next_report
        if stream is not None and (force or time.monotonic() >= next_report):
            resolved = sum(finished(t['target_id']) for t in targets)
            print(f'[{utc_now()}] Retries: {resolved}/{len(targets)} queries resolved, '
                f'{len(recovered)} recovered, {sum(counts.values())} starts, '
                f'{len(active)} active; paused={",".join(sorted(paused)) or "none"}', file=stream, flush=True)
            next_report = time.monotonic()+progress_interval

    with _graceful_signals() as signals, ThreadPoolExecutor(max_workers=10) as pool:
        inflight = {}; sequence = sum(counts.values())
        report(force=True)
        while True:
            if signals:
                stop = True
            report()
            primary_done = all(finished(t['target_id']) for t in targets if t['role'] == 'primary')
            candidates = [t for t in targets if not finished(t['target_id']) and t['target_id'] not in active
                          and (t['role'] == 'primary' or primary_done)]
            # Give unstarted eligible targets their first retry before taking a
            # further round, retaining original seeded order for ties.
            candidates.sort(key=lambda t: counts.get(t['target_id'], 0))
            while not stop and len(inflight) < 10:
                ready = next((t for t in candidates if _provider(models[t['target_id']]) not in paused
                    and next_start.get(_provider(models[t['target_id']]), 0) <= time.monotonic()), None)
                if ready is None:
                    break
                candidates.remove(ready); tid = ready['target_id']; model = models[tid]
                provider = _provider(model); index = counts.get(tid, 0)+1
                request, data = requests[tid]; q = qmap[originals[tid]['identity']['item_id']]
                try:
                    adir = output/'artifacts'/tid/str(index); adir.mkdir(parents=True, exist_ok=False)
                    with (adir/'request.json').open('xb') as stream_file:
                        stream_file.write(data); stream_file.flush(); os.fsync(stream_file.fileno())
                    ref = artifact_ref(adir/'request.json', output)
                    previous = append_event(output/'events.jsonl', {**event_base, 'event': 'start',
                        'target_id': tid, 'retry_index': index, 'provider': provider,
                        'started_at': utc_now(), 'request': ref}, previous)
                    counts[tid] = index; active.add(tid)
                    sequence += 1
                    future = pool.submit(_dispatch_worker, request, model, q, configs[tid], mock, transport, queued[tid])
                    inflight[future] = (sequence, tid, index, adir, ref)
                    next_start[provider] = time.monotonic()+POLICY['provider_min_interval_seconds'].get(provider, 0)
                except BaseException as exc:
                    unexpected = exc; stop = True; break
            if not inflight:
                if stop or not candidates or all(_provider(models[t['target_id']]) in paused for t in candidates):
                    break
                time.sleep(.1)
                continue
            done, _ = wait(inflight, timeout=.25, return_when=FIRST_COMPLETED)
            results = []
            for future in done:
                details = inflight.pop(future)
                try:
                    result = future.result()
                    results.append((result['finished_at'], details[0], details, result))
                except BaseException as exc:
                    unexpected = unexpected or exc; stop = True
            for _, _, details, result in sorted(results, key=lambda r: (r[0], r[1])):
                _, tid, index, adir, ref = details
                request, _ = requests[tid]; model = models[tid]; provider = _provider(model)
                try:
                    record = _save_response(configs[tid], qmap[originals[tid]['identity']['item_id']], model,
                        originals[tid]['identity'], request, ref, adir, output, result)
                    raw = result['raw']; status = raw.get('status') if raw else None
                    streaks[provider] = streaks.get(provider, 0)+1 if status == 429 else 0
                    reason = 'provider_rate_limit_stop' if streaks[provider] >= 3 else None
                    record['scheduler'] = {'provider': provider, 'response_status': status, 'stop_reason': reason}
                    record['timing']['queue_wait_seconds'] = measurement(result['queue_wait_seconds'])
                    validate_finish(record, source_manifests[tid], allowed_ids={attempt_identity(originals[tid])})
                    previous = append_event(output/'events.jsonl', {**event_base, 'event': 'finish',
                        'target_id': tid, 'retry_index': index, 'record': record}, previous)
                    active.remove(tid); queued[tid] = time.monotonic()
                    next_start[provider] = max(next_start.get(provider, 0),
                        result['started_monotonic']+POLICY['provider_min_interval_seconds'].get(provider, 0))
                    if status == 429:
                        next_start[provider] = max(next_start.get(provider, 0), time.monotonic()+_retry_after_seconds(raw))
                    if reason:
                        paused.setdefault(provider, []).append(previous)
                    if record['status'] == 'valid':
                        recovered[tid] = index
                    if record['status'] == 'drift':
                        stop = True
                except BaseException as exc:
                    unexpected = unexpected or exc; stop = True
        report(force=True)
    if unexpected is not None:
        raise unexpected
    return read_retry_package(output)[2]
