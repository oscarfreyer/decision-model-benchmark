"""Manual, fictional-only regression checks for supplementary failure retries."""
import base64
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import signal
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from benchmarking.common import (ROOT, file_hash, fingerprint, load_config,
    read_jsonl, read_package, run_inventory, strict_json, write_new)
from benchmarking.run import (append_event, make_manifest, mock_dispatch,
    run_package, validate_input_manifest)
from benchmarking.retry_common import (read_retry_package, pause_hashes,
    validate_configs, validate_source_configs)
from benchmarking.retries import run_retries
from preparation.prepare_medqa import prepare


def http_error(status):
    return {'status': status, 'body_base64': base64.b64encode(b'{}').decode(), 'headers': {}}


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.real_sleep = time.sleep; self.real_clock = time.monotonic
        # Accelerate fictional provider pacing; transport remains thread-based.
        self.clock_patch = patch('time.monotonic', side_effect=lambda: self.real_clock()*1000)
        self.sleep_patch = patch('time.sleep', side_effect=lambda n: self.real_sleep(min(n, .001)))
        self.clock_patch.start(); self.sleep_patch.start()
        self.configs = {}; self.config_paths = {}
        for name, filename in (('original', 'config.toml'), ('decisions', 'decisions.toml')):
            text = (ROOT/'benchmarking'/filename).read_text().replace(
                'study_registry = "benchmarking/output/studies"',
                'study_registry = '+json.dumps(str(self.root/'original-registry')))
            path = self.root/(name+'.toml'); path.write_text(text)
            self.config_paths[name] = path; self.configs[name] = load_config(path)
        def rows(split, n):
            return [{'question': f'Fictional {split} symbol {i}',
                     'options': {k: f'{split} {i} {k}' for k in 'ABCD'}, 'answer_idx': 'A'} for i in range(n)]
        dev = self.root/'dev.jsonl'; test = self.root/'test.jsonl'
        write_new(dev, rows('development', 4), jsonl=True)
        write_new(test, rows('test', 3), jsonl=True)
        self.paths = prepare(dev_source=dev, test_source=test, seed=818, count=1,
            source_url='https://example.invalid/fictional', rights_basis='fictional fixture',
            acquired_at='2026-10-07', workspace=self.root, fictional=True, expected_test_items=3)
        self.questions = self.paths['evaluation']; self.public = self.paths['evaluation_manifest']
        self.protocol = self.root/'protocol.md'; self.protocol.write_text('Fictional approved retry protocol.\n')
        self.sources = [self.root/'original', self.root/'luna']
        calls = Counter(); lock = threading.Lock()
        def source_transport(request, model, q):
            if model['model_id'] == 'z-ai/glm-5':
                return http_error(503)
            if model['adapter'] == 'openai_decisions_gateway':
                with lock:
                    calls['luna'] += 1; n = calls['luna']
                if n == 1:
                    return http_error(502)
            return mock_dispatch(request, model, q)
        prep = validate_input_manifest(self.public, self.questions)
        for name, output in zip(('original', 'decisions'), self.sources):
            config = self.configs[name]
            inventory = run_inventory(config, self.questions, self.public, config_path=self.config_paths[name],
                protocol_path=self.protocol)
            manifest = make_manifest(config, self.questions, study_id='fictional-'+name,
                experiment='E1+E2', provenance='mock', seed=config['execution']['seed'],
                keys_sha256=prep['keys_sha256'], preparation_sha256=prep['preparation_manifest_sha256'],
                inventory=inventory, content_sha256=prep.get('content_manifest_sha256'))
            result = run_package(manifest, self.questions, config, output, mock=True, inventory=inventory,
                input_manifest_path=self.public, config_path=self.config_paths[name], transport=source_transport)
            self.assertTrue(result['complete'])
        self.output = self.root/'retries'; self.registry = self.root/'retry-registry'

    def tearDown(self):
        self.sleep_patch.stop(); self.clock_patch.stop(); self.temp.cleanup()

    def collect(self, **kwargs):
        options = dict(mock=True, mock_registry=self.registry)
        options.update(kwargs)
        return run_retries(self.sources, self.configs, self.config_paths, self.questions,
            self.public, self.protocol, self.output, **options)

    def test_read_only_plan_exact_failed_targets_and_first_valid_selection(self):
        preview = self.collect(check_only=True)
        self.assertEqual(preview['targets'], 4); self.assertEqual(preview['max_additional_calls'], 40)
        self.assertFalse(self.output.exists()); self.assertFalse(self.registry.exists())
        hashes = {str(p): file_hash(p) for source in self.sources for p in source.rglob('*') if p.is_file()}
        calls = Counter()
        def transport(request, model, q):
            key = (model['model_id'], q.item_id); calls[key] += 1
            if calls[key] == 1:
                return http_error(503)
            raw = mock_dispatch(request, model, q)
            body = json.loads(base64.b64decode(raw['body_base64']))
            if model['adapter'] == 'openai_decisions_gateway':
                body['answers']['answer']['choice'] = 'B'
            else:
                body['choices'][0]['message']['content'] = '{"answer":"B"}'
            raw['body_base64'] = base64.b64encode(json.dumps(body).encode()).decode()
            return raw
        result = self.collect(transport=transport)
        self.assertTrue(result['complete']); self.assertEqual(result['starts'], 8)
        self.assertEqual(result['recovered'], 4)
        self.assertEqual(hashes, {str(p): file_hash(p) for source in self.sources for p in source.rglob('*') if p.is_file()})
        from analysis.recovery import recovery_report
        report = recovery_report(self.output, self.paths['evaluation_keys'])
        recovered = [r for r in report['responses'] if r['selected_retry_index']]
        self.assertEqual(len(recovered), 4)
        self.assertTrue(all(r['selected_retry_index'] == 2 and not r['recovery_correct'] for r in recovered))
        self.assertTrue(all(not r['original_correct'] for r in recovered))
        self.assertEqual(len(report['responses']), 63)
        self.assertEqual({r['denominator'] for r in report['runs'].values()}, {3})

    def test_ten_start_exhaustion_and_resume_never_retries_valid_or_exhausted(self):
        calls = []
        def fail(*args):
            calls.append(1); return http_error(503)
        result = self.collect(transport=fail)
        self.assertTrue(result['complete']); self.assertEqual(result['starts'], 40)
        self.assertEqual(result['exhausted'], 4); self.assertEqual(result['recovered'], 0)
        self.assertTrue(self.collect(transport=lambda *args: self.fail('exhausted query dispatched'))['complete'])
        self.assertEqual(len(calls), 40)

    def test_ctrl_c_resume_preserves_cap_and_duplicate_destination_is_rejected(self):
        calls = []
        def stop(*args):
            calls.append(1); signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            return http_error(503)
        result = self.collect(transport=stop)
        self.assertFalse(result['complete']); self.assertFalse(result['indeterminate'])
        resumed = self.collect()
        self.assertTrue(resumed['complete']); self.assertGreater(resumed['starts'], len(calls))
        manifest = read_retry_package(self.output)[0]
        self.output = self.root/'duplicate'
        with self.assertRaisesRegex(ValueError, 'already claimed'):
            self.collect()
        self.assertFalse(self.output.exists())
        claim = self.registry/manifest['campaign_id']/'claim.json'
        self.assertTrue(claim.exists())

    def test_provider_pause_requires_all_exact_stop_hashes_before_resuming(self):
        result = self.collect(transport=lambda *args: http_error(429))
        self.assertFalse(result['complete']); self.assertTrue(result['paused_providers'])
        with self.assertRaisesRegex(ValueError, 'stop-event review'):
            self.collect()
        with self.assertRaisesRegex(ValueError, 'exact paused-provider'):
            self.collect(reviewer='Fictional reviewer', reviewed_stops=['0'*64])
        result = self.collect(reviewer='Fictional reviewer', reviewed_stops=pause_hashes(result['paused_providers']))
        self.assertTrue(result['complete'])
        self.assertTrue(any(e['event'] == 'review' for e in read_jsonl(self.output/'events.jsonl')))

    def test_indeterminate_start_blocks_resend_and_counts_toward_cap(self):
        result = self.collect()
        manifest, attempts, _ = read_retry_package(self.output)
        events = read_jsonl(self.output/'events.jsonl')
        # Simulate a crash after the final persisted start, before its response.
        last_start = max(i for i, e in enumerate(events) if e['event'] == 'start')
        write = ''.join(json.dumps(e)+'\n' for e in events[:last_start+1])
        (self.output/'events.jsonl').write_text(write)
        coverage = read_retry_package(self.output)[2]
        self.assertTrue(coverage['indeterminate'])
        self.assertGreater(coverage['starts'], coverage['terminal'])
        with self.assertRaisesRegex(ValueError, 'indeterminate retry starts'):
            self.collect(transport=lambda *args: self.fail('resend forbidden'))

    def test_request_protocol_source_and_worker_drift_are_rejected(self):
        self.collect()
        self.protocol.write_text('Changed fictional protocol\n')
        with self.assertRaisesRegex(ValueError, 'bindings changed'):
            self.collect()
        self.protocol.write_text('Fictional approved retry protocol.\n')
        config = self.configs['original']
        config['execution']['concurrency'] = 11
        with self.assertRaisesRegex(ValueError, 'ten-worker'):
            self.collect(check_only=True)
        config['execution']['concurrency'] = 10
        original_manifest = self.sources[0]/'manifest.json'
        original_manifest.write_text(original_manifest.read_text()+' ')
        with self.assertRaisesRegex(ValueError, 'source evidence'):
            read_retry_package(self.output)

    def test_historical_openai_pacing_amendment_rejects_unapproved_changes(self):
        source = self.root/'historical-pacing'
        (source/'source').mkdir(parents=True)
        config_path = source/'source/config.json'
        for name, old_intervals in (
                ('original', {'InferenceNet': 5.0, 'Mistral': 2.0}),
                ('decisions', {'OpenAI': 1.0})):
            with self.subTest(configuration=name):
                saved = deepcopy(self.configs[name])
                saved['execution']['provider_min_interval_seconds'] = old_intervals
                config_path.write_text(json.dumps(saved))
                validate_source_configs(self.configs, [source])
                self.assertEqual(json.loads(config_path.read_text()), saved)
                config_path.write_text(json.dumps(self.configs[name]))
                validate_source_configs(self.configs, [source])
                saved['execution']['provider_min_interval_seconds'] = {**old_intervals, 'OpenAI': 2.0}
                config_path.write_text(json.dumps(saved))
                with self.assertRaisesRegex(ValueError, 'beyond accepted'):
                    validate_source_configs(self.configs, [source])
                saved['execution']['provider_min_interval_seconds'] = old_intervals
                saved['execution']['timeout_seconds'] = 601
                config_path.write_text(json.dumps(saved))
                with self.assertRaisesRegex(ValueError, 'beyond accepted'):
                    validate_source_configs(self.configs, [source])
                changed = deepcopy(self.configs)
                changed[name]['execution']['provider_min_interval_seconds']['OpenAI'] = 1.0
                with self.assertRaisesRegex(ValueError, 'provider pacing'):
                    validate_configs(changed)

    def test_response_drift_stops_dispatch_and_blocks_resume(self):
        def drift(request, model, q):
            raw = mock_dispatch(request, model, q)
            body = json.loads(base64.b64decode(raw['body_base64']))
            body['provider'] = 'Unapproved fictional provider'
            raw['body_base64'] = base64.b64encode(json.dumps(body).encode()).decode()
            return raw
        result = self.collect(transport=drift)
        self.assertTrue(result['drift']); self.assertFalse(result['complete'])
        with self.assertRaisesRegex(ValueError, 'drift require human review'):
            self.collect(transport=lambda *args: self.fail('drifted campaign dispatched'))

    def test_one_shared_pool_and_one_active_call_per_original_query(self):
        # More than ten simultaneous targets across provider/model identities.
        config = self.configs['original']; prep = validate_input_manifest(self.public, self.questions)
        source = self.root/'many-original-failures'
        inventory = run_inventory(config, self.questions, self.public, config_path=self.config_paths['original'])
        manifest = make_manifest(config, self.questions, study_id='fictional-many-failures',
            experiment='E1+E2', provenance='mock', seed=config['execution']['seed'],
            keys_sha256=prep['keys_sha256'], preparation_sha256=prep['preparation_manifest_sha256'],
            inventory=inventory, content_sha256=prep.get('content_manifest_sha256'))
        run_package(manifest, self.questions, config, source, mock=True, inventory=inventory,
            input_manifest_path=self.public, config_path=self.config_paths['original'],
            transport=lambda request, model, q: mock_dispatch(request, model, q) if model['adapter'] == 'jev_gateway' else http_error(503))
        self.sources[0] = source
        lock = threading.Lock(); active = set(); peak = [0]; pools = []
        from concurrent.futures import ThreadPoolExecutor
        def executor(**kwargs):
            pools.append(kwargs['max_workers']); return ThreadPoolExecutor(**kwargs)
        def transport(request, model, q):
            key = (q.item_id, fingerprint(request))
            with lock:
                self.assertNotIn(key, active); active.add(key); peak[0] = max(peak[0], len(active))
            self.real_sleep(.005)
            with lock:
                active.remove(key)
            return mock_dispatch(request, model, q)
        with patch('benchmarking.retries.ThreadPoolExecutor', executor):
            self.assertTrue(self.collect(transport=transport)['complete'])
        self.assertEqual(pools, [10]); self.assertLessEqual(peak[0], 10)
        self.assertGreater(peak[0], 1)

    def test_reader_rejects_an_eleventh_start_and_restart_after_first_valid(self):
        self.collect()
        manifest, attempts, coverage = read_retry_package(self.output)
        target = attempts[0]['target_id']
        first = next(e for e in read_jsonl(self.output/'events.jsonl') if e['event'] == 'start' and e['target_id'] == target)
        journal = (self.output/'events.jsonl').read_bytes()
        for index in (2, 11):
            event = {k: v for k, v in first.items() if k not in ('event_sha256', 'previous_sha256')}
            event['retry_index'] = index
            append_event(self.output/'events.jsonl', event, coverage['last_event_sha256'])
            with self.assertRaisesRegex(ValueError, 'retry start|identity/index'):
                read_retry_package(self.output)
            (self.output/'events.jsonl').write_bytes(journal)

    def test_missing_all_attempt_measurements_stay_missing(self):
        from analysis.recovery import measurements
        rows = [{'billing': {'observed_usd': {'value': None}}},
                {'billing': {'observed_usd': {'value': '0.00'}}}]
        value = measurements(rows, 'billing', 'observed_usd', unresolved=1)
        self.assertIsNone(value['total']); self.assertEqual(value['known_sum'], '0.00')
        self.assertEqual(value['missing'], 2)

    def test_secondary_recoveries_never_replace_primary_decision_answers(self):
        config = self.configs['decisions']; prep = validate_input_manifest(self.public, self.questions)
        source = self.root/'luna-secondary-failures'; calls = [0]; lock = threading.Lock()
        def source_transport(request, model, q):
            with lock:
                calls[0] += 1; index = calls[0]
            return mock_dispatch(request, model, q) if index <= 3 else http_error(502)
        inventory = run_inventory(config, self.questions, self.public,
            config_path=self.config_paths['decisions'], protocol_path=self.protocol)
        manifest = make_manifest(config, self.questions, study_id='fictional-secondary-failures',
            experiment='E1+E2', provenance='mock', seed=config['execution']['seed'],
            keys_sha256=prep['keys_sha256'], preparation_sha256=prep['preparation_manifest_sha256'],
            inventory=inventory, content_sha256=prep.get('content_manifest_sha256'))
        run_package(manifest, self.questions, config, source, mock=True, inventory=inventory,
            input_manifest_path=self.public, config_path=self.config_paths['decisions'], transport=source_transport)
        self.sources[1] = source
        self.assertTrue(self.collect()['complete'])
        from analysis.recovery import recovery_report
        report = recovery_report(self.output, self.paths['evaluation_keys'])
        d01 = [r for r in report['responses'] if r['identity']['config_id'] == 'D01']
        primary = [r for r in d01 if r['role'] == 'primary']
        repeats = [r for r in d01 if r['role'] == 'secondary']
        self.assertEqual(len(primary), 3); self.assertEqual(len(repeats), 6)
        self.assertTrue(all(r['selected_retry_index'] is None and r['recovery_answer'] == r['original_answer'] for r in primary))
        self.assertTrue(all(r['selected_retry_index'] == 1 for r in repeats))


if __name__ == '__main__':
    unittest.main()
