"""Manual fictional regressions for the versioned native Luna producer."""
import base64
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarking import adapters
from benchmarking.common import (ROOT, Question, Option, DECISIONS_MODEL, DECISIONS_MODEL_VERSION, config_errors, file_hash, fingerprint,
    load_config, read_package, write_new, runtime_blockers, validate_finish)
from benchmarking.run import (build_prompt, plan_counts, make_manifest, run_package,
    run_inventory, validate_input_manifest, mock_dispatch)


class DecisionsCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = load_config(ROOT / 'benchmarking/decisions.toml')
        self.config['execution']['study_registry'] = str(self.root / 'registry')
        self.config['execution']['provider_min_interval_seconds'] = {}
        self.model = self.config['models']['D01']
        # Unreviewed-policy fixtures must remain independent of the frozen live configuration.
        self.model.update(sum_tolerance='', precision_evidence='')
        self.q = Question('medqa', 'fictional-1', 'evaluation', 'Fictional symbol?',
            tuple(Option(k, 'Symbol ' + k) for k in 'ABCD'))
        self.protocol = self.root / 'protocol.md'
        self.protocol.write_text('# Fictional protocol\n')

    def tearDown(self):
        self.temp.cleanup()

    def raw(self, answer, **extra):
        body = {'model': self.model['version'], 'provider': 'OpenAI',
            'answers': {'answer': answer}, 'usage': {}, **extra}
        return {'status': 200, 'body_base64': base64.b64encode(json.dumps(body).encode()).decode(), 'headers': {}}

    def choice(self, **extra):
        return {'type': 'choice', 'choice': 'A',
            'probabilities': {'A': '.2', 'B': '.5', 'C': '.2', 'D': '.1'},
            'confidence': '.77', **extra}

    def test_native_request_has_no_generative_controls_or_reference_key(self):
        request = adapters.build_request(self.q, self.model, build_prompt(self.q, self.config))
        self.assertEqual(set(request), {'model', 'state', 'questions', 'provider'})
        self.assertEqual(list(request['questions']['answer']['criteria']), list('ABCD'))
        self.assertNotIn('JSON', request['state'])
        self.assertFalse(request['provider']['allow_fallbacks'])
        self.assertEqual(request['provider']['only'], ['openai'])

    def test_choice_not_argmax_and_confidence_not_concentration(self):
        self.model.update(sum_tolerance='0', precision_evidence='fictional exact values')
        result = adapters.parse(self.raw(self.choice()), self.model, self.q.labels)
        self.assertEqual(result['selected_label'], 'A')
        self.assertEqual(result['probability']['selected_probability'], '0.2')
        self.assertTrue(result['probability']['valid'])
        self.assertEqual(result['provider_confidence']['value'], '.77')
        self.assertIsNone(result['probability']['concentration'])

    def test_observed_response_version_is_pinned_without_changing_requested_alias(self):
        self.assertEqual(self.model['version'], DECISIONS_MODEL_VERSION)
        self.assertEqual(config_errors(self.config), [])
        request = adapters.build_request(self.q, self.model, build_prompt(self.q, self.config))
        self.assertEqual(request['model'], DECISIONS_MODEL)
        result = adapters.parse(self.raw(self.choice()), self.model, self.q.labels)
        self.assertEqual(result['status'], 'valid')
        self.assertEqual(result['returned_model'], DECISIONS_MODEL_VERSION)
        self.assertEqual(adapters.parse(self.raw(self.choice(), model=DECISIONS_MODEL),
                                        self.model, self.q.labels)['failure_reason'], 'returned_model_version_mismatch')
        self.model['version'] = DECISIONS_MODEL_VERSION.replace('20261006', '20261007')
        self.assertIn('exact reviewed Luna Decisions response version required', config_errors(self.config))

    def test_unreviewed_or_bad_vector_retains_accuracy_and_native_score(self):
        result = adapters.parse(self.raw(self.choice()), self.model, self.q.labels)
        self.assertEqual(result['status'], 'valid')
        self.assertEqual(result['probability']['reason'], 'sum_tolerance_unreviewed')
        self.assertEqual(result['probability']['selected_probability'], '0.2')
        self.model['sum_tolerance'] = '0'
        result = adapters.parse(self.raw(self.choice(probabilities={'A': '.2'})), self.model, self.q.labels)
        self.assertEqual(result['status'], 'valid')
        self.assertEqual(result['probability']['reason'], 'label_coverage')
        for vector in ({k: '0' for k in 'ABCD'}, {'A': True, 'B': '.3', 'C': '.3', 'D': '.4'}):
            result = adapters.parse(self.raw(self.choice(probabilities=vector)), self.model, self.q.labels)
            self.assertEqual(result['status'], 'valid')
            self.assertFalse(result['probability']['valid'])

    def test_exact_precision_boundary_and_arbitrary_finite_native_scalar(self):
        self.model['sum_tolerance'] = '0'
        vector = {'A': '0.20000000000000000000000000001', 'B': '.5', 'C': '.2', 'D': '.1'}
        result = adapters.parse(self.raw(self.choice(probabilities=vector, confidence='-1e400')), self.model, self.q.labels)
        self.assertFalse(result['probability']['valid'])
        self.assertEqual(result['probability']['reason'], 'sum_outside_tolerance')
        self.assertEqual(result['provider_confidence']['value'], '-1e400')
        self.model['sum_tolerance'] = '0.00000000000000000000000000001'
        self.assertTrue(adapters.parse(self.raw(self.choice(probabilities=vector)), self.model, self.q.labels)['probability']['valid'])

    def test_shared_tolerance_includes_decimal_boundaries_without_normalizing_native_score(self):
        self.model.update(sum_tolerance='0.02', precision_evidence='fictional shared eligibility policy')
        for last, expected in (('.08', True), ('.12', True),
                               ('.07999999999999999999999999999', False),
                               ('.12000000000000000000000000001', False)):
            with self.subTest(last=last):
                vector = {'A': '.2', 'B': '.5', 'C': '.2', 'D': last}
                result = adapters.parse(self.raw(self.choice(probabilities=vector)), self.model, self.q.labels)
                self.assertEqual(result['status'], 'valid')
                self.assertEqual(result['selected_label'], 'A')
                self.assertEqual(result['probability']['selected_probability'], '0.2')
                self.assertEqual(result['probability']['valid'], expected)
                if not expected:
                    self.assertEqual(result['probability']['reason'], 'sum_outside_tolerance')

    def test_refusal_unknown_schema_and_identity_drift(self):
        result = adapters.parse(self.raw({'type': 'refusal'}), self.model, self.q.labels)
        self.assertEqual(result['status'], 'refused')
        self.assertIsNone(result['selected_label'])
        self.assertEqual(adapters.parse(self.raw({'type': 'unexpected'}), self.model, self.q.labels)['status'], 'invalid')
        for field, value in (('model', 'wrong-model'), ('provider', 'wrong-provider')):
            self.assertEqual(adapters.parse(self.raw(self.choice(), **{field: value}), self.model, self.q.labels)['status'], 'drift')

    def fixture(self, experiment='E1+E2'):
        qpath = self.root / 'questions.jsonl'
        if not qpath.exists():
            write_new(qpath, [{'benchmark': 'medqa', 'split': 'evaluation', 'item_id': self.q.item_id,
                'stem': self.q.stem, 'options': [{'label': o.label, 'text': o.text} for o in self.q.options]}], jsonl=True)
        public = self.root / 'public.json'
        if not public.exists():
            write_new(public, {'schema_version': 2, 'type': 'inference_input', 'provenance': 'mock',
                'benchmark': 'medqa', 'split': 'evaluation', 'count': 1, 'questions_sha256': file_hash(qpath),
                'keys_sha256': '1' * 64, 'preparation_manifest_sha256': '2' * 64,
                'ordered_ids': [self.q.item_id], 'source_rows': [1]})
        inventory = run_inventory(self.config, qpath, public, study_id='fictional-luna',
            config_path=ROOT/'benchmarking/decisions.toml', protocol_path=self.protocol)
        manifest = make_manifest(self.config, qpath, study_id='fictional-luna', experiment=experiment,
            provenance='mock', seed=self.config['execution']['seed'], keys_sha256='1'*64,
            preparation_sha256='2'*64, inventory=inventory)
        return qpath, public, inventory, manifest

    def test_plan_and_package_roundtrip_repeat_barrier_resume_and_tampering(self):
        self.assertEqual(plan_counts(self.config)['total'], 3819)
        self.assertEqual(runtime_blockers(self.config), [])
        qpath, public, inventory, manifest = self.fixture()
        output = self.root/'attempts'
        seen = []
        def transport(request, model, question):
            seen.append(question.item_id)
            return mock_dispatch(request, model, question)
        args = dict(mock=True, inventory=inventory, input_manifest_path=public,
            config_path=ROOT/'benchmarking/decisions.toml', transport=transport)
        result = run_package(manifest, qpath, self.config, output, **args)
        self.assertTrue(result['complete'])
        saved, records, coverage = read_package(output)
        self.assertEqual(saved['schema_version'], 4)
        self.assertEqual([r['replicate'] for r in saved['runs']], [1, 2, 3])
        self.assertEqual(coverage['terminal'], 3)
        self.assertTrue(all('provider_confidence' in r for r in records))
        run_package(manifest, qpath, self.config, output, **args)
        self.assertEqual(len(seen), 3)
        forged = copy.deepcopy(records[0])
        forged['probability']['valid'] = True
        with self.assertRaisesRegex(ValueError, 'probability validity'):
            validate_finish(forged, saved)
        (output/'source/study-protocol.md').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'protocol'):
            read_package(output)

    def test_e2_without_registered_e1_and_unreviewed_live_evaluation_create_no_output(self):
        qpath, public, inventory, manifest = self.fixture(experiment='E2')
        with self.assertRaisesRegex(ValueError, 'complete registered E1'):
            run_package(manifest, qpath, self.config, self.root/'missing-primary', mock=True,
                inventory=inventory, input_manifest_path=public, config_path=ROOT/'benchmarking/decisions.toml')
        self.assertFalse((self.root/'missing-primary').exists())
        # Full-cardinality fictional inputs reach the precision gate before
        # any output, registration or transport call can be created.
        questions = [{'benchmark': 'medqa', 'split': 'evaluation', 'item_id': f'fictional-{i}',
            'stem': f'Fictional symbol {i}?', 'options': [{'label': k, 'text': f'Symbol {k}'} for k in 'ABCD']}
            for i in range(1273)]
        qpath.write_text(''.join(json.dumps(q) + '\n' for q in questions))
        public_data = json.loads(public.read_text())
        public_data.update(provenance='prepared', count=1273, questions_sha256=file_hash(qpath),
            ordered_ids=[q['item_id'] for q in questions], source_rows=list(range(1,1274)))
        public.write_text(json.dumps(public_data))
        with patch.dict('os.environ', {'OPENROUTER_API_KEY': 'fictional-not-a-key'}), patch('benchmarking.adapters.dispatch') as dispatch:
            inventory = run_inventory(self.config, qpath, public, protocol_path=self.protocol,
                config_path=ROOT/'benchmarking/decisions.toml')
            manifest = make_manifest(self.config, qpath, study_id='fictional-live', experiment='E1+E2',
                provenance='definitive', seed=self.config['execution']['seed'], keys_sha256='1'*64,
                preparation_sha256='2'*64, inventory=inventory)
            with self.assertRaisesRegex(ValueError, 'precision'):
                run_package(manifest, qpath, self.config, self.root/'unreviewed-live', mock=False,
                    inventory=inventory, input_manifest_path=public)
            dispatch.assert_not_called()
        self.assertFalse((self.root/'unreviewed-live').exists())


if __name__ == '__main__':
    unittest.main()
