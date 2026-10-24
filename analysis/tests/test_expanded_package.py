"""Manual schema-6 package-to-analysis regression with fictional inputs only.

Use the producer's request, response and journal writers, deterministic timestamps
and the real saved-package reader. No scheduler, transport or research data runs.
Only the historical six-file loader is stubbed; its verification has separate tests.
"""
import base64
from contextlib import chdir
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch

from benchmarking.common import (ROOT, Option, Question, artifact_ref, file_hash,
    fingerprint, question_record, write_new)
from benchmarking.configuration_ids import (MAPPINGS, ORIGINAL_NAMESPACE,
    LUNA_NAMESPACE, canonical_configuration)
from benchmarking.extension import CONFIG, prepare, save_response
from benchmarking.extension_contract import Ledger, SCHEMA, SOURCES, attempt_id
from benchmarking.run import append_event
from analysis.dataset.expanded import decision_settings, load_expanded_inputs
from analysis.dataset.inputs import folder_inventory
from analysis.methods.expanded import RESOURCE_POLICY
from analysis.methods.pipeline import analyze
from analysis.tests.test_expanded_sap import raw


def synthetic_package(root):
    """Two questions, all 11 extension models/three runs, recovery and exhaustion."""
    config_path = root/'config.toml'
    config_path.write_text(CONFIG.read_text().replace('reviewed = false', 'reviewed = true')
        .replace('meaning = "unreviewed"', 'meaning = "fictional option probability"')
        .replace('evidence = ""', 'evidence = "synthetic test evidence only"'))
    config = tomllib.loads(config_path.read_text())
    questions = [Question('fictional', f'q{i}', 'evaluation', f'Fictional question {i}',
        tuple(Option(label, 'Option '+label) for label in 'ABCD')) for i in range(2)]
    question_path = root/'questions.jsonl'
    write_new(question_path, [question_record(q) for q in questions], jsonl=True)
    keys = {('fictional', 'evaluation', q.item_id): {'reference_answer': 'A' if i == 0 else 'B'}
            for i, q in enumerate(questions)}
    public_path = root/'inference.json'
    write_new(public_path, {'schema_version': 2, 'type': 'inference_input', 'provenance': 'mock',
        'benchmark': 'fictional', 'split': 'evaluation', 'count': len(questions),
        'questions_sha256': file_hash(question_path), 'keys_sha256': fingerprint(list(keys.values())),
        'preparation_manifest_sha256': 'b'*64, 'ordered_ids': [q.item_id for q in questions],
        'source_rows': [1, 2]})
    protocol_path = root/'protocol.md'
    protocol_path.write_text('Fictional schema-6 interface fixture. No scientific approval.')
    output = root/'benchmarking/output/fictional-extension'
    manifest, inventory, snapshots, _, requests = prepare(config, config_path, question_path,
        public_path, protocol_path, output, mock=True)
    output.mkdir(parents=True)
    for name, path in {**{name: ROOT/name for name in SOURCES}, **snapshots}.items():
        destination = output/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    write_new(output/'manifest.json', manifest)
    write_new(output/'provenance.json', inventory)
    ledger = Ledger(manifest)

    def event(data):
        base = {'schema_version': SCHEMA, 'manifest_sha256': fingerprint(manifest), **data}
        row = {**base, 'previous_sha256': ledger.previous}
        row['event_sha256'] = fingerprint(row)
        previous = ledger.previous
        ledger.apply(row, output)
        append_event(output/'events.jsonl', base, previous)

    event({'event': 'plan'})
    epoch = datetime(2026, 10, 11, tzinfo=timezone.utc)
    sequence = 0
    while ledger.candidates():
        tid = ledger.candidates()[0]
        target = ledger.targets[tid]
        cid, item = target['identity']['config_id'], target['identity']['item_id']
        model = config['models'][cid]
        index = ledger.counts.get(tid, 0)
        lineage = {'original_query_id': tid, 'attempt_index': index,
            'attempt_id': attempt_id(tid, index),
            'parent_attempt_id': attempt_id(tid, index-1) if index else None}
        adir = output/'artifacts'/tid/str(index)
        write_new(adir/'request.json', requests[cid, item])
        ref = artifact_ref(adir/'request.json', output)
        start = epoch + timedelta(seconds=10*sequence)
        sequence += 1
        event({'event': 'start', **lineage, 'provider': model['route'],
               'request': ref, 'started_at': start.isoformat()})
        recovering = cid == 'D03' and item == 'q0' and target['replicate'] == 1
        failed = (recovering and index == 0) or (cid == 'D13' and item == 'q0')
        duration = 8 if recovering and index == 0 else 2
        cost = 9 if recovering and index == 0 else 0.4
        body = {'model': model['expected_returned_model'], 'provider': model['route'],
            'id': lineage['attempt_id'],
            'answers': {'answer': {'type': 'choice', 'choice': 'B' if recovering else 'A',
                'probabilities': {'A': 0.7, 'B': 0.1, 'C': 0.1, 'D': 0.1}, 'confidence': 17}},
            'usage': {'input_tokens': 10, 'output_tokens': 0, 'cost': cost}}
        response = {'status': 503 if failed else 200, 'headers': {},
                    'body_base64': base64.b64encode(json.dumps(body).encode()).decode()}
        result = {'raw': response, 'failure': None, 'error': None,
            'started_at': start.isoformat(), 'finished_at': (start+timedelta(seconds=duration)).isoformat(),
            'duration_seconds': duration, 'queue_wait_seconds': 0}
        record = save_response(output, adir, ref, result, model, target, lineage)
        record['scheduler'] = {'provider': model['route'], 'response_status': response['status'],
                               'stop_reason': None}
        event({'event': 'finish', **lineage, 'record': record})
    return output, manifest, questions, keys


def historical_stub(extension, questions, keys):
    """Supply the same fictional items under the Original and Luna namespaces."""
    sources, rows = [], []
    for namespace, workflow in ((ORIGINAL_NAMESPACE, 'lean-v1'), (LUNA_NAMESPACE, 'lean-decisions-v1')):
        configs, runs = [], []
        for source_id, cid in MAPPINGS[namespace].items():
            config = canonical_configuration(namespace, source_id)
            configs.append({**config, 'config_id': source_id, 'base_model': config['model_id']})
            for rep in ((1, 2, 3) if cid.startswith('D') else (1,)):
                run_id = f'{namespace}-{source_id}-r{rep}'
                runs.append({'config_id': source_id, 'run_id': run_id, 'replicate': rep,
                    'role': 'primary' if rep == 1 else 'secondary',
                    'experiment_id': 'E1' if rep == 1 else 'E2'})
                for q in questions:
                    row = raw(cid=source_id, item=q.item_id, rep=rep)
                    row['identity']['run_id'] = run_id
                    rows.append(row)
        sources.append({'study_id': namespace, 'execution_workflow': workflow, 'provenance': 'mock',
            'benchmark': 'fictional', 'split': 'evaluation', 'items': deepcopy(extension['items']),
            'bindings': {k: extension['bindings'][k] for k in ('questions', 'keys')},
            'configurations': configs, 'runs': runs})
    coverage = {'expected': len(rows), 'terminal': len(rows), 'complete': True,
                'unstarted': [], 'indeterminate': []}
    return ({'source_manifests': sources}, rows, coverage, keys, questions,
            {'input_manifest.json': 'a'*64})


class ExpandedPackageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root/'analysis').mkdir()
        self.package, manifest, questions, keys = synthetic_package(self.root)
        self.historical = historical_stub(manifest, questions, keys)
        self.selection = {'path': str(self.package.relative_to(self.root)),
                          'inventory_sha256': fingerprint(folder_inventory(self.package))}

    def load(self, selections=None):
        with patch('analysis.dataset.expanded.load_input_folder', return_value=deepcopy(self.historical)):
            return load_expanded_inputs(self.root/'analysis/input', 'a'*64,
                [self.selection] if selections is None else selections, repository_root=self.root)

    def test_package_reader_to_analysis_preserves_selection_mapping_and_resources(self):
        before = folder_inventory(self.package)
        manifest, records, coverage, keys, _, _ = self.load()
        self.assertTrue(coverage['complete'])
        self.assertEqual(coverage['terminal'], 108)  # Two questions x 54 planned runs.
        self.assertEqual(len(manifest['configurations']), 28)
        self.assertEqual({(r['source_namespace'], r['source_identity']['config_id'])
            for r in records if r['identity']['config_id'] in ('D01', 'D02')},
            {(ORIGINAL_NAMESPACE, 'J01'), (LUNA_NAMESPACE, 'D01')})
        recovered = next(r for r in records if r['identity']['config_id'] == 'D03'
                         and r['identity']['item_id'] == 'q0' and r['attempt_index'] == 1)
        self.assertEqual(recovered['selected_label'], 'B')  # Valid wrong answer remains final.
        self.assertEqual(recovered['analysis_selection']['outcome'], 'recovered')
        self.assertEqual(recovered['analysis_selection']['cost_attempt_id'], recovered['attempt_id'])
        self.assertEqual(recovered['analysis_selection']['duration_attempt_id'], recovered['attempt_id'])
        exhausted = [r for r in records if r['identity']['config_id'] == 'D13'
                     and r['identity']['item_id'] == 'q0']
        self.assertEqual(len(exhausted), 3)
        self.assertTrue(all(r['attempt_index'] == 10 and
            r['analysis_selection']['outcome'] == 'exhausted' for r in exhausted))
        methods = {'normalization': 'native_selected_normalized_distribution',
            'primary_family': 'decision_related_273', 'resource_imputation': RESOURCE_POLICY,
            'decision_models': decision_settings(manifest)}
        summary = analyze(manifest, records, coverage, keys, seed=20261008, methods=methods)
        self.assertTrue(summary['study_complete'])
        self.assertEqual(len(summary['paired']['families']['S1']), 273)
        self.assertEqual(len(summary['paired']['families']['S2']), 3)
        self.assertEqual(len(summary['repeat_operations']), 26)
        self.assertEqual(summary['primary']['D03']['accuracy'], 0)
        self.assertEqual(summary['primary']['D13']['accuracy'], 0)
        op = summary['primary']['D03']['operations']
        self.assertAlmostEqual(op['analysis_usd']['total'], 0.8)
        self.assertEqual(op['analysis_latency']['total_seconds'], 4)
        self.assertEqual(op['retry_calls_n'], 1)
        self.assertEqual(summary['decision_models']['D03']['confidence']['eligible_n'], 2)
        self.assertEqual(summary['decision_models']['D13']['confidence']['eligible_n'], 1)
        self.assertEqual(folder_inventory(self.package), before)

    def test_relative_and_absolute_paths_work_from_repository_and_notebook_directory(self):
        absolute = {**self.selection, 'path': str(self.package)}
        baseline = self.load()
        for directory in (self.root, self.root/'analysis'):
            with self.subTest(directory=directory), chdir(directory):
                self.assertEqual(self.load(), baseline)
                self.assertEqual(self.load([absolute]), baseline)
                with self.assertRaisesRegex(ValueError, 'Duplicate extension package'):
                    self.load([self.selection, absolute])

    def test_real_reader_rejects_tampering_even_with_refreshed_outer_digest(self):
        _, records, _, _, _, _ = self.load()
        extension_record = next(r for r in records if r['identity']['config_id'] == 'D03')
        artifact = self.package/extension_record['artifacts']['raw_response']['path']
        artifact.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'independent inventory digest'):
            self.load()
        refreshed = {**self.selection, 'inventory_sha256': fingerprint(folder_inventory(self.package))}
        with self.assertRaises(ValueError):
            self.load([refreshed])

    def test_interrupted_last_retry_remains_indeterminate(self):
        journal = self.package/'events.jsonl'
        lines = journal.read_text().splitlines()
        last = json.loads(lines[-1])
        self.assertEqual(last['event'], 'finish')
        self.assertEqual(last['attempt_index'], 10)
        journal.write_text('\n'.join(lines[:-1])+'\n')
        selection = {**self.selection, 'inventory_sha256': fingerprint(folder_inventory(self.package))}
        _, records, coverage, _, _, _ = self.load([selection])
        identity = last['record']['identity']
        self.assertFalse(coverage['complete'])
        self.assertEqual(coverage['terminal'], 107)
        self.assertEqual(len(coverage['indeterminate']), 1)
        self.assertFalse(any(r['identity'] == identity for r in records))


class DecisionFieldReviewTests(unittest.TestCase):
    """Fictional review evidence cannot enable a different collection or field."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)/'review.json'
        model = {'model_id': 'fictional/model', 'expected_returned_model': 'fictional/version',
                 'route': 'Fictional', 'surface': 'systemone',
                 'scalar_semantics': {'answers.answer.confidence': 'unreviewed'}}
        bound = {'execution_workflow': 'decision-extension-v1', 'models': {'D03': model}}
        self.manifest = {'configurations': [{'config_id': 'D03', 'model_id': 'fictional/model',
            'probability_semantics': {'reviewed': False}, 'scalar_semantics': model['scalar_semantics']}],
            'source_packages': [{'manifest': bound, 'manifest_sha256': fingerprint(bound)}]}
        self.hashes = {'extension:'+'a'*64: 'a'*64}
        self.review = {'schema_version': 1, 'review_id': 'fictional-review',
            'reviewed_at': '2026-10-12', 'reviewer': 'Fictional reviewer',
            'source_record': 'fictional-evidence', 'package_manifest_sha256': fingerprint(bound),
            'package_inventory_sha256': 'a'*64, 'models': {'D03': {
                'model_id': model['model_id'], 'returned_model': model['expected_returned_model'],
                'provider': model['route'], 'surface': model['surface'],
                'source_model_sha256': fingerprint(model), 'labels': list('ABCD'),
                'vector_field': 'answers.answer.probabilities',
                'selected_label_field': 'answers.answer.choice',
                'probability_semantics': {'reviewed': True,
                    'meaning': 'native_label_indexed_choice_probability', 'evidence': 'Fictional field evidence'},
                'sources': ['https://example.invalid/fictional'],
                'development_evidence': {'fixture': 'fictional'},
                'scalar_semantics': dict(model['scalar_semantics'])}}}

    def settings(self, review=None, manifest=None, hashes=None):
        self.path.write_text(json.dumps(self.review if review is None else review))
        return decision_settings(self.manifest if manifest is None else manifest,
            field_review_path=self.path, input_hashes=self.hashes if hashes is None else hashes)

    def test_review_enables_only_probabilities_and_preserves_source(self):
        before = deepcopy(self.manifest)
        baseline = decision_settings(self.manifest)
        reviewed = self.settings()['D03']
        self.assertFalse(baseline['D03']['probability_semantics_verified'])
        self.assertTrue(reviewed['probability_semantics_verified'])
        self.assertEqual(reviewed['scalar_semantics']['answers.answer.confidence'], 'unreviewed')
        self.assertEqual(reviewed['field_review_evidence']['review_file_sha256'], file_hash(self.path))
        self.assertEqual(self.manifest, before)
        self.assertEqual(decision_settings(self.manifest), baseline)

    def test_analysis_validates_the_same_bound_field_review(self):
        methods = {'normalization': 'native_selected_normalized_distribution',
            'resource_imputation': RESOURCE_POLICY, 'decision_models': self.settings()}
        manifest = {**self.manifest, 'execution_workflow': 'analysis-reporting-ids-v1',
                    'provenance': 'definitive'}

        class ReviewAccepted(Exception):
            """Stop before numerical analysis; this fixture tests the review gate."""

        # Keep the real review reader and integrity checks. Stub unrelated manifest
        # validation and stop at the next stage to isolate this boundary.
        with patch('analysis.methods.pipeline.validate_manifest'), patch(
                'analysis.methods.pipeline.roster_check', side_effect=ReviewAccepted) as next_stage:
            with self.assertRaises(ReviewAccepted):
                analyze(manifest, [], {}, {}, seed=20261008, methods=methods,
                        field_review_path=self.path, input_hashes=self.hashes)
            next_stage.assert_called_once_with(manifest)
            next_stage.reset_mock()
            with self.assertRaisesRegex(ValueError, 'settings differ'):
                analyze(manifest, [], {}, {}, seed=20261008, methods=methods)
            with self.assertRaisesRegex(ValueError, 'verified package inventory'):
                analyze(manifest, [], {}, {}, seed=20261008, methods=methods,
                        field_review_path=self.path, input_hashes={})
            altered = deepcopy(methods)
            altered['decision_models']['D03']['probability_semantics_verified'] = False
            with self.assertRaisesRegex(ValueError, 'settings differ'):
                analyze(manifest, [], {}, {}, seed=20261008, methods=altered,
                        field_review_path=self.path, input_hashes=self.hashes)
            next_stage.assert_not_called()

    def test_wrong_package_or_missing_evidence_fails(self):
        variants = []
        for field, value in [('package_manifest_sha256', 'b'*64), ('package_inventory_sha256', 'b'*64)]:
            review = deepcopy(self.review); review[field] = value; variants.append(review)
        for field, value in [('model_id', 'fictional/other'), ('provider', 'Other'),
                ('source_model_sha256', 'b'*64), ('labels', list('ABCE')),
                ('vector_field', 'answers.answer.confidence'), ('development_evidence', {}), ('sources', [])]:
            review = deepcopy(self.review); review['models']['D03'][field] = value; variants.append(review)
        review = deepcopy(self.review); review['models']['D03']['probability_semantics']['evidence'] = ''
        variants.append(review)
        for review in variants:
            with self.subTest(review=review), self.assertRaises(ValueError):
                self.settings(review)
        with self.assertRaises(ValueError):
            self.settings(hashes={})

    def test_unreviewed_and_historical_only_remain_unreviewed(self):
        review = deepcopy(self.review)
        review['models']['D03']['probability_semantics']['reviewed'] = False
        self.assertFalse(self.settings(review)['D03']['probability_semantics_verified'])
        manifest = deepcopy(self.manifest); manifest['source_packages'] = []
        self.assertEqual(self.settings(manifest=manifest), decision_settings(manifest))

    def test_scalar_requires_separate_evidence(self):
        # Evidence integrity remains enforced although scalars are no longer analyzed.
        review = deepcopy(self.review); model = review['models']['D03']
        field = 'answers.answer.confidence'; meaning = 'fictional_scalar_meaning'
        model['scalar_semantics'][field] = meaning
        with self.assertRaises(ValueError):
            self.settings(review)
        model['scalar_review_evidence'] = {field: {'reviewed': True, 'meaning': meaning,
            'evidence': 'Fictional scalar evidence', 'source': 'https://example.invalid/scalar'}}
        self.assertEqual(self.settings(review)['D03']['scalar_semantics'][field], meaning)
        self.assertEqual(self.manifest['configurations'][0]['scalar_semantics'][field], 'unreviewed')
        self.assertTrue(self.settings(review)['D03']['probability_semantics_verified'])
        unreviewed = deepcopy(review)
        unreviewed['models']['D03']['scalar_semantics'][field] = 'unreviewed'
        unreviewed['models']['D03'].pop('scalar_review_evidence')
        self.assertTrue(self.settings(unreviewed)['D03']['probability_semantics_verified'])
