"""Manual SAP 1.8 regression against captured pre-amendment source, fictional inputs only."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis.tests.test_expanded_sap import raw
from analysis.methods import expanded, pipeline
from analysis.support import notebook
from analysis.dataset.expanded import assemble_view
from benchmarking.configuration_ids import canonical_configuration, CURRENT_NAMESPACE


def reference_functions():
    """Load explicitly pinned test-only source; never discover historical releases."""
    fixture = Path(__file__).with_name('fixtures')/'provider_scalar_baseline.json'
    functions = json.loads(fixture.read_text())['functions']
    result = {}
    for name, module in [('stability', pipeline), ('analyze', pipeline),
                         ('secondary', expanded), ('figure_data', notebook)]:
        record = functions[name]
        source = record['source']
        if hashlib.sha256(source.encode()).hexdigest() != record['function_sha256']:
            raise AssertionError('Pre-amendment reference source changed: '+name)
        namespace = {**vars(module), 'math': math}
        exec(compile(source, str(fixture)+':'+name, 'exec'), namespace)
        result[name] = namespace[name]
    return result


def retained(value):
    """Independent expected-result projection for the old numerical schema."""
    if isinstance(value, dict):
        return {k: retained(v) for k, v in value.items()
                if k not in ('provider_scalar', 'provider_scalars') and not k.endswith('_change_separate')}
    if isinstance(value, list):
        return [retained(v) for v in value]
    return value


class ScalarRemoval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = reference_functions()
        configs = [{**canonical_configuration(CURRENT_NAMESPACE, cid),
                    'base_model': canonical_configuration(CURRENT_NAMESPACE, cid)['model_id']}
                   for cid in expanded.ROSTER]
        runs = [{'config_id': cid, 'replicate': rep, 'role': 'primary' if rep == 1 else 'secondary',
                 'experiment_id': 'E1' if rep == 1 else 'E2', 'run_id': f'{cid}-r{rep}'}
                for cid in expanded.ROSTER for rep in ((1, 2, 3) if cid in expanded.DECISIONS else (1,))]
        source = {'study_id': 'fictional-scalar-removal', 'benchmark': 'fictional', 'split': 'evaluation',
                  'items': [{'item_id': f'q{i}', 'labels': list('ABCD')} for i in range(6)],
                  'provenance': 'mock', 'bindings': {'questions': 'q', 'keys': 'k'},
                  'configurations': configs, 'runs': runs}
        rows = []
        for run in runs:
            cid, rep = run['config_id'], run['replicate']
            for i in range(6):
                row = raw(cid=cid, rep=rep, item=f'q{i}',
                          status='http_error' if i == 5 and rep == 2 else 'valid')
                if i == 1 or (i == 3 and rep == 3):
                    row['selected_label'] = 'B'
                if i == 2 and rep == 2:
                    row['probability']['native_vector'] = {'A': '.6', 'B': '.2', 'C': '.1', 'D': '.1'}
                if i == 4:
                    row['probability']['native_vector']['A'] = '.8'  # Ineligible sum; answer remains valid.
                row['probability']['concentration'] = str(rep/10)
                row['provider_confidence'] = {'value': str(rep/10), 'reason': None}
                row['provider_scalars'] = {'answers.answer.confidence': {
                    'value': str(rep/10), 'reason': None, 'semantics': 'fictional reviewed scalar'}}
                rows.append(row)
        cls.manifest, cls.rows, cls.coverage = assemble_view([(source, rows, {}, CURRENT_NAMESPACE)])
        cls.keys = {('fictional', 'evaluation', f'q{i}'): {'reference_answer': 'A'} for i in range(6)}
        cls.methods = {'normalization': 'native_selected_normalized_distribution',
            'resource_imputation': expanded.RESOURCE_POLICY, 'primary_family': 'decision_related_273',
            'decision_models': {cid: {'probability_semantics_verified': cid != 'D13',
                'scalar_fields': ['concentration'] if cid == 'D01' else ['provider_confidence'],
                'scalar_semantics': {'answers.answer.confidence': 'fictional reviewed scalar'}}
                for cid in expanded.DECISIONS}}
        arguments = (cls.manifest, cls.rows, cls.coverage, cls.keys)
        cls.inputs_before = deepcopy(arguments)
        cls.current = pipeline.analyze(*arguments, seed=20261008, methods=cls.methods)
        with patch.object(pipeline, 'stability', cls.reference['stability']), \
                patch.object(expanded, 'secondary', cls.reference['secondary']):
            cls.previous = cls.reference['analyze'](*arguments, seed=20261008, methods=cls.methods)
        cls.old_plots = cls.reference['figure_data'](cls.previous)
        cls.new_plots = notebook.figure_data(cls.current)

    def assert_no_scalars(self, value):
        text = json.dumps(value)
        for token in ('provider_scalar', 'provider_confidence', 'concentration',
                      'answers.answer.confidence', '_change_separate', 'Native scalar'):
            self.assertNotIn(token, text)

    def test_all_retained_numerical_results_and_plot_arrays_match_previous_source(self):
        expected = retained(self.previous)
        expected['method']['repeatability_scope'] = 'probability-only-20261012'
        self.assertEqual(self.current, expected)
        self.assertEqual((self.manifest, self.rows, self.coverage, self.keys), self.inputs_before)
        self.assert_no_scalars(self.current)
        self.assert_no_scalars(self.new_plots)
        self.assertTrue(any(k.endswith('_repeatability_provider_confidence') for k in self.old_plots))
        self.assertEqual(self.new_plots, {k: v for k, v in self.old_plots.items()
                                        if not k.endswith('_repeatability_provider_confidence')})
        from analysis.support.figures import presentation_plot_data
        before = presentation_plot_data(self.previous, self.old_plots, confidence_display='roc')
        after = presentation_plot_data(self.current, self.new_plots, confidence_display='roc')
        self.assertEqual(before, after)
        panel = after['repeatability_panel']
        self.assertEqual(set(panel), {'repeatability_confidence', 'repeatability_vector'})
        for curves in panel.values():
            self.assertEqual({(r['earlier_run'], r['later_run']) for r in curves}, {(1, 2), (1, 3), (2, 3)})
        self.assert_no_scalars(after)

    def test_scalar_values_and_review_state_cannot_change_retained_results(self):
        records = {pipeline.attempt_identity(r): pipeline.flat_record(r) for r in self.rows}
        altered = deepcopy(records)
        for row in altered.values():
            row['provider_scalars'] = {}
            for field in ('concentration', 'provider_confidence', 'answers.answer.confidence'):
                row['probability'][field] = None
                row['probability'][field+'_reason'] = 'unreviewed'
        methods = deepcopy(self.methods)
        for model in methods['decision_models'].values():
            model.update(scalar_fields=[], scalar_semantics={})
        actual = expanded.secondary(self.manifest, self.manifest['runs'], altered, self.keys,
                                    self.coverage, methods, 20261008)
        self.assertEqual(actual, self.current['decision_models'])
        self.assertEqual(actual['D13']['confidence']['eligible_n'], 0)
        self.assertEqual(actual['D13']['stability']['pairs'][0]['vector_eligible_n'], 0)
        self.assertGreater(actual['D13']['stability']['pairs'][0]['valid_pair_n'], 0)

    def historical_bundle(self):
        # Export requires the real expanded roster/count; this fixture tests the
        # reporting boundary only, with fictional saved statistics and no raw data.
        summary = deepcopy(self.previous)
        summary['coverage']['expected'] = 68742
        summary['missing_visual_sensitivity'] = {'accuracy': {}, 'paired': {'families': {}}}
        summary['inputs'] = {'retained_scalar_review': 'historical evidence'}
        reviews = {name: [] for name in ('answers', 'statuses', 'questions', 'secondary_answers')}
        reviews['selected_attempts'] = [{
            'identity': self.rows[0]['identity'], 'probability': deepcopy(self.rows[0]['probability']),
            'provider_scalars': deepcopy(self.rows[0]['provider_scalars']),
            'provider_confidence': deepcopy(self.rows[0]['provider_confidence'])}]
        return {'schema_version': 1, 'type': notebook.RESULT_TYPE, 'created_at': '2026-10-11',
            'summary': summary, 'review_tables': reviews, 'figure_data': deepcopy(self.old_plots),
            'statistical_release': {'bindings': {'sap': {'version': '1.7', 'sha256': 'b'*64},
                'settings': {'methods': deepcopy(self.methods)}}, 'approval': {'sha256': 'a'*64}}}

    def test_historical_render_notebook_rows_and_export_preserve_original_bundle(self):
        from analysis.support.figures import (export_current_report, presentation_plot_data,
                                              report_plot_values)
        original = self.historical_bundle()
        before = deepcopy(original)
        names = {cid: cid for cid in expanded.DECISIONS}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'historical.json'
            payload = json.dumps(original).encode()
            path.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            view = notebook.load_results(path, digest)
            self.assertEqual(view['statistical_release'], original['statistical_release'])
            self.assertEqual(view['summary']['inputs'], original['summary']['inputs'])
            self.assert_no_scalars(view['review_tables'])
            self.assert_no_scalars(view['figure_data'])
            tables = notebook.repeatability_tables(view['summary'], names)
            self.assertEqual(tables, notebook.repeatability_tables(original['summary'], names))
            self.assert_no_scalars(tables)
            self.assertEqual(len(tables['Pairwise changes']), 13*3*3)
            self.assertEqual(len(tables['Exact agreement across all three runs']), 13*3)
            self.assertFalse(any('field' in row for row in tables['Pairwise changes']))
            # Exercise the actual notebook display cell against unfiltered old data.
            import pandas as pd
            nb = json.loads((Path(__file__).parents[1]/'Analysis.ipynb').read_text())
            source = next(''.join(c['source']) for c in nb['cells']
                          if c.get('id') == '67e03116')
            displayed = []
            exec(compile(source, 'Analysis.ipynb:repeatability', 'exec'), {
                'SHOW_SECONDARY': True, 'summary': original['summary'], 'MODEL_NAMES': names,
                'repeatability_tables': notebook.repeatability_tables, 'pd': pd,
                'Markdown': str, 'display': displayed.append, 'figures': {}, 'families': {},
            })
            frames = [value.data for value in displayed if hasattr(value, 'data')]
            self.assertEqual(len(frames), 3)
            for frame in frames:
                self.assert_no_scalars(frame.to_json())
            info = notebook.render_result_info(view, path, digest, sap_version='1.8', methods={})
            self.assertEqual(info['saved_sap_version'], '1.7')
            self.assertFalse(info['sap_matches'])
            self.assert_no_scalars(presentation_plot_data(original['summary'], original['figure_data']))
            expected = report_plot_values(view['summary'], view['figure_data'])
            record = {'figures': {name: {'plotted_values': values} for name, values in expected.items()}}

            class Figure:
                def savefig(self, path, **kwargs):
                    Path(path).write_text('fictional graphic; no drawing needed for CSV regression')

            with patch('analysis.support.figures.presentation_code_record', return_value={'fixture': 'source'}):
                folder = export_current_report('analysis/output/export', path, digest,
                    {name: Figure() for name in expected}, record, root=root, source_before={'fixture': 'source'})
            for csv in folder.glob('*.csv'):
                with self.subTest(table=csv.name):
                    self.assert_no_scalars(csv.read_text())
            exported = json.loads((folder/'export_manifest.json').read_text())
            self.assertEqual(exported['results_sha256'], digest)
            self.assertEqual(exported['saved_sap']['version'], '1.7')
            self.assertEqual(exported['report_scope'], 'probability-only-20261012')
            self.assertTrue({'repeatability.csv', 'selected_attempts.csv'} <= set(exported['files']))
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(original, before)

    def test_new_bundle_and_selected_attempt_tables_exclude_scalars(self):
        from common import Option, Question
        questions = [Question('fictional', f'q{i}', 'evaluation', 'Fictional question',
                             tuple(Option(label, label) for label in 'ABCD')) for i in range(6)]
        reviews = notebook.response_tables(self.manifest, self.rows, self.coverage, self.keys, questions)
        self.assert_no_scalars(reviews)
        self.assertIn('native_vector', reviews['selected_attempts'][0]['probability'])
        self.assertEqual(self.rows, self.inputs_before[1])
        with tempfile.TemporaryDirectory() as root:
            result, path, digest = notebook.save_results(Path(root)/'new', self.current, reviews,
                {'sap': {'version': '1.8'}, 'settings': {'methods': self.methods}}, {'sha256': 'a'*64})
            self.assert_no_scalars(result['summary'])
            self.assert_no_scalars(result['review_tables'])
            self.assert_no_scalars(result['figure_data'])
            self.assertEqual(notebook.load_results(path, digest), result)
            self.assertEqual(result['statistical_release']['bindings']['settings']['methods'], self.methods)
