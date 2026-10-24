"""Manual synthetic regressions for SAP 1.8. Never reads research inputs."""
from copy import deepcopy
import unittest
import numpy as np

from benchmarking.configuration_ids import (canonical_configuration, CURRENT_NAMESPACE,
    ORIGINAL_NAMESPACE, LUNA_NAMESPACE)
from common import measurement, empty_probability
from analysis.dataset.expanded import assemble_view, select_extension, validate_mapped_record
from analysis.methods.expanded import ROSTER, DECISIONS, PAIRS, contrasts, paired, impute_resources, secondary
from analysis.methods.pipeline import flat_record, run_operations
from analysis.methods.statistics import holm, bootstrap_differences


def raw(cid='D03', item='q0', rep=1, index=0, status='valid', cost='2', duration='4'):
    return {'identity': {'benchmark': 'fictional', 'split': 'evaluation', 'item_id': item,
            'config_id': cid, 'run_id': f'{cid}-r{rep}'}, 'attempt_index': index,
        'attempt_id': f'{cid}-{item}-{rep}-{index}', 'status': status,
        'selected_label': 'A' if status == 'valid' else None,
        'probability': {**empty_probability('fictional'), 'native_vector': {'A': '.7','B': '.1','C': '.1','D': '.1'}},
        'usage': {**{k: measurement(None, 'unavailable') for k in ('input_tokens','output_tokens','reasoning_tokens','final_tokens','billed_tokens')},
                  'accounting_definition': 'fictional', 'reasoning_in_output': False},
        'billing': {'observed_usd': measurement(cost, 'missing' if cost is None else None),
                    'estimated_usd': measurement(None, 'unavailable'), 'rate_provenance': 'fixture', 'rate_date': None},
        'timing': {'duration_seconds': measurement(duration, 'missing' if duration is None else None),
                  'started_at': '2026-10-11T10:00:00+00:00', 'finished_at': '2026-10-11T10:00:04+00:00'},
        'route': {'returned_route': 'fixture'}}


class ExpandedSAP(unittest.TestCase):
    def test_namespaces_disambiguate_d01_and_reject_mismatch(self):
        self.assertEqual(canonical_configuration(LUNA_NAMESPACE, 'D01')['config_id'], 'D02')
        self.assertEqual(canonical_configuration(CURRENT_NAMESPACE, 'D01')['model_id'], 'typesafe/jev-1.13')
        self.assertEqual(canonical_configuration(ORIGINAL_NAMESPACE, 'C25', effort='none')['config_id'], 'G15')
        for ns,cid,model in [(None,'D01',None),(ORIGINAL_NAMESPACE,'D01',None),
                             (CURRENT_NAMESPACE,'D01','openai/gpt-6-luna-decisions')]:
            with self.assertRaises(ValueError): canonical_configuration(ns,cid,model_id=model)

    def test_complete_families_orientation_and_holm(self):
        cs = contrasts()
        self.assertEqual(sum(c[0]=='S1' for c in cs),273)
        self.assertEqual(sum(c[0]=='S2' for c in cs),3)
        self.assertEqual(len({(a,b) for _,_,a,b in cs}),276)
        self.assertIn(('S1','D01-D13','D01','D13'),cs)
        self.assertIn(('S2','P09','G10','G09'),cs)
        matrix = {cid: np.array([1,0,1,0]) for cid in ROSTER}
        configs = {cid: {'effort':'native'} for cid in ROSTER}
        result = paired(matrix,configs,PAIRS,seed=20261008)
        self.assertEqual(len(result['families']['S1']),273)
        self.assertTrue(all(r['p_holm']==1 for family in result['families'].values() for r in family))
        self.assertAlmostEqual(holm([.0001]+[1]*272,273)[0], .0273)
        with self.assertRaises(ValueError): paired({k:v for k,v in matrix.items() if k!='D13'},configs,PAIRS,seed=20261008)

    def test_shared_draw_digest_and_intervals_survive_contrast_blocks(self):
        d=np.tile(np.array([[1],[0],[-1],[1]]),(1,33))
        many=bootstrap_differences(d,seed=20261008)
        one=bootstrap_differences(d[:,:1],seed=20261008)
        self.assertEqual(many['intervals_pp'],one['intervals_pp']*33)
        self.assertEqual(many['sampled_question_indices_sha256'],one['sampled_question_indices_sha256'])

    def test_select_wrong_first_valid_exhaustion_pending_and_extra_retry(self):
        trials=[raw(index=0,status='http_error'),raw(index=1)]
        trials[1]['selected_label']='B'  # No key is consulted.
        selected=select_extension(trials,{},'fixture')
        self.assertEqual(selected[0]['selected_label'],'B')
        self.assertEqual(selected[0]['analysis_selection']['cost_attempt_id'],trials[1]['attempt_id'])
        failed=[raw(index=i,status='http_error') for i in range(11)]
        self.assertEqual(select_extension(failed,{},'fixture')[0]['attempt_index'],10)
        self.assertEqual(select_extension(failed[:10],{},'fixture'),[])
        with self.assertRaises(ValueError):select_extension(trials+[raw(index=2)],{},'fixture')

    def test_selected_final_duration_and_common_donor_pools_preserve_raw(self):
        history=[raw(index=0,status='http_error',cost='99',duration='999'),raw(index=1,cost='0',duration='0')]
        chosen=select_extension(history,{},'fixture')[0]
        missing=raw(item='q1',status='http_error',cost=None,duration=None)
        failed=raw(item='q2',status='http_error',cost='4',duration='8')
        rows=[flat_record(x) for x in (chosen,missing,failed)]
        before=deepcopy(rows)
        op=run_operations(rows,1,selected_final=True)
        self.assertEqual(op['latency']['observed_sum_seconds'],8)
        summary={'D03':{'run_id':'D03-r1','complete':True,'planned_n':3,'correct':1,'operations':op}}
        impute_resources(summary,{'D03':rows})
        self.assertEqual(rows,before)
        cost=op['analysis_usd'];duration=op['analysis_latency']
        self.assertEqual(cost['total'],6)
        self.assertEqual(cost['imputation']['donor_mean_exact'],'2')
        self.assertEqual(duration['total_seconds'],12)
        self.assertEqual(duration['imputation']['donor_n'],2)
        self.assertFalse(op['observed_usd']['complete'])
        self.assertEqual(cost['imputation']['assignments'][0]['identity']['item_id'],'q1')

    def test_no_donors_or_incomplete_run_cannot_be_imputed_complete(self):
        missing=flat_record(raw(cost=None,duration=None))
        donor=flat_record(raw(item='q1',cost='2',duration='4'))
        for case,rows,complete,planned_n in (
            ('no donors',[missing],True,1),
            ('unstarted run',[],False,1),
            ('incomplete run with donor',[missing,donor],False,3),
        ):
            with self.subTest(case=case):
                before=deepcopy(rows)
                op=run_operations(rows,1,complete_ledger=complete,selected_final=True)
                summary={'D03':{'run_id':'D03-r1','complete':complete,'planned_n':planned_n,
                                'correct':1 if complete else None,'operations':op}}
                impute_resources(summary,{'D03':rows})
                self.assertEqual(rows,before)
                self.assertIsNone(op['analysis_usd']['total'])
                self.assertIsNone(op['analysis_latency']['total_seconds'])
                for metric in ('analysis_usd','analysis_latency'):
                    self.assertFalse(op[metric]['complete'])
                    self.assertEqual(op[metric]['imputed_n'],0)
                    self.assertEqual(op[metric]['imputation']['assignments'],[])
                self.assertEqual(op['analysis_latency']['median'],4 if rows and donor in rows else None)

    def test_common_vector_boundary_review_gate_and_failure_denominators(self):
        from analysis.methods.decisions import native_vector
        runs=[];records={};keys={};items=[{'item_id':f'q{i}'} for i in range(3)]
        for cid in DECISIONS:
            for rep in (1,2,3):
                runs.append({'config_id':cid,'replicate':rep,'role':'primary' if rep==1 else 'secondary','run_id':f'{cid}-r{rep}'})
                for i in range(3):
                    row=raw(cid=cid,item=f'q{i}',rep=rep,status='http_error' if i==2 else 'valid')
                    if i==1:row['probability']['native_vector']={'A':'.72','B':'.1','C':'.1','D':'.1'}
                    row['provider_scalars']={'unreviewed.extra':{'value':'0.5','reason':None,'semantics':'unreviewed'}}
                    flat=flat_record(row);records[tuple(flat['identity'][k] for k in ('benchmark','split','item_id','config_id','run_id'))]=flat
                    keys[('fictional','evaluation',f'q{i}') ]={'reference_answer':'A' if i==0 else 'B'}
        manifest={'benchmark':'fictional','split':'evaluation','items':items,'provenance':'mock'}
        methods={'decision_models':{cid:{'probability_semantics_verified':True} for cid in DECISIONS}}
        methods['decision_models']['D13']['probability_semantics_verified']=False
        result=secondary(manifest,runs,records,keys,{'unstarted':[],'indeterminate':[]},methods,20261008)
        for cid in DECISIONS[:-1]:
            self.assertEqual(result[cid]['confidence']['eligible_n'],2)
            stability=result[cid]['stability']
            self.assertEqual(stability['three_way']['valid_triple_n'],2)
            self.assertEqual(sum(stability['pairs'][0]['transitions'].values()),3)
            self.assertEqual(stability['runs']['1']['accuracy'],1/3)
        for cid in DECISIONS:
            self.assertNotIn('provider_scalars', result[cid]['stability'])
            self.assertNotIn('provider_scalar', result[cid]['stability']['confidence_three_way'])
        self.assertEqual(result['D13']['confidence']['eligible_n'],0)
        self.assertEqual(result['D13']['stability']['three_way']['valid_triple_n'],2)
        row=flat_record(raw());row['probability']['native_vector']={'A':'.720000000000001','B':'.1','C':'.1','D':'.1'}
        self.assertEqual(native_vector(row,sum_tolerance='.02')[1],'sum_outside_tolerance')

    def test_missing_models_are_planned_and_cannot_be_saved_complete(self):
        c=canonical_configuration(ORIGINAL_NAMESPACE,'J01')
        source={'benchmark':'fictional','split':'evaluation','items':[{'item_id':'q0'}], 'provenance':'mock',
            'bindings':{'questions':'q','keys':'k'},'configurations':[{**c,'config_id':'J01'}],
            'runs':[{'config_id':'J01','replicate':1,'role':'primary','experiment_id':'E1','run_id':'J01-r1'}]}
        view,rows,coverage=assemble_view([(source,[raw(cid='J01')],{},ORIGINAL_NAMESPACE)])
        self.assertFalse(coverage['complete']);self.assertEqual(coverage['expected'],54)
        self.assertEqual(len(view['missing_configurations']),27)
        validate_mapped_record(rows[0],view)
        rows[0]['selected_label']='B'
        with self.assertRaises(ValueError):validate_mapped_record(rows[0],view)
        from analysis.support.notebook import save_results
        with self.assertRaisesRegex(ValueError,'incomplete'):
            save_results('unused',{'study_complete':False,'primary_complete':False},{},{'fixture':1},{'sha256':'a'*64})

    def test_active_figure_selection_excludes_removed_scalar_graphics(self):
        from analysis.support.figures import current_report_figures, FIGURE_SETTINGS
        names=current_report_figures({'configuration_namespace':CURRENT_NAMESPACE,'decision_models':dict.fromkeys(DECISIONS)})
        self.assertFalse(any('provider_confidence' in n for n in names))
        self.assertEqual(names, ['accuracy', 'resource_panel', 'confidence_panel', 'repeatability_panel'])
        self.assertFalse(FIGURE_SETTINGS['rc_params']['axes.formatter.use_locale'])


    def test_expanded_analysis_integration_retains_failed_accuracy_and_secondary_runs(self):
        from analysis.methods.pipeline import analyze
        from analysis.methods.expanded import RESOURCE_POLICY
        configs=[{**canonical_configuration(CURRENT_NAMESPACE,cid),
                  'base_model':canonical_configuration(CURRENT_NAMESPACE,cid)['model_id']} for cid in ROSTER]
        runs=[{'config_id':cid,'replicate':rep,'role':'primary' if rep==1 else 'secondary',
               'experiment_id':'E1' if rep==1 else 'E2','run_id':f'{cid}-r{rep}'}
              for cid in ROSTER for rep in ((1,2,3) if cid in DECISIONS else (1,))]
        source={'study_id':'fictional-expanded','benchmark':'fictional','split':'evaluation','items':[{'item_id':'q0','labels':list('ABCD')}],
                'provenance':'mock','bindings':{'questions':'q','keys':'k'},'configurations':configs,'runs':runs}
        raw_rows=[raw(cid=r['config_id'],rep=r['replicate'],
                      status='http_error' if r['config_id']=='D13' else 'valid') for r in runs]
        view,rows,coverage=assemble_view([(source,raw_rows,{},CURRENT_NAMESPACE)])
        methods={'normalization':'native_selected_normalized_distribution','resource_imputation':RESOURCE_POLICY,
                 'primary_family':'decision_related_273','decision_models':{cid:{'probability_semantics_verified':True} for cid in DECISIONS}}
        result=analyze(view,rows,coverage,{('fictional','evaluation','q0'):{'reference_answer':'A'}},
                       seed=20261008,methods=methods)
        self.assertTrue(result['study_complete'])
        self.assertEqual(result['primary']['D13']['accuracy'],0)
        self.assertEqual(len(result['paired']['families']['S1']),273)
        self.assertEqual(len(result['repeat_operations']),26)
        self.assertEqual(result['decision_models']['D13']['confidence']['eligible_n'],0)
        self.assertEqual(result['decision_models']['D13']['stability']['pairs'][0]['transitions'],
                         {'http_error -> http_error':1})
