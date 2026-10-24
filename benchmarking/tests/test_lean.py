"""Manual regression suite: fictional-only dispatch, no private inputs or services."""
import copy
import base64
from http.client import IncompleteRead
from concurrent.futures import Future
import json
import io
import re
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
import threading
import time
import unittest
from unittest.mock import patch, MagicMock

from benchmarking.common import (ROOT, CLEAN_REPOSITORY_FILES, load_config, file_hash, fingerprint,
    read_package, run_inventory, strict_json, write_new, read_jsonl, runtime_blockers,
    load_questions, validate_transport, OPENROUTER_ENDPOINTS, TYPESAFE_ENDPOINT, JEV_DECISIONS_ENDPOINT, stop_events, attempt_identity, CAPACITY_AMENDMENT, capacity_amended_config, DEEPSEEK_TEN_AMENDMENT, deepseek_ten_amended_config, study_context)
from benchmarking import adapters
from benchmarking.run import (make_manifest, mock_dispatch, run_package, smoke_test, validate_input_manifest, plan_counts, main, reviewed_continuation, append_event, resolve_incomplete_read)
from preparation.prepare_medqa import prepare


class LeanExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.config=copy.deepcopy(load_config())
        self.config['execution'].update(concurrency=20,provider_concurrency=20,study_registry=str(self.root/'registry'),seed=41,
            scheduler_version=2,provider_concurrency_overrides={'SiliconFlow':20},provider_min_interval_seconds={})
        self.dev=self.root/'dev.jsonl';self.test=self.root/'test.jsonl'
        def rows(split,n):
            return [{'question':f'Fictional {split} symbol {i}',
                     'options':{k:f'{split} {i} {k}' for k in 'ABCD'},'answer_idx':'A'} for i in range(n)]
        write_new(self.dev,rows('development',4),jsonl=True)
        write_new(self.test,rows('test',3),jsonl=True)
        self.paths=prepare(dev_source=self.dev,test_source=self.test,seed=818,count=1,
            source_url='https://example.invalid/fictional',rights_basis='fictional fixture',
            acquired_at='2026-10-06',workspace=self.root,fictional=True,expected_test_items=3)
        self.q=self.paths['questions'];self.public=self.paths['questions_manifest']

    def tearDown(self): self.temp.cleanup()

    def stopped_parent(self):
        self.config['execution'].update(concurrency=1,provider_concurrency=1,provider_concurrency_overrides={})
        def transport(request,model,q):
            if model['route']=='InferenceNet': return {'status':429,'body_base64':base64.b64encode(b'{}').decode(),'headers':{}}
            return mock_dispatch(request,model,q)
        self.execute(experiment='E1+E2',transport=transport)
        self.assertTrue(stop_events(self.root/'attempts'))
        return self.root/'attempts',transport

    def continue_package(self, parent, output, *, transport=None, check_only=False, stops=None, amendment=None):
        q,public=self.paths['evaluation'],self.paths['evaluation_manifest']
        prep=validate_input_manifest(public,q)
        inventory=run_inventory(self.config,q,public,study_id='lean-fictional',
            resume_output=output,continuation=True)
        manifest=make_manifest(self.config,q,study_id='lean-fictional',experiment='E1+E2',provenance='mock',
            seed=41,keys_sha256=prep['keys_sha256'],preparation_sha256=prep['preparation_manifest_sha256'],
            inventory=inventory,content_sha256=prep.get('content_manifest_sha256'))
        manifest=reviewed_continuation(manifest,parent,output,reviewer='Fictional reviewer',
            reviewed_stops=stop_events(parent) if stops is None else stops,amendment=amendment)
        return run_package(manifest,q,self.config,output,mock=True,transport=transport,
            inventory=inventory,input_manifest_path=public,check_only=check_only)

    def test_continuation_excludes_all_parent_attempts_and_retains_failures(self):
        parent,_=self.stopped_parent();before=read_package(parent)
        hashes={str(p.relative_to(parent)):file_hash(p) for p in parent.rglob('*') if p.is_file()}
        calls=[]
        def transport(request,model,q):
            calls.append((q.item_id,model['model_id']));return mock_dispatch(request,model,q)
        result=self.continue_package(parent,self.root/'v2',transport=transport)
        self.assertTrue(result['complete']);self.assertEqual(result['terminal'],54)
        self.assertEqual(len(calls),len(before[2]['unstarted']))
        _,combined,_=read_package(self.root/'v2')
        self.assertEqual(len({attempt_identity(r) for r in combined}),54)
        self.assertEqual(sum(r['status']=='http_error' for r in combined),3)
        self.assertEqual(hashes,{str(p.relative_to(parent)):file_hash(p) for p in parent.rglob('*') if p.is_file()})
        self.assertEqual(len(result['segments']),2)
        from analysis.pipeline import load_study
        self.assertEqual(len(load_study(self.root/'v2',self.paths['evaluation'],self.paths['evaluation_keys'])[1]),54)

    def test_continuation_check_and_bad_review_create_no_files(self):
        parent,_=self.stopped_parent();output=self.root/'v2'
        registry=self.root/'registry'/fingerprint('lean-fictional')/'packages.jsonl'
        original=registry.read_bytes()
        result=self.continue_package(parent,output,check_only=True)
        self.assertFalse(output.exists());self.assertEqual(registry.read_bytes(),original)
        self.assertEqual(result['remaining_E1']+result['remaining_E2'],len(read_package(parent)[2]['unstarted']))
        with self.assertRaisesRegex(ValueError,'stop review'):
            self.continue_package(parent,output,stops=['0'*64])
        self.assertFalse(output.exists());self.assertEqual(registry.read_bytes(),original)

    def test_check_only_cli_returns_success_without_complete_field(self):
        parent,_=self.stopped_parent();output=self.root/'v2'
        registry=self.root/'registry'/fingerprint('lean-fictional')/'packages.jsonl'
        original=registry.read_bytes();stdout=io.StringIO();stderr=io.StringIO()
        argv=['benchmarking.run','run','--questions',str(self.paths['evaluation']),
            '--input-manifest',str(self.paths['evaluation_manifest']),'--study-id','lean-fictional',
            '--experiment','E1+E2','--output',str(output),'--continue-from',str(parent),
            '--reviewer','Fictional reviewer','--reviewed-stop',stop_events(parent)[0],'--mock','--check-only']
        with patch.object(sys,'argv',argv),patch('benchmarking.run.load_config',return_value=self.config), \
             patch('sys.stdout',stdout),patch('sys.stderr',stderr):
            self.assertEqual(main(),0)
        self.assertTrue(json.loads(stdout.getvalue())['check_only'])
        self.assertEqual(stderr.getvalue(),'')
        self.assertFalse(output.exists());self.assertEqual(registry.read_bytes(),original)

    def test_continuation_ctrl_c_resume_and_sibling_duplicate_are_safe(self):
        parent,_=self.stopped_parent();output=self.root/'v2';calls=[]
        def pause(*args):
            calls.append(1);signal.getsignal(signal.SIGINT)(signal.SIGINT,None)
            return mock_dispatch(*args)
        partial=self.continue_package(parent,output,transport=pause)
        self.assertFalse(partial['complete']);self.assertFalse(partial['indeterminate'])
        def finish(*args): calls.append(1);return mock_dispatch(*args)
        self.assertTrue(self.continue_package(parent,output,transport=finish)['complete'])
        self.assertEqual(len(calls),len(read_package(parent)[2]['unstarted']))
        with self.assertRaisesRegex(ValueError,'already has a continuation'):
            self.continue_package(parent,self.root/'duplicate')
        self.assertFalse((self.root/'duplicate').exists())

    def test_repeated_provider_stops_require_distinct_reviewed_segments(self):
        parent,transport=self.stopped_parent();second=self.root/'v2'
        partial=self.continue_package(parent,second,transport=transport)
        self.assertFalse(partial['complete']);self.assertTrue(stop_events(second))
        with self.assertRaisesRegex(ValueError,'rate-limit stop'):
            self.continue_package(parent,second)
        result=self.continue_package(second,self.root/'v3')
        self.assertTrue(result['complete']);self.assertEqual(len(result['segments']),3)
        records=read_package(self.root/'v3')[1]
        self.assertEqual(sum(r['status']=='http_error' for r in records),6)
        self.assertEqual(len({attempt_identity(r) for r in records}),54)

    def test_continuation_rejects_settings_changes_and_parent_tampering(self):
        parent,_=self.stopped_parent();output=self.root/'v2'
        self.config['execution']['timeout_seconds']=601
        with self.assertRaisesRegex(ValueError,'scientific settings changed'):
            self.continue_package(parent,output)
        self.config['execution']['timeout_seconds']=600
        self.continue_package(parent,output)
        with (parent/'events.jsonl').open('a') as stream: stream.write('\n')
        with self.assertRaisesRegex(ValueError,'parent evidence changed'): read_package(output)

    def test_continuation_keeps_distinct_source_freezes(self):
        parent,_=self.stopped_parent();original=read_package(parent)[0]
        checkout=self.root/'changed-source'
        for name in CLEAN_REPOSITORY_FILES:
            target=checkout/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
        with (checkout/'benchmarking/run.py').open('a') as stream: stream.write('\n# Fictional revised orchestration source.\n')
        with patch('benchmarking.common.ROOT',checkout),patch('benchmarking.run.ROOT',checkout):
            self.assertTrue(self.continue_package(parent,self.root/'v2')['complete'])
            changed=read_package(self.root/'v2')[0]
            self.assertNotEqual(changed['bindings']['code'],original['bindings']['code'])
            self.assertEqual(changed['continuation']['root_bindings'],original['bindings'])
        self.assertEqual(read_package(parent)[0],original)
        with self.assertRaisesRegex(ValueError,'bindings changed'):
            self.continue_package(parent,self.root/'v2')

    def test_unresolved_parent_start_blocks_continuation(self):
        parent,_=self.stopped_parent();manifest,records,coverage=read_package(parent)
        ident=dict(zip(('benchmark','split','item_id','config_id','run_id'),coverage['unstarted'][0]))
        append_event(parent/'events.jsonl',{'schema_version':3,'bindings':manifest['bindings'],
            'manifest_sha256':fingerprint(manifest),'event':'start','identity':ident,
            'request':records[0]['artifacts']['request'],'started_at':'2026-10-07T00:00:00+00:00'},coverage['last_event_sha256'])
        with self.assertRaisesRegex(ValueError,'unresolved parent starts|E2 start'):
            self.continue_package(parent,self.root/'v2')
        self.assertFalse((self.root/'v2').exists())

    def interrupted_parent(self):
        self.config['execution'].update(concurrency=1,provider_concurrency=1,provider_concurrency_overrides={})
        with self.assertRaisesRegex(RuntimeError,'fictional old-adapter crash'):
            self.execute(experiment='E1+E2',transport=lambda *args: (_ for _ in ()).throw(RuntimeError('fictional old-adapter crash')))
        parent=self.root/'attempts'
        start=next(e for e in read_jsonl(parent/'events.jsonl') if e['event']=='start')
        return parent,start

    def test_reviewed_interruption_preserves_prefix_and_never_resends(self):
        parent,start=self.interrupted_parent();journal=parent/'events.jsonl';before=journal.read_bytes()
        original={str(p.relative_to(parent)):file_hash(p) for p in parent.rglob('*') if p.is_file() and p!=journal}
        with patch('socket.socket.connect',side_effect=AssertionError('network forbidden')):
            result=resolve_incomplete_read(parent,start_event=start['event_sha256'],reviewer='Fictional reviewer',partial_bytes=7)
        self.assertTrue(journal.read_bytes().startswith(before));self.assertEqual(result['terminal'],1)
        self.assertFalse(result['indeterminate']);self.assertEqual(stop_events(parent),[result['reviewed_event']])
        self.assertEqual(original,{str(p.relative_to(parent)):file_hash(p) for p in parent.rglob('*') if p.is_file() and p!=journal})
        record=read_package(parent)[1][0]
        self.assertEqual(record['status'],'transport_error');self.assertIsNone(record['selected_label'])
        self.assertIsNone(record['timing']['duration_seconds']['value']);self.assertIsNone(record['billing']['observed_usd']['value'])
        self.assertEqual(record['recovery_review']['partial_bytes'],7)
        with self.assertRaisesRegex(ValueError,'exact single unresolved'):
            resolve_incomplete_read(parent,start_event=start['event_sha256'],reviewer='Fictional reviewer',partial_bytes=7)
        resolved_bytes=journal.read_bytes();calls=[]
        def transport(request,model,q): calls.append((q.item_id,fingerprint(request)));return mock_dispatch(request,model,q)
        self.assertTrue(self.continue_package(parent,self.root/'recovered',transport=transport)['complete'])
        self.assertEqual(journal.read_bytes(),resolved_bytes);self.assertEqual(len(calls),53)
        saved_request=strict_json((parent/start['request']['path']).read_text())
        self.assertNotIn((start['identity']['item_id'],fingerprint(saved_request)),calls)
        with self.assertRaisesRegex(ValueError,'already has a continuation'):
            resolve_incomplete_read(parent,start_event=start['event_sha256'],reviewer='Fictional reviewer',partial_bytes=7)

    def test_interruption_review_rejects_wrong_hash_and_existing_artifacts(self):
        parent,start=self.interrupted_parent();journal=parent/'events.jsonl';before=journal.read_bytes()
        with self.assertRaisesRegex(ValueError,'exact single unresolved'):
            resolve_incomplete_read(parent,start_event='0'*64,reviewer='Fictional reviewer',partial_bytes=7)
        artifact=parent/start['request']['path'];write_new(artifact.parent/'raw_response.json',{'fictional':'preserve'})
        with self.assertRaisesRegex(ValueError,'additional artifacts'):
            resolve_incomplete_read(parent,start_event=start['event_sha256'],reviewer='Fictional reviewer',partial_bytes=7)
        self.assertEqual(journal.read_bytes(),before)

    def test_incomplete_response_preserves_partial_bytes_without_parsing_or_retry(self):
        model=self.config['models']['C03'];q=load_questions(self.paths['evaluation'])[0]
        original=mock_dispatch({},model,q);partial=base64.b64decode(original['body_base64'])
        response=MagicMock();response.__enter__.return_value=response
        response.status=200;response.headers={'x-request-id':'fictional-id'}
        response.read.side_effect=IncompleteRead(partial)
        opener=MagicMock();opener.open.return_value=response
        with patch('benchmarking.adapters.urllib.request.build_opener',return_value=opener), \
             patch.dict('os.environ',{'OPENROUTER_API_KEY':'fictional-key'}):
            raw=adapters.dispatch({},model,600)
        self.assertEqual(base64.b64decode(raw['body_base64']),partial);opener.open.assert_called_once()
        parsed=adapters.parse(raw,model,q.labels)
        self.assertEqual(parsed['status'],'transport_error');self.assertIsNone(parsed['selected_label'])
        calls=[]
        def transport(*args): calls.append(1);return raw
        coverage=self.execute(experiment='E1+E2',transport=transport)
        self.assertTrue(coverage['complete']);self.assertFalse(coverage['indeterminate']);self.assertEqual(len(calls),54)
        self.assertTrue(all(r['status']=='transport_error' for r in read_package(self.root/'attempts')[1]))

    def test_reader_rejects_secondary_start_before_primary_coverage(self):
        parent,_=self.stopped_parent();manifest,records,coverage=read_package(parent)
        run=next(r for r in manifest['runs'] if r['role']=='secondary')
        ident={'benchmark':manifest['benchmark'],'split':manifest['split'],'item_id':manifest['items'][0]['item_id'],
            'config_id':run['config_id'],'run_id':run['run_id']}
        append_event(parent/'events.jsonl',{'schema_version':3,'bindings':manifest['bindings'],
            'manifest_sha256':fingerprint(manifest),'event':'start','identity':ident,
            'request':records[0]['artifacts']['request'],'started_at':'2026-10-07T00:00:00+00:00'},coverage['last_event_sha256'])
        with self.assertRaisesRegex(ValueError,'E2 start before complete E1'): read_package(parent)

    def test_only_reviewed_provider_streak_resets(self):
        self.config['execution'].update(concurrency=1,provider_concurrency=1,provider_concurrency_overrides={})
        openai_calls=0
        failure={'status':429,'body_base64':base64.b64encode(b'{}').decode(),'headers':{}}
        def original(request,model,q):
            nonlocal openai_calls
            if model['route']=='OpenAI':
                openai_calls+=1
                if openai_calls in (4,5): return failure
            if model['route']=='InferenceNet': return failure
            return mock_dispatch(request,model,q)
        self.execute(experiment='E1+E2',transport=original)
        def continued(request,model,q):
            return failure if model['route']=='OpenAI' else mock_dispatch(request,model,q)
        self.continue_package(self.root/'attempts',self.root/'v2',transport=continued)
        local=read_package(self.root/'v2',include_parents=False)[1]
        failures=[r for r in local if r['scheduler']['provider']=='OpenAI']
        self.assertEqual(len(failures),1)
        self.assertEqual(failures[0]['scheduler']['stop_reason'],'provider_rate_limit_stop')

    @contextmanager
    def immediate_scheduler(self):
        # Completed futures and virtual pacing exercise the real coordinator
        # without network calls, real sleeps or timing-dependent assertions.
        tick=[0.0]
        clock=SimpleNamespace(monotonic=lambda:tick[0],sleep=lambda delay:tick.__setitem__(0,tick[0]+delay))
        class ImmediateExecutor:
            def __init__(self,**kwargs): pass
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def submit(self,fn,*args):
                future=Future()
                try: future.set_result(fn(*args))
                except BaseException as exc: future.set_exception(exc)
                return future
        with patch('benchmarking.run.time',clock),patch('benchmarking.run.ThreadPoolExecutor',ImmediateExecutor):
            yield tick

    def test_provider_pause_finishes_other_e1_and_does_not_release_e2(self):
        self.config['execution'].update(scheduler_version=3,provider_concurrency_overrides={'SiliconFlow':20,'InferenceNet':1})
        calls=[]
        def limited(request,model,q):
            calls.append((model['route'],q.item_id))
            if model['route']=='InferenceNet': return {'status':429,'body_base64':base64.b64encode(b'{}').decode(),'headers':{}}
            return mock_dispatch(request,model,q)
        with self.immediate_scheduler():result=self.execute(experiment='E1+E2',transport=limited)
        manifest,records,coverage=read_package(self.root/'attempts')
        self.assertFalse(result['complete']);self.assertFalse(result['indeterminate'])
        self.assertEqual(result['paused_providers'],['InferenceNet'])
        self.assertEqual(sum(route=='InferenceNet' for route,_ in calls),3)
        self.assertEqual(sum(r['status']=='valid' for r in records),42)
        self.assertEqual(sum('-E1-' in r['identity']['run_id'] for r in records),45)
        self.assertTrue(all(i[3] in ('C13','C14') or '-E2-' in i[-1] for i in coverage['unstarted']))
        self.assertTrue(all('-E1-' in r['identity']['run_id'] for r in records))
        with self.assertRaisesRegex(ValueError,'rate-limit stop'):
            self.execute(experiment='E1+E2',transport=limited)
        self.assertEqual(len(calls),45)

    def test_pause_latches_after_later_inflight_success_and_all_paused_exits(self):
        self.config['execution'].update(scheduler_version=3,concurrency=4,provider_concurrency=4,provider_concurrency_overrides={})
        for model in self.config['models'].values():model.update(route='fictional-provider',provider_slug='fictional-provider')
        count=[0]
        def limited(request,model,q):
            count[0]+=1
            raw=mock_dispatch(request,model,q)
            if count[0]<=3:raw['status']=429
            return raw
        with self.immediate_scheduler():result=self.execute(experiment='E1+E2',transport=limited)
        self.assertEqual(count[0],4);self.assertEqual(result['terminal'],4)
        self.assertFalse(result['indeterminate']);self.assertFalse(result['complete'])
        self.assertEqual(result['paused_providers'],['fictional-provider'])
        self.assertEqual([r['scheduler']['stop_reason'] for r in read_package(self.root/'attempts')[1]],
                         [None,None,'provider_rate_limit_stop',None])

    def test_amended_continuation_retains_registry_parent_and_exact_payloads(self):
        parent,_=self.stopped_parent();prior,original,coverage=read_package(parent)
        before={str(p.relative_to(parent)):file_hash(p) for p in parent.rglob('*') if p.is_file()}
        old_config=copy.deepcopy(self.config);self.config=capacity_amended_config(self.config)
        with self.immediate_scheduler():
            check=self.continue_package(parent,self.root/'amended',check_only=True,amendment=CAPACITY_AMENDMENT)
            self.assertFalse((self.root/'amended').exists())
            result=self.continue_package(parent,self.root/'amended',amendment=CAPACITY_AMENDMENT)
        self.assertTrue(result['complete']);self.assertEqual(result['terminal'],54)
        current,records,_=read_package(self.root/'amended')
        self.assertEqual(study_context(prior),study_context(current))
        self.assertEqual(current['continuation']['execution_amendment'],CAPACITY_AMENDMENT)
        self.assertEqual(current['continuation']['root_schedule'],prior['schedule'])
        self.assertEqual(sum(r['status']=='http_error' for r in records),3)
        self.assertEqual(len({attempt_identity(r) for r in records}),54)
        self.assertEqual(before,{str(p.relative_to(parent)):file_hash(p) for p in parent.rglob('*') if p.is_file()})
        questions=load_questions(self.paths['evaluation'])
        from benchmarking.run import build_prompt
        for q in questions:
            for cid in self.config['model_roster']:
                self.assertEqual(adapters.build_request(q,old_config['models'][cid],build_prompt(q,old_config),mock=True),
                                 adapters.build_request(q,self.config['models'][cid],build_prompt(q,self.config),mock=True))
        from analysis.pipeline import load_study
        self.assertEqual(len(load_study(self.root/'amended',self.paths['evaluation'],self.paths['evaluation_keys'])[1]),54)

    def test_capacity_amendment_requires_flag_and_rejects_other_changes(self):
        parent,_=self.stopped_parent();self.config=capacity_amended_config(self.config)
        with self.assertRaisesRegex(ValueError,'scientific settings changed'):
            self.continue_package(parent,self.root/'unreviewed')
        changes=[('timeout_seconds',601),('concurrency',2),('rate_limit_stop_after',4)]
        for key,value in changes:
            config=copy.deepcopy(self.config);config['execution'][key]=value
            with patch.object(self,'config',config),self.assertRaisesRegex(ValueError,'unapproved configuration'):
                self.continue_package(parent,self.root/key,amendment=CAPACITY_AMENDMENT)
        with self.assertRaisesRegex(ValueError,'unsupported execution amendment'):
            self.continue_package(parent,self.root/'unknown',amendment='unreviewed-change')
        self.assertTrue(all(not (self.root/name).exists() for name in ('unreviewed','unknown',*[k for k,_ in changes])))

    def test_capacity_amendment_paces_both_deepseek_modes_and_preserves_e2_barrier(self):
        self.config=capacity_amended_config(self.config);starts=[]
        with self.immediate_scheduler() as tick:
            def transport(request,model,q):
                if model['route']=='InferenceNet':starts.append((tick[0],model['effort']))
                return mock_dispatch(request,model,q)
            result=self.execute(experiment='E1+E2',transport=transport)
        self.assertTrue(result['complete']);self.assertEqual(len(starts),6)
        self.assertTrue(all(b[0]-a[0]>=10 for a,b in zip(starts,starts[1:])))
        self.assertEqual({effort for _,effort in starts},{'low','high'})
        primary=0
        for event in read_jsonl(self.root/'attempts/events.jsonl'):
            if event['event']=='finish' and '-E1-' in event['identity']['run_id']:primary+=1
            if event['event']=='start' and '-E2-' in event['identity']['run_id']:self.assertEqual(primary,48)

    def test_repeated_provider_pause_requires_a_reviewed_child_with_root_schedule(self):
        test=self.root/'test-four.jsonl'
        write_new(test,[{'question':f'Fictional larger test {i}','options':{k:f'{i} {k}' for k in 'ABCD'},'answer_idx':'A'} for i in range(4)],jsonl=True)
        self.paths=prepare(dev_source=self.dev,test_source=test,seed=818,count=1,
            source_url='https://example.invalid/fictional',rights_basis='fictional fixture',
            acquired_at='2026-10-06',workspace=self.root/'four',fictional=True,expected_test_items=4)
        parent,_=self.stopped_parent();self.config=capacity_amended_config(self.config)
        second=self.root/'amended';calls=[0]
        def limited(request,model,q):
            if model['route']=='InferenceNet':return {'status':429,'body_base64':base64.b64encode(b'{}').decode(),'headers':{}}
            calls[0]+=1
            return mock_dispatch(request,model,q)
        with self.immediate_scheduler():result=self.continue_package(parent,second,transport=limited,amendment=CAPACITY_AMENDMENT)
        self.assertFalse(result['complete']);self.assertTrue(stop_events(second))
        with self.assertRaisesRegex(ValueError,'rate-limit stop'):
            self.continue_package(parent,second,amendment=CAPACITY_AMENDMENT)
        with self.immediate_scheduler():result=self.continue_package(second,self.root/'final')
        self.assertTrue(result['complete']);self.assertEqual(result['terminal'],72)
        self.assertEqual(len(result['segments']),3)
        self.assertEqual(read_package(self.root/'final')[0]['continuation']['root_schedule'],read_package(parent)[0]['schedule'])

    def test_capacity_limit_is_shared_by_both_modes_under_concurrent_workers(self):
        self.config=capacity_amended_config(self.config)
        self.config['execution']['provider_min_interval_seconds']={}
        active=0;peak=0;lock=threading.Lock();efforts=set()
        def transport(request,model,q):
            nonlocal active,peak
            if model['route']!='InferenceNet':
                time.sleep(.005);return mock_dispatch(request,model,q)
            with lock:active+=1;peak=max(peak,active);efforts.add(model['effort'])
            try:
                time.sleep(.005);return mock_dispatch(request,model,q)
            finally:
                with lock:active-=1
        self.assertTrue(self.execute(experiment='E1+E2',transport=transport)['complete'])
        self.assertEqual(peak,1);self.assertEqual(efforts,{'low','high'})

    def test_scheduler_three_keeps_identity_drift_as_a_global_stop(self):
        self.config['execution']['scheduler_version']=3
        self.config['execution'].update(concurrency=1,provider_concurrency=1,provider_concurrency_overrides={})
        calls=[]
        def drift(request,model,q):
            calls.append(1);changed=copy.deepcopy(model);changed['version']='fictional-unrelated-model'
            return mock_dispatch(request,changed,q)
        with self.immediate_scheduler():result=self.execute(experiment='E1+E2',transport=drift)
        self.assertFalse(result['complete']);self.assertFalse(result['indeterminate'])
        self.assertEqual(len(calls),1)
        self.assertEqual(read_package(self.root/'attempts')[1][0]['status'],'drift')
        with self.assertRaisesRegex(ValueError,'drift'):self.execute(experiment='E1+E2',transport=drift)

    def operator_drained_parent(self):
        self.config=capacity_amended_config(self.config)
        requested=[]
        def interrupted(request,model,q):
            requested.append('SIGINT')
            return mock_dispatch(request,model,q)
        with patch('benchmarking.run._graceful_signals',return_value=nullcontext(requested)),self.immediate_scheduler():
            self.execute(experiment='E1+E2',transport=interrupted)
        parent=self.root/'attempts'
        self.assertFalse(stop_events(parent))
        self.assertFalse(read_package(parent)[2]['indeterminate'])
        self.assertTrue(read_package(parent)[2]['unstarted'])
        return parent

    def test_ten_slot_amendment_continues_drained_parent_without_stop_event(self):
        parent=self.operator_drained_parent()
        before={str(p.relative_to(parent)):file_hash(p) for p in parent.rglob('*') if p.is_file()}
        inherited=read_package(parent)[1]
        self.config=deepseek_ten_amended_config(self.config)
        with self.immediate_scheduler():result=self.continue_package(parent,self.root/'ten',amendment=DEEPSEEK_TEN_AMENDMENT)
        self.assertTrue(result['complete']);self.assertEqual(result['terminal'],54)
        manifest,records,_=read_package(self.root/'ten')
        self.assertEqual(manifest['continuation']['execution_amendment'],DEEPSEEK_TEN_AMENDMENT)
        self.assertEqual(manifest['continuation']['reviewed_stop_events'],[])
        self.assertEqual(study_context(manifest),study_context(read_package(parent)[0]))
        self.assertEqual(records[:len(inherited)],inherited)
        self.assertEqual(before,{str(p.relative_to(parent)):file_hash(p) for p in parent.rglob('*') if p.is_file()})
        primary=0
        for event in read_jsonl(parent/'events.jsonl')+read_jsonl(self.root/'ten/events.jsonl'):
            if event['event']=='finish' and '-E1-' in event['identity']['run_id']:primary+=1
            if event['event']=='start' and '-E2-' in event['identity']['run_id']:self.assertEqual(primary,48)

    def test_ten_slot_amendment_rejects_missing_flag_and_other_setting_changes(self):
        parent=self.operator_drained_parent();self.config=deepseek_ten_amended_config(self.config)
        with self.assertRaisesRegex(ValueError,'scientific settings changed'):
            self.continue_package(parent,self.root/'unreviewed-ten')
        for field,value in [('timeout_seconds',601),('rate_limit_stop_after',4),('provider_min_interval_seconds',{'InferenceNet':2.0})]:
            config=copy.deepcopy(self.config);config['execution'][field]=value
            with patch.object(self,'config',config),self.assertRaisesRegex(ValueError,'unapproved configuration'):
                self.continue_package(parent,self.root/field,amendment=DEEPSEEK_TEN_AMENDMENT)
        self.assertFalse((self.root/'unreviewed-ten').exists())

    def test_ten_slot_check_only_cli_accepts_drained_parent_without_stop_hash(self):
        parent=self.operator_drained_parent();self.config=deepseek_ten_amended_config(self.config)
        output=self.root/'ten-cli';stdout=io.StringIO();stderr=io.StringIO()
        registry=self.root/'registry'/fingerprint('lean-fictional')/'packages.jsonl'
        original=registry.read_bytes()
        argv=['benchmarking.run','run','--questions',str(self.paths['evaluation']),
            '--input-manifest',str(self.paths['evaluation_manifest']),'--study-id','lean-fictional',
            '--experiment','E1+E2','--output',str(output),'--continue-from',str(parent),
            '--reviewer','Fictional reviewer','--execution-amendment',DEEPSEEK_TEN_AMENDMENT,'--mock','--check-only']
        with patch.object(sys,'argv',argv),patch('benchmarking.run.load_config',return_value=self.config), \
             patch('sys.stdout',stdout),patch('sys.stderr',stderr):
            self.assertEqual(main(),0)
        checked=json.loads(stdout.getvalue())
        self.assertEqual(checked['reviewed_stop_events'],[])
        self.assertEqual(checked['schedule_policy']['provider_concurrency_overrides']['InferenceNet'],10)
        self.assertEqual(checked['schedule_policy']['provider_min_interval_seconds']['InferenceNet'],5.0)
        self.assertEqual(stderr.getvalue(),'')
        self.assertFalse(output.exists());self.assertEqual(registry.read_bytes(),original)

    def test_ten_slot_amendment_rejects_indeterminate_parent(self):
        parent=self.operator_drained_parent();manifest,_,coverage=read_package(parent)
        ident=dict(zip(('benchmark','split','item_id','config_id','run_id'),coverage['unstarted'][0]))
        request=next(parent.glob('artifacts/*/request.json'))
        append_event(parent/'events.jsonl',{'schema_version':3,'bindings':manifest['bindings'],
            'manifest_sha256':fingerprint(manifest),'event':'start','identity':ident,
            'request':{'path':str(request.relative_to(parent)),'sha256':file_hash(request)},
            'started_at':'2026-10-07T10:00:00+00:00'},coverage['last_event_sha256'])
        self.config=deepseek_ten_amended_config(self.config)
        with self.assertRaisesRegex(ValueError,'unresolved parent starts'):
            self.continue_package(parent,self.root/'open-ten',amendment=DEEPSEEK_TEN_AMENDMENT)
        self.assertFalse((self.root/'open-ten').exists())

    def test_ten_slot_amendment_requires_one_slot_scheduler_three_parent(self):
        with self.assertRaisesRegex(ValueError,'scheduler version 3'):
            deepseek_ten_amended_config(self.config)
        self.config=capacity_amended_config(self.config)
        amended=deepseek_ten_amended_config(self.config)
        self.assertEqual(amended['execution']['provider_concurrency_overrides']['InferenceNet'],10)
        self.assertEqual(amended['execution']['provider_min_interval_seconds']['InferenceNet'],5.0)
        self.assertEqual(self.config['execution']['provider_concurrency_overrides']['InferenceNet'],1)
        with self.assertRaisesRegex(ValueError,'limit 1 parent'):
            deepseek_ten_amended_config(amended)

    def execute(self, *, experiment='E1',output=None,mock=True,transport=None,study='lean-fictional'):
        q,public=self.paths['evaluation'],self.paths['evaluation_manifest']
        prep=validate_input_manifest(public,q)
        inventory=run_inventory(self.config,q,public,study_id=study)
        manifest=make_manifest(self.config,q,study_id=study,experiment=experiment,provenance='mock' if mock else 'definitive',
            seed=41,keys_sha256=prep['keys_sha256'],preparation_sha256=prep['preparation_manifest_sha256'],inventory=inventory,
            content_sha256=prep.get('content_manifest_sha256'))
        result=run_package(manifest,q,self.config,output or self.root/'attempts',mock=mock,transport=transport,
            inventory=inventory,input_manifest_path=public,progress_stream=None)
        return result

    def test_manual_live_path_needs_no_private_approvals_and_does_not_load_keys(self):
        # Exercise the full-count live branch using fictional inputs and transport,
        # then stop after one receipt; no approval files, private data or network.
        full_test=self.root/'full-fictional-test.jsonl'
        write_new(full_test,[{'question':f'Fictional full test {i}',
            'options':{k:f'full {i} {k}' for k in 'ABCD'},'answer_idx':'A'} for i in range(1273)],jsonl=True)
        self.paths=prepare(dev_source=self.dev,test_source=full_test,seed=818,count=1,
            source_url='https://example.invalid/fictional',rights_basis='fictional fixture',
            acquired_at='2026-10-06',workspace=self.root/'live-fixture',fictional=True,expected_test_items=1273)
        public_path=self.paths['evaluation_manifest']
        public=strict_json(public_path.read_text());public['provenance']='prepared'
        public_path.write_text(json.dumps(public))
        for name in ('keys','audit_keys','evaluation_keys','answers'): self.paths[name].unlink()
        self.config['execution'].update(concurrency=1,provider_concurrency=1,provider_concurrency_overrides={})
        calls=[]
        def transport(*args):
            calls.append(1);signal.getsignal(signal.SIGINT)(signal.SIGINT,None)
            return mock_dispatch(*args)
        with patch.dict('os.environ',{'OPENROUTER_API_KEY':'fictional-test-key'}), \
             patch('socket.socket.connect',side_effect=AssertionError('network forbidden')):
            coverage=self.execute(mock=False,transport=transport)
        self.assertFalse(coverage['complete']);self.assertEqual(len(calls),1)
        self.assertFalse(coverage['indeterminate']);self.assertTrue(coverage['unstarted'])
        manifest,records,_=read_package(self.root/'attempts')
        self.assertEqual(manifest['execution_workflow'],'lean-v1')
        self.assertEqual(manifest['provenance'],'definitive');self.assertEqual(len(manifest['items']),1273)
        self.assertEqual(len(records),1)
        self.assertEqual(strict_json((self.root/'attempts/source/config.json').read_text()),self.config)

    def test_exact_resume_never_resends_and_new_git_commit_retains_original_provenance(self):
        calls=[]
        def transport(*args): calls.append(1);return mock_dispatch(*args)
        self.execute(transport=transport)
        with patch('benchmarking.common.execution_git_record',return_value={'commit':'a'*40,'dirty':False}):
            self.execute(transport=transport)
        self.assertEqual(len(calls),48)
        with self.assertRaisesRegex(ValueError,'another output|repeated attempts'):
            self.execute(output=self.root/'duplicate',transport=transport)
        self.config['execution']['timeout_seconds']=601
        with self.assertRaisesRegex(ValueError,'bindings changed'): self.execute(transport=transport)
        self.assertEqual(len(calls),48)

    def test_saved_source_and_raw_response_tampering_fail_without_original_checkout(self):
        self.execute()
        package=self.root/'attempts'
        moved=self.root/'relocated';shutil.copytree(package,moved)
        self.assertTrue(read_package(moved)[2]['complete'])
        (moved/'source/benchmarking/run.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'source mismatch'): read_package(moved)
        ref=read_package(package)[1][0]['artifacts']['raw_response']['path']
        (package/ref).write_text('{}')
        with self.assertRaisesRegex(ValueError,'artifact hash'): read_package(package)

    def test_indeterminate_start_is_preserved_and_never_resent(self):
        calls=[]
        def transport(*args): calls.append(1);raise RuntimeError('fictional worker interruption')
        self.config['execution'].update(concurrency=1,provider_concurrency=1,provider_concurrency_overrides={})
        with self.assertRaises(RuntimeError): self.execute(transport=transport)
        self.assertEqual(len(read_package(self.root/'attempts')[2]['indeterminate']),1)
        with self.assertRaisesRegex(ValueError,'indeterminate'): self.execute(transport=transport)
        self.assertEqual(len(calls),1)

    def test_shared_provider_concurrency_and_e2_follow_primary_finishes(self):
        self.config['execution'].update(provider_concurrency=2,provider_concurrency_overrides={'SiliconFlow':2})
        active={};peak={};global_peak=[0];lock=threading.Lock()
        def transport(request,model,q):
            route=model['route']
            with lock:
                active[route]=active.get(route,0)+1;peak[route]=max(peak.get(route,0),active[route])
                global_peak[0]=max(global_peak[0],sum(active.values()))
            try:
                time.sleep(.01);return mock_dispatch(request,model,q)
            finally:
                with lock: active[route]-=1
        coverage=self.execute(experiment='E1+E2',transport=transport)
        self.assertTrue(coverage['complete']);self.assertEqual(coverage['terminal'],54)
        self.assertTrue(all(n<=2 for n in peak.values()))
        self.assertGreater(global_peak[0],1);self.assertLessEqual(global_peak[0],20)
        events=read_jsonl(self.root/'attempts/events.jsonl');primary=0
        for event in events:
            if event['event']=='finish' and '-E1-' in event['identity']['run_id']: primary+=1
            if event['event']=='start' and '-E2-' in event['identity']['run_id']: self.assertEqual(primary,48)

    def test_graceful_stop_drains_inflight_and_keeps_unstarted_work_resumable(self):
        calls=[];lock=threading.Lock()
        def transport(*args):
            with lock:
                calls.append(1)
                if len(calls)==1: signal.getsignal(signal.SIGINT)(signal.SIGINT,None)
            time.sleep(.005);return mock_dispatch(*args)
        stopped=self.execute(experiment='E1+E2',transport=transport)
        self.assertFalse(stopped['complete']);self.assertFalse(stopped['indeterminate'])
        self.assertTrue(stopped['unstarted']);self.assertEqual(stopped['terminal'],len(calls))
        resumed=self.execute(experiment='E1+E2',transport=lambda *args:mock_dispatch(*args))
        self.assertTrue(resumed['complete']);self.assertEqual(resumed['terminal'],54)

    def test_identity_drift_blocks_resume_without_repair_or_resend(self):
        calls=[]
        def drift(request,model,q):
            calls.append(1);changed=copy.deepcopy(model);changed['version']='fictional-unrelated-model'
            return mock_dispatch(request,changed,q)
        self.execute(transport=drift)
        before=len(calls)
        with self.assertRaisesRegex(ValueError,'drift'): self.execute(transport=drift)
        self.assertEqual(len(calls),before)

    def test_separate_e2_requires_registered_complete_primary(self):
        with self.assertRaisesRegex(ValueError,'complete registered E1'): self.execute(experiment='E2')
        self.execute(experiment='E1',output=self.root/'primary')
        coverage=self.execute(experiment='E2',output=self.root/'secondary')
        self.assertEqual(coverage['terminal'],6)

    def test_answer_bearing_schema_and_changed_handoff_fail_before_calls(self):
        evaluation=self.paths['evaluation']
        row=strict_json(evaluation.read_text().splitlines()[0]);row['answer']='A'
        evaluation.write_text(json.dumps(row)+'\n')
        with self.assertRaisesRegex(ValueError,'answer-free'): self.execute(transport=lambda *args:self.fail('dispatch forbidden'))
        self.assertFalse((self.root/'registry').exists())

    def test_transport_allowlist_matches_each_adapter_and_rejects_cross_adapter_endpoints(self):
        endpoints={'openrouter':OPENROUTER_ENDPOINTS,'jev_gateway':{JEV_DECISIONS_ENDPOINT},'jev':{TYPESAFE_ENDPOINT}}
        for adapter,allowed in endpoints.items():
            for endpoint in set().union(*endpoints.values()) | {'https://example.invalid/completions','',None}:
                with self.subTest(adapter=adapter,endpoint=endpoint):
                    model={'adapter':adapter,'endpoint':endpoint}
                    if endpoint in allowed: validate_transport(model)
                    else:
                        with self.assertRaisesRegex(ValueError,'unapproved transport endpoint'): validate_transport(model)
        with self.assertRaisesRegex(ValueError,'unapproved transport endpoint'):
            validate_transport({'adapter':'unknown','endpoint':JEV_DECISIONS_ENDPOINT})

    def test_invalid_transport_is_rejected_before_registration_starts_or_smoke_calls(self):
        q=load_questions(self.q)[0]
        for cid,endpoint in (('C03','https://example.invalid/completions'),('C03',JEV_DECISIONS_ENDPOINT),
                             ('J01',next(iter(OPENROUTER_ENDPOINTS)))):
            with self.subTest(config_id=cid,endpoint=endpoint):
                config=copy.deepcopy(self.config);config['models'][cid]['endpoint']=endpoint
                self.assertIn(cid+': unapproved transport endpoint',runtime_blockers(config))
                model=config['models'][cid]
                with self.assertRaisesRegex(ValueError,'unapproved transport endpoint'):
                    adapters.build_request(q,model,'fictional prompt',mock=True)
                with patch('benchmarking.adapters.urllib.request.build_opener') as opener:
                    with self.assertRaisesRegex(ValueError,'unapproved transport endpoint'):
                        adapters.dispatch({},model,600)
                    opener.assert_not_called()
                with patch.object(self,'config',config):
                    with self.assertRaisesRegex(ValueError,'unapproved transport endpoint'):
                        self.execute(transport=lambda *args:self.fail('dispatch forbidden'))
                    with self.assertRaisesRegex(ValueError,'unapproved transport endpoint'):
                        smoke_test(config,self.q,self.public,self.root/'bad-smoke',mock=True,
                                   transport=lambda *args:self.fail('dispatch forbidden'))
                self.assertFalse((self.root/'registry').exists())
                self.assertFalse((self.root/'attempts').exists())
                self.assertFalse((self.root/'bad-smoke').exists())

    def scripted_completion_batch(self, statuses, finish_order, enumeration, name):
        """Exercise the coordinator with completed futures and controlled receipt times."""
        config=copy.deepcopy(self.config)
        config['execution'].update(concurrency=4,provider_concurrency=4,provider_concurrency_overrides={})
        for model in config['models'].values():
            model.update(route='fictional-provider',provider_slug='fictional-provider')
        submitted=[];waits=[]
        class CompletedExecutor:
            def __init__(self,**kwargs): pass
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def submit(self,fn,*args):
                index=len(submitted);future=Future()
                result=fn(*args)
                if index<4:
                    result['raw']['status']=statuses[index]
                    result['started_at']='2026-10-06T00:00:00+00:00'
                    result['finished_at']=f'2026-10-06T00:00:{finish_order[index]:02d}+00:00'
                submitted.append(future);future.set_result(result)
                return future
        def completed_wait(inflight,**kwargs):
            batch=list(inflight)
            ordered=[batch[i] for i in enumeration] if not waits else list(reversed(batch))
            waits.append(1)
            return ordered,set()
        output=self.root/name
        with patch.object(self,'config',config), \
             patch('benchmarking.run.ThreadPoolExecutor',CompletedExecutor), \
             patch('benchmarking.run.wait',completed_wait):
            coverage=self.execute(output=output,study=name)
            before=len(submitted)
            if not coverage['complete']:
                with self.assertRaisesRegex(ValueError,'rate-limit stop'):
                    self.execute(output=output,study=name)
                self.assertEqual(len(submitted),before)
        return coverage,read_package(output)[1],len(submitted)

    def test_429_stop_uses_finish_times_for_every_completed_batch_enumeration(self):
        for enumeration in ((0,1,2,3),(0,3,1,2),(3,2,1,0)):
            name='finish-order-'+''.join(map(str,enumeration))
            coverage,records,started=self.scripted_completion_batch(
                (429,429,429,200),(1,2,3,4),enumeration,name)
            self.assertFalse(coverage['complete']);self.assertFalse(coverage['indeterminate'])
            self.assertEqual(started,4);self.assertEqual(coverage['terminal'],4)
            self.assertEqual([r['scheduler']['response_status'] for r in records],[429,429,429,200])
            self.assertEqual([r['scheduler']['stop_reason'] for r in records],
                [None,None,'provider_rate_limit_stop',None])

    def test_429_success_reset_follows_finish_time_instead_of_submission_order(self):
        coverage,records,started=self.scripted_completion_batch(
            (429,429,429,200),(1,3,4,2),(0,1,2,3),'success-reset')
        self.assertTrue(coverage['complete']);self.assertEqual(started,48)
        self.assertEqual([r['scheduler']['response_status'] for r in records[:4]],[429,200,429,429])
        self.assertTrue(all(r['scheduler']['stop_reason'] is None for r in records))

    def test_equal_finish_times_use_submission_sequence_and_drain_after_stop(self):
        coverage,records,started=self.scripted_completion_batch(
            (429,429,429,200),(1,1,1,1),(0,3,1,2),'finish-tie')
        self.assertFalse(coverage['complete']);self.assertFalse(coverage['indeterminate'])
        self.assertEqual(started,4)
        self.assertEqual([r['scheduler']['response_status'] for r in records],[429,429,429,200])
        self.assertEqual(records[2]['scheduler']['stop_reason'],'provider_rate_limit_stop')
        self.assertEqual(records[3]['scheduler']['stop_reason'],None)

    def test_smoke_stops_on_failure_and_retains_request_receipt(self):
        calls=[]
        def timeout(*args): calls.append(1);raise TimeoutError('fictional')
        result=smoke_test(self.config,self.q,self.public,self.root/'smoke',mock=True,transport=timeout)
        self.assertFalse(result['passed']);self.assertEqual(len(calls),1)
        self.assertEqual(result['outcomes'][0]['status'],'timeout')
        self.assertTrue((self.root/'smoke/provenance.json').exists())
        with self.assertRaisesRegex(ValueError,'already exists'):
            smoke_test(self.config,self.q,self.public,self.root/'smoke',mock=True,transport=timeout)

    def test_smoke_detects_returned_control_mismatch_before_other_calls(self):
        calls=[]
        def wrong_controls(*args):
            calls.append(1);raw=mock_dispatch(*args)
            body=json.loads(base64.b64decode(raw['body_base64']))
            body['effective_controls']={'temperature':999}
            raw['body_base64']=base64.b64encode(json.dumps(body).encode()).decode()
            return raw
        result=smoke_test(self.config,self.q,self.public,self.root/'control-drift',mock=True,transport=wrong_controls)
        self.assertFalse(result['passed']);self.assertEqual(len(calls),1)
        self.assertEqual(result['outcomes'][0]['status'],'drift')
        self.assertEqual(result['outcomes'][0]['failure_reason'],'effective_controls_mismatch')

    def test_clean_allowlist_has_no_root_helper_test_archive_or_private_vault_dependency(self):
        clean=self.root/'code-only';clean.mkdir()
        for name in CLEAN_REPOSITORY_FILES:
            target=clean/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
        command=[sys.executable,'-B','-m','benchmarking.run','smoke','--questions',str(self.q),
                 '--input-manifest',str(self.public),'--output',str(clean/'benchmarking/output/mock-smoke'),'--mock']
        result=subprocess.run(command,cwd=clean,capture_output=True,text=True,env={'PATH':__import__('os').environ['PATH']})
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse((clean/'common.py').exists());self.assertFalse((clean/'analysis').exists())
        self.assertTrue(strict_json((clean/'benchmarking/output/mock-smoke/result.json').read_text())['passed'])

    def test_public_source_omits_retired_experiment_and_collection_rejects_it(self):
        for name in CLEAN_REPOSITORY_FILES:
            with self.subTest(path=name):
                self.assertIsNone(re.search(r'(?i)(?<![a-z0-9])e0(?![a-z0-9])',name+'\n'+(ROOT/name).read_text()))
        with self.assertRaisesRegex(ValueError,'unsupported experiment'): plan_counts(self.config,'E0')
        with self.assertRaisesRegex(ValueError,'unsupported experiment'):
            self.execute(experiment='E0',transport=lambda *args:self.fail('dispatch forbidden'))
        self.assertFalse((self.root/'registry').exists())

    def test_cli_requires_one_explicit_mode_before_loading_inputs(self):
        for command in ('run','smoke'):
            arguments=[command,'--questions',str(self.root/'missing-questions'),
                '--input-manifest',str(self.root/'missing-manifest'),'--output',str(self.root/'mode-output')]
            if command=='run': arguments+=['--study-id','fictional-mode-test']
            for flags in ([],['--live','--mock']):
                with self.subTest(command=command,flags=flags), \
                     patch.object(sys,'argv',['benchmarking.run',*arguments,*flags]), \
                     patch('sys.stderr',new_callable=io.StringIO) as stderr, \
                     patch('benchmarking.run.load_config',side_effect=AssertionError('config must not be loaded')) as load:
                    with self.assertRaises(SystemExit) as rejected: main()
                    self.assertEqual(rejected.exception.code,2)
                    self.assertIn('--live',stderr.getvalue());self.assertIn('--mock',stderr.getvalue())
                    load.assert_not_called()
                    self.assertFalse((self.root/'mode-output').exists())
        self.assertFalse((self.root/'registry').exists())

    def test_original_real_source_hash_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'original study split'):
            prepare(dev_source=self.dev,test_source=self.test,seed=818,count=1,
                source_url='https://example.invalid/fictional',rights_basis='fictional fixture',
                acquired_at='2026-10-06',workspace=self.root/'wrong-real',expected_test_items=1273)

if __name__=='__main__': unittest.main()
