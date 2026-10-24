"""Manual fictional regression suite for AW-179. No network or research inputs."""
import base64
import copy
import json
from pathlib import Path
import tempfile
import os
import tomllib
import unittest
from unittest.mock import patch, MagicMock
from types import SimpleNamespace

from benchmarking.common import Question, Option, fingerprint, measurement, empty_probability
from benchmarking.configuration_ids import (CURRENT_NAMESPACE, ORIGINAL_NAMESPACE, LUNA_NAMESPACE,
    DRAFT_NAMESPACE, CONFIGURATIONS, EXTENSION_IDS, canonical_configuration)
from benchmarking import native_decisions as native
from benchmarking.extension_contract import (Ledger, SCHEMA, WORKFLOW, attempt_id, configurations,
    runs, config_errors, request_payload)
from benchmarking.extension import CONFIG, readiness, registry_claim, collect, queue_phase, worker
from benchmarking.extension_contract import read_extension_package
from benchmarking.common import file_hash, question_record, write_new


def config():
    return tomllib.loads(CONFIG.read_text())


def question():
    return Question('medqa','fictional-1','development','Keep the entire stem: α & longer text.',
                    tuple(Option(k,'Option '+k) for k in 'ABCD'))


def response(model_config, answer=None, **changes):
    body = {'model':model_config['expected_returned_model'],'provider':model_config['route'],'id':'fictional-id',
            'answers':{'answer':answer or {'type':'choice','choice':'A',
                'probabilities':{'A':0.1,'B':0.2,'C':0.3,'D':0.4},'confidence':17}},
            'usage':{'input_tokens':100,'output_tokens':0,'cost':0}}
    body.update(changes)
    return {'status':200,'headers':{},'body_base64':base64.b64encode(json.dumps(body).encode()).decode()}


def manifest(ids=('D03',), items=('fictional-1',), development=False):
    c = config()
    return {'schema_version':SCHEMA,'execution_workflow':WORKFLOW,'config_namespace':CURRENT_NAMESPACE,
        'provenance':'mock','purpose':'development' if development else 'evaluation',
        'benchmark':'medqa','split':'development','study_id':'fictional',
        'models':{cid:c['models'][cid] for cid in ids},'configurations':configurations(c,ids),
        'items':[{'item_id':i,'content_sha256':'a'*64,'labels':list('ABCD')} for i in items],
        'runs':runs('fictional',ids,development),
        'schedule':{'item_order':list(items),'policy':c['execution']},'bindings':{}}


def emit(ledger, data):
    event = {'schema_version':SCHEMA,'manifest_sha256':fingerprint(ledger.manifest),
             **data,'previous_sha256':ledger.previous}
    event['event_sha256'] = fingerprint(event)
    ledger.apply(event)
    return event


def new_ledger(**kwargs):
    ledger = Ledger(manifest(**kwargs));emit(ledger,{'event':'plan'});return ledger


def start(ledger, tid):
    index = ledger.counts.get(tid,0)
    lineage = {'original_query_id':tid,'attempt_index':index,'attempt_id':attempt_id(tid,index),
               'parent_attempt_id':attempt_id(tid,index-1) if index else None}
    emit(ledger,{'event':'start',**lineage,'provider':ledger.provider(tid),
                 'request':{'path':'fictional/request.json','sha256':'a'*64},
                 'started_at':'2026-10-11T00:00:00+00:00'})
    return lineage


def finish(ledger, tid, *, status='valid', http=200):
    index = ledger.active[tid];target = ledger.targets[tid];cid = target['identity']['config_id']
    model = ledger.manifest['models'][cid]
    lineage = {'original_query_id':tid,'attempt_index':index,'attempt_id':attempt_id(tid,index),
               'parent_attempt_id':attempt_id(tid,index-1) if index else None}
    ref = {'path':'fictional/request.json','sha256':'a'*64}
    record = {**lineage,'identity':target['identity'],'source_namespace':CURRENT_NAMESPACE,
        'source_config_id':cid,'current_config_id':cid,'status':status,
        'selected_label':'A' if status == 'valid' else None,
        'failure_reason':None if status == 'valid' else 'fictional failure',
        'probability':empty_probability(),'probability_semantics':model['probability_semantics'],
        'provider_scalars':{k:{**measurement(),'semantics':v} for k,v in model['scalar_semantics'].items()},
        'usage':{k:measurement() for k in ('input_tokens','output_tokens','reasoning_tokens','final_tokens','billed_tokens')},
        'billing':{'observed_usd':measurement('0'),'estimated_usd':measurement(),'currency':'USD'},
        'timing':{'started_at':'2026-10-11T00:00:00+00:00','finished_at':'2026-10-11T00:00:01+00:00',
                  'duration_seconds':measurement(1),'queue_wait_seconds':measurement(0)},
        'route':{'requested_model':model['model_id'],'returned_model':model['expected_returned_model'],
                 'requested_route':model['route'],'returned_route':model['route']},
        'artifacts':{'request':ref,'raw_response':ref,'final':ref,'reasoning':None},
        'scheduler':{'provider':model['route'],'response_status':http,
            'stop_reason':'provider_rate_limit_stop' if http == 429 and ledger.streaks.get(model['route'],0) >= 2 else None}}
    return emit(ledger,{'event':'finish',**lineage,'record':record})


class NamespaceTests(unittest.TestCase):
    def test_all_ids_and_namespace_collision(self):
        self.assertEqual(len(CONFIGURATIONS),28)
        self.assertEqual(canonical_configuration(ORIGINAL_NAMESPACE,'J01')['config_id'],'D01')
        self.assertEqual(canonical_configuration(LUNA_NAMESPACE,'D01')['config_id'],'D02')
        self.assertEqual(canonical_configuration(DRAFT_NAMESPACE,'D12')['config_id'],'D13')
        self.assertEqual(canonical_configuration(CURRENT_NAMESPACE,'D01')['model_id'],'typesafe/jev-1.13')
        with self.assertRaises(ValueError): canonical_configuration('', 'D01')
        with self.assertRaises(ValueError): canonical_configuration(LUNA_NAMESPACE,'D01',model_id='typesafe/jev-1.13')
        with self.assertRaises(ValueError): canonical_configuration(CURRENT_NAMESPACE,'G01',effort='high')

    def test_historical_kimi_disabled_thinking(self):
        self.assertEqual(canonical_configuration(ORIGINAL_NAMESPACE,'C25',effort='none')['effort'],'instant')
        with self.assertRaises(ValueError): canonical_configuration(CURRENT_NAMESPACE,'G15',effort='none')


class NativeTests(unittest.TestCase):
    def test_all_eleven_routes_preserve_complete_inputs(self):
        c = config();q = question()
        for cid in EXTENSION_IDS:
            m = c['models'][cid];request = native.build_request(q,m,c['prompt'])
            self.assertEqual(request,request_payload(q,m,c['prompt']))
            self.assertIn(q.stem,request['state'])
            self.assertEqual(list(request['questions']['answer']['criteria']),list('ABCD'))
            self.assertEqual(request['provider']['only'],[m['provider_slug']])
            self.assertFalse(request['provider']['allow_fallbacks'])
            self.assertEqual(set(request),{'model','state','questions','provider'})
            for surface, endpoint in native.ENDPOINTS.items():
                native.validate_model({**m,'surface':surface,'endpoint':endpoint})

    def test_selected_label_is_not_argmax_and_scalar_is_separate(self):
        m = config()['models']['D03'];p = native.parse(response(m),m,'ABCD')
        self.assertEqual(p['selected_label'],'A')
        self.assertEqual(p['probability']['selected_probability'],'0.1')
        self.assertEqual(p['provider_scalars']['answers.answer.confidence']['value'],'17')
        self.assertIsNone(p['probability']['concentration'])

    def test_missing_or_bad_vector_does_not_retry_valid_answer(self):
        m = config()['models']['D03']
        for vector in (None,{}, {'A':True,'B':0,'C':0,'D':0}, {'A':1.03,'B':0,'C':0,'D':0}):
            p = native.parse(response(m,{'type':'choice','choice':'A','probabilities':vector}),m,'ABCD')
            self.assertEqual(p['status'],'valid');self.assertFalse(p['probability']['valid'])

    def test_inclusive_decimal_boundary(self):
        m = config()['models']['D03']
        raw = response(m)
        body = base64.b64decode(raw['body_base64']).decode().replace('0.1, "B": 0.2, "C": 0.3, "D": 0.4',
            '0.25, "B": 0.25, "C": 0.25, "D": 0.27')
        raw['body_base64'] = base64.b64encode(body.encode()).decode()
        self.assertTrue(native.parse(raw,m,'ABCD')['probability']['valid'])

    def test_drift_refusal_and_http_failure(self):
        m = config()['models']['D03']
        for key in ('model','provider'):
            self.assertEqual(native.parse(response(m,**{key:'wrong'}),m,'ABCD')['status'],'drift')
        self.assertEqual(native.parse(response(m,{'type':'refusal'}),m,'ABCD')['status'],'refused')
        raw = {'status':429,'headers':{'retry-after':'30'},'body_base64':base64.b64encode(b'Busy').decode()}
        self.assertEqual(native.parse(raw,m,'ABCD')['status'],'http_error')
        self.assertEqual(native.parse(response(m,answers=[]),m,'ABCD')['status'],'drift')

    def test_unknown_scalar_and_unsupported_controls(self):
        m = config()['models']['D03'];p = native.parse(response(m,score=2),m,'ABCD')
        self.assertEqual(p['provider_scalars']['score']['semantics'],'unreviewed')
        with self.assertRaises(ValueError): native.build_request(question(),{**m,'request_settings':{'seed':1}},config()['prompt'])
        with self.assertRaises(ValueError): native.validate_model({**m,'endpoint':'https://example.org'})


    def test_single_http_call_preserves_headers_and_bytes(self):
        m = config()['models']['D03'];wire = b'{"native":"response"}'
        class Reply:
            status = 200
            headers = {'X-Request-ID':'fixture','Retry-After':'5','Authorization':'must-not-save'}
            def __init__(self):
                self.closed = False
                self.fp = SimpleNamespace(raw=SimpleNamespace(_sock=MagicMock()))
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def isclosed(self): return self.closed
            def read1(self,size):
                self.closed = True
                return wire
        opener = MagicMock();opener.open.return_value = Reply()
        with patch.dict(os.environ,{'OPENROUTER_API_KEY':'fictional-key'}), patch('urllib.request.build_opener',return_value=opener):
            raw = native.dispatch(native.build_request(question(),m,config()['prompt']),m,600)
        opener.open.assert_called_once()
        self.assertEqual(base64.b64decode(raw['body_base64']),wire)
        self.assertEqual(raw['headers'],{'x-request-id':'fixture','retry-after':'5'})
        self.assertNotIn('fictional-key',json.dumps(raw))


class LedgerTests(unittest.TestCase):
    def test_33_runs_and_42009_outcomes(self):
        m = manifest(ids=EXTENSION_IDS,items=tuple(f'q{i}' for i in range(1273)))
        self.assertEqual(len(m['runs']),33);self.assertEqual(len(Ledger(m).targets),42009)

    def test_valid_answer_stops_independently_of_correctness(self):
        l = new_ledger();tid = l.candidates()[0];start(l,tid);finish(l,tid)
        self.assertIn(tid,l.selected)
        with self.assertRaises(ValueError): start(l,tid)

    def test_persisted_ten_retry_cap_and_three_runs(self):
        l = new_ledger();tid = l.candidates()[0]
        for _ in range(11):
            start(l,tid);finish(l,tid,status='invalid')
        self.assertEqual(l.counts[tid],11);self.assertEqual(l.selected[tid]['attempt_index'],10)
        with self.assertRaises(ValueError): start(l,tid)
        while l.candidates():
            t = l.candidates()[0];start(l,t);finish(l,t)
        self.assertTrue(l.coverage()['complete']);self.assertEqual(l.coverage()['terminal'],3)

    def test_initial_round_primary_recovery_and_secondary_barriers(self):
        l = new_ledger(items=('q1','q2'))
        first,second = l.candidates()
        start(l,first);finish(l,first,status='invalid')
        self.assertEqual(l.candidates(),[second])
        start(l,second)
        self.assertEqual(l.candidates(),[])
        finish(l,second)
        self.assertEqual(l.candidates(),[first])
        start(l,first);finish(l,first)
        self.assertTrue(all(l.targets[t]['role']=='secondary' for t in l.candidates()))

    def test_duplicate_and_indeterminate(self):
        l = new_ledger();tid = l.candidates()[0];start(l,tid)
        with self.assertRaises(ValueError): start(l,tid)
        self.assertEqual(l.coverage()['indeterminate'],[tid]);self.assertFalse(l.coverage()['complete'])

    def test_provider_pause_latched_and_other_provider_continues(self):
        l = new_ledger(ids=('D03','D08'),items=('q1','q2','q3','q4'))
        for _ in range(3):
            tid = next(t for t in l.candidates() if l.provider(t)=='SiliconFlow')
            start(l,tid);finish(l,tid,status='http_error',http=429)
        self.assertIn('SiliconFlow',l.paused)
        self.assertTrue(all(l.provider(t)=='Liquid' for t in l.candidates()))
        with self.assertRaises(ValueError): emit(l,{'event':'review','reviewer':'Oscar','reviewed_stop_events':[]})
        hashes = sorted(h for values in l.paused.values() for h in values)
        emit(l,{'event':'review','reviewer':'Oscar','reviewed_stop_events':hashes})
        self.assertFalse(l.paused)

    def test_ten_shared_workers_and_drift(self):
        l = new_ledger(ids=EXTENSION_IDS)
        for _ in range(10): start(l,l.candidates()[0])
        with self.assertRaises(ValueError): start(l,l.candidates()[0])
        tid = next(iter(l.active));finish(l,tid,status='drift')
        self.assertIn(l.provider(tid),l.drift)
        self.assertFalse(l.coverage()['complete'])

    def test_changed_parent_or_event_hash_rejected(self):
        l = new_ledger();tid = l.candidates()[0]
        with self.assertRaises(ValueError):
            emit(l,{'event':'start','original_query_id':tid,'attempt_index':0,
                    'attempt_id':attempt_id(tid,0),'parent_attempt_id':'bad','provider':l.provider(tid),
                    'request':{'path':'x','sha256':'a'*64}})
        with self.assertRaises(ValueError): l.apply({'event_sha256':'a'*64})

    def test_registry_rejects_second_destination_even_new_study_name(self):
        c = config();m = manifest()
        with tempfile.TemporaryDirectory() as temporary:
            c['execution']['registry'] = temporary
            with registry_claim(c,m,Path(temporary)/'one'): pass
            changed = copy.deepcopy(m);changed['study_id'] = 'second'
            with self.assertRaises(ValueError):
                with registry_claim(c,changed,Path(temporary)/'two'): pass

    def test_unverified_configuration_cannot_collect(self):
        c = config()
        c['models']['D03']['verification'] = {}
        self.assertEqual(config_errors(c),[])
        blockers = readiness(c,ids=['D03'],questions=[question()])
        self.assertTrue(any('schema' in b for b in blockers))
        self.assertTrue(any('input' in b for b in blockers))
        self.assertTrue(any('receipt' in b for b in blockers))


class PackageTests(unittest.TestCase):
    def test_readiness_uses_preflight_without_mandatory_tokenizer_audit(self):
        c = config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root/'receipt/source').mkdir(parents=True)
            (root/'receipt/source/config.toml').write_text(CONFIG.read_text())
            (root/'receipt/events.jsonl').write_text('fictional evidence')
            for cid in ('D03','D06','D07','D11'):
                model = c['models'][cid]
                model['verification'].update(route=True, controls=True, schema=True, full_input=True,
                    development_receipt='receipt', development_receipt_sha256=file_hash(root/'receipt/events.jsonl'))
                receipt = ({'provenance':'development','models':{cid:copy.deepcopy(model)}},
                    [{'identity':{'config_id':cid},'status':'valid','probability':{'valid':True}}],
                    {'complete':True})
                with patch('benchmarking.extension.read_extension_package',return_value=receipt):
                    def check(questions=None):
                        return readiness(c, ids=[cid], questions=questions, base=root)
                    blockers = check([question()])
                    self.assertEqual(blockers, [])
                    self.assertTrue(any('question file' in b for b in check()))
                    broken = Question('medqa','broken','development','',question().options)
                    self.assertTrue(any('preflight failed' in b for b in check([broken])))
                    model['verification']['input_audit'] = 'missing-audit.json'
                    self.assertTrue(any('audit unavailable' in b for b in check([question()])))

    def test_queue_wait_starts_at_phase_eligibility_and_resets_on_resume(self):
        ledger = new_ledger(items=('q1','q2'))
        queued = {}
        queue_phase(ledger, queued, 10)
        primary = list(ledger.candidates())
        self.assertEqual(set(queued), set(primary))
        first, second = primary
        start(ledger, first);finish(ledger, first, status='invalid')
        queued[first] = 20  # Retry wait starts when the previous attempt finishes.
        queue_phase(ledger, queued, 30)
        self.assertEqual(queued[first], 20)
        self.assertEqual(queued[second], 10)
        start(ledger, second);finish(ledger, second)
        start(ledger, first);finish(ledger, first)
        secondary = list(ledger.candidates())
        self.assertTrue(secondary)
        self.assertTrue(all(tid not in queued for tid in secondary))
        queue_phase(ledger, queued, 100)
        self.assertTrue(all(queued[tid] == 100 for tid in secondary))
        resumed = {}
        queue_phase(ledger, resumed, 500)
        self.assertEqual(set(resumed), set(secondary))
        self.assertTrue(all(value == 500 for value in resumed.values()))

    def test_request_duration_excludes_phase_queue_and_gate_wait(self):
        clock = {'now':100}
        def gate_enter(provider):
            clock['now'] = 105
        def transport(request, model, q):
            clock['now'] = 107
            return response(model)
        with patch('benchmarking.extension.time.monotonic',side_effect=lambda:clock['now']):
            result = worker({}, config()['models']['D03'], question(), True, transport, 90,
                            SimpleNamespace(enter=gate_enter))
        self.assertEqual(result['queue_wait_seconds'], 15)
        self.assertEqual(result['duration_seconds'], 2)

    def test_saved_development_resume_no_resend_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = root/'config.toml'
            cfg.write_text(CONFIG.read_text().replace('benchmarking/output/extension-registry',str(root/'registry')))
            questions = root/'questions.jsonl'
            write_new(questions,[question_record(question())],jsonl=True)
            public = root/'inference.json'
            write_new(public,{'schema_version':2,'type':'inference_input','provenance':'mock',
                'benchmark':'medqa','split':'development','count':1,'questions_sha256':file_hash(questions),
                'keys_sha256':'a'*64,'preparation_manifest_sha256':'b'*64,
                'ordered_ids':[question().item_id],'source_rows':[1]})
            protocol = root/'protocol.md';protocol.write_text('Fictional development protocol')
            output = root/'package';calls = []
            def transport(request,model,q):
                calls.append(request)
                return response(model)
            kwargs = dict(mock=True,development=True,ids=['D03'],transport=transport)
            first = collect(cfg,questions,public,protocol,output,**kwargs)
            self.assertTrue(first['complete']);self.assertEqual(len(calls),1)
            second = collect(cfg,questions,public,protocol,output,resume=True,**kwargs)
            self.assertTrue(second['complete']);self.assertEqual(len(calls),1)
            manifest,records,_ = read_extension_package(output)
            self.assertEqual(manifest['purpose'],'development')
            protocol.write_text('Changed protocol')
            with self.assertRaises(ValueError): collect(cfg,questions,public,protocol,output,resume=True,**kwargs)
            path = output/records[0]['artifacts']['raw_response']['path']
            path.write_text('{}')
            with self.assertRaises(ValueError): read_extension_package(output)

    def test_late_success_does_not_unlatch_provider_pause(self):
        ledger = new_ledger(items=('q1','q2','q3','q4'))
        active = list(ledger.candidates())
        for tid in active: start(ledger,tid)
        for tid in active[:3]: finish(ledger,tid,status='http_error',http=429)
        finish(ledger,active[3])
        self.assertIn('SiliconFlow',ledger.paused)
        self.assertEqual(ledger.candidates(),[])

    def test_mode_required_before_any_input_read(self):
        from benchmarking.extension import main
        with patch('sys.argv',['extension','collect']), patch('pathlib.Path.read_text',side_effect=AssertionError('read')):
            with self.assertRaises(SystemExit) as caught: main()
            self.assertNotEqual(caught.exception.code,0)

    def test_retry_after_seconds_and_http_date(self):
        from benchmarking.run import _retry_after_seconds
        self.assertEqual(_retry_after_seconds({'headers':{'retry-after':'30'}}),30)
        self.assertEqual(_retry_after_seconds({'headers':{'retry-after':'30'}},elapsed_seconds=12),18)
        self.assertEqual(_retry_after_seconds({'headers':{'retry-after':'bad'}}),0)


if __name__ == '__main__': unittest.main()
