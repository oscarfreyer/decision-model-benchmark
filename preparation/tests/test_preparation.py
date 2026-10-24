"""Preparation source/order/key and content-identity invariants; fictional only."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from common import ROOT, file_hash, load_keys, load_questions, question_record, strict_json, write_new
from preparation.prepare_medqa import prepare
from preparation.subset import subset

class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.dev=self.root/'source/dev.jsonl';self.test=self.root/'source/test.jsonl'
        def rows(name,count): return [{'question':f'Fictional {name} {i}','options':{x:f'{name} {i} {x}' for x in 'BACD'},'answer_idx':'A'} for i in range(count)]
        write_new(self.dev,rows('dev',6),jsonl=True);write_new(self.test,rows('test',2),jsonl=True)
        self.kwargs=dict(dev_source=self.dev,test_source=self.test,seed=818,source_url='https://example.invalid/fictional',
            rights_basis='fictional fixture',acquired_at='2026-10-04',count=4,expected_test_items=2,fictional=True,workspace=self.root)
        self.paths=prepare(**self.kwargs)
    def tearDown(self): self.temp.cleanup()
    def test_preparation_preserves_option_order_text_ids(self):
        qs=load_questions(self.paths['evaluation']);self.assertEqual(qs[0].labels,tuple('BACD'))
        self.assertEqual(qs[0].stem,'Fictional test 0');self.assertEqual(len(load_keys(self.paths['evaluation_keys'],qs)),2)
        self.assertFalse(any('answer' in q for q in [question_record(x) for x in qs]))
    def test_preparation_reproducible_without_overwrite(self):
        with self.assertRaises(FileExistsError): prepare(**self.kwargs)
        other=self.root/'other';paths=prepare(**{**self.kwargs,'workspace':other})
        self.assertEqual(file_hash(paths['questions']),file_hash(self.paths['questions']))
        self.assertEqual(file_hash(paths['keys']),file_hash(self.paths['keys']))
    def test_nonstandard_real_count_refused(self):
        with self.assertRaisesRegex(ValueError,'fictional'): prepare(**{**self.kwargs,'fictional':False,'workspace':self.root/'real'})
    def test_duplicate_overlap_refused(self):
        same=self.root/'same.jsonl';write_new(same,[strict_json(self.dev.read_text().splitlines()[0])]*2,jsonl=True)
        with self.assertRaisesRegex(ValueError,'duplicate|overlap'): prepare(**{**self.kwargs,'test_source':same,'workspace':self.root/'overlap'})
    def test_reference_label_mismatch_refused(self):
        raw=strict_json(self.dev.read_text().splitlines()[0]);raw['answer']='wrong';bad=self.root/'bad.jsonl';write_new(bad,[raw]*6,jsonl=True)
        with self.assertRaisesRegex(ValueError,'answer text'): prepare(**{**self.kwargs,'dev_source':bad,'workspace':self.root/'bad'})
    def test_subset_parent_mapping_reproducibility_overwrite(self):
        a=self.root/'sub1';b=self.root/'sub2'
        for out in (a,b): subset(self.paths['selection'],self.paths['questions'],self.paths['keys'],out,2)
        self.assertEqual(file_hash(a/'questions.jsonl'),file_hash(b/'questions.jsonl'))
        m=strict_json((a/'manifest.json').read_text());p=strict_json(self.paths['selection'].read_text())
        self.assertEqual(m['selected_source_rows'],p['selected_source_rows'][:2]);self.assertEqual(m['ordered_selected_ids'],p['ordered_selected_ids'][:2])
        self.assertEqual(m['parent_manifest_sha256'],file_hash(self.paths['selection']))
        with self.assertRaises(FileExistsError): subset(self.paths['selection'],self.paths['questions'],self.paths['keys'],a,2)

    def test_content_identity_survives_relocation_and_new_provenance(self):
        second=prepare(**{**self.kwargs,'workspace':self.root/'different','acquired_at':'2026-10-05','name':'another-name'})
        for role in ('questions','keys','evaluation','evaluation_keys','content'):
            self.assertEqual(file_hash(self.paths[role]),file_hash(second[role]))
        self.assertNotEqual(file_hash(self.paths['selection']),file_hash(second['selection']))
        for paths,name in ((self.paths,'s1'),(second,'s2')):
            subset(paths['selection'],paths['questions'],paths['keys'],self.root/name,2)
        self.assertEqual(file_hash(self.root/'s1/content-manifest.json'),file_hash(self.root/'s2/content-manifest.json'))
        self.assertNotEqual(file_hash(self.root/'s1/manifest.json'),file_hash(self.root/'s2/manifest.json'))
        different=prepare(**{**self.kwargs,'workspace':self.root/'different-seed','seed':17})
        self.assertNotEqual(file_hash(self.paths['content']),file_hash(different['content']))

    def test_primary_cli_prepares_parent_and_subset_without_second_script(self):
        workspace=self.root/'cli'
        command=[sys.executable,'-B','-m','preparation.prepare_medqa','--dev-source',str(self.dev),
            '--test-source',str(self.test),'--seed','818','--count','4','--subset-count','2',
            '--source-url','https://example.invalid/fictional','--rights-basis','fictional fixture',
            '--acquired-at','2026-10-05','--workspace',str(workspace),'--fictional','--expected-test-items','2']
        result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        parent=workspace/'preparation/output'
        sub=parent/'medqa-development-4-subset-2'
        self.assertTrue((sub/'content-manifest.json').is_file())
        self.assertEqual(len(load_questions(sub/'questions.jsonl')),2)
        self.assertEqual([strict_json(x) for x in (sub/'questions.jsonl').read_text().splitlines()],
                         [strict_json(x) for x in (parent/'questions/medqa-development-4.jsonl').read_text().splitlines()[:2]])
        before={str(p):file_hash(p) for p in parent.rglob('*') if p.is_file()}
        self.assertNotEqual(subprocess.run(command,cwd=ROOT,capture_output=True).returncode,0)
        self.assertEqual(before,{str(p):file_hash(p) for p in parent.rglob('*') if p.is_file()})
