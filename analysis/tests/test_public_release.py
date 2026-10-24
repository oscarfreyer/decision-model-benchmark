"""Fictional boundary checks for the public derivative; no research inputs."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from analysis.public_release import checked_data, upstream_keys, validate_public_manifest, validate_public_record


class PublicDataBoundary(unittest.TestCase):
    def test_changed_or_escaping_data_cannot_be_authenticated(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'provenance').mkdir();payload=root/'fictional.json'
            payload.write_text('fictional')
            entry={'files':{'fictional.json':hashlib.sha256(payload.read_bytes()).hexdigest()}}
            (root/'provenance/data_manifest.json').write_text(json.dumps(entry))
            checked_data(root)
            payload.write_text('changed')
            with self.assertRaises(ValueError):checked_data(root)
            entry['files']={'../outside.json':'0'*64}
            (root/'provenance/data_manifest.json').write_text(json.dumps(entry))
            with self.assertRaises(ValueError):checked_data(root)

    def test_upstream_keys_require_exact_source_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'fictional.jsonl';path.write_text('{}\n')
            with self.assertRaises(ValueError):upstream_keys(path,{'source_test_sha256':'0'*64})

    def test_unknown_status_or_identity_cannot_enter_scoring(self):
        manifest={'_public_item_ids':{'fictional-1'},'_public_runs':{'fictional-run':'D01'}}
        row={'identity':{'benchmark':'medqa','split':'evaluation','item_id':'fictional-1','config_id':'D01','run_id':'fictional-run'},'status':'valid','selected_label':'A'}
        validate_public_record(row,manifest)
        row['status']='invented_status'
        with self.assertRaises(ValueError):validate_public_record(row,manifest)
        row['status']='valid';row['identity']['config_id']='D02'
        with self.assertRaises(ValueError):validate_public_record(row,manifest)

    def test_incomplete_roster_cannot_change_analysis_families(self):
        from benchmarking.configuration_ids import CURRENT_NAMESPACE
        manifest={'public_derivative':'structured-results-v1','configuration_namespace':CURRENT_NAMESPACE,'configurations':[{'config_id':'D01'}]}
        with self.assertRaises(ValueError):validate_public_manifest(manifest)
