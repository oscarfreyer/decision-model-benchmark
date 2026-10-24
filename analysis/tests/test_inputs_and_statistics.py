"""Fictional checks for input integrity, study completeness and paired statistics.

No research data, credentials, inference, network or execution checkout needed.
"""
import json
from pathlib import Path
import tempfile
import unittest

from analysis.dataset.inputs import (MANIFEST, check_complete_study, folder_inventory,
                             sha256_file, verify_inventory)
from analysis.methods.core import CORE_ROSTER
from analysis.methods.statistics import exact_mcnemar, holm, paired_counts


class PortableInputs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "results/main").mkdir(parents=True)
        (self.root / "results/main/manifest.json").write_text(json.dumps({"execution_workflow": "lean-v1"}))
        (self.root / "questions.jsonl").write_text("fictional questions\n")
        (self.root / "keys.jsonl").write_text("fictional keys\n")
        self.data = {"schema_version": 1, "type": "analysis_input_inventory",
                     "packages": ["results/main"], "package_closure": ["results/main"],
                     "questions": "questions.jsonl", "keys": "keys.jsonl",
                     "files": folder_inventory(self.root)}
        self.save_inventory()

    def save_inventory(self):
        path = self.root / MANIFEST
        path.write_text(json.dumps(self.data))
        self.digest = sha256_file(path)

    def test_verified_inventory_contains_only_relative_references(self):
        result = verify_inventory(self.root, self.digest)
        self.assertEqual(result, self.data)
        self.assertNotIn(str(self.root), json.dumps(result))

    def test_missing_changed_and_extra_payloads_fail(self):
        for kind in ("missing", "changed", "extra"):
            with self.subTest(kind=kind):
                keys = self.root / "keys.jsonl"
                original = keys.read_bytes()
                if kind == "missing": keys.unlink()
                elif kind == "changed": keys.write_text("changed bytes\n")
                else: (self.root / "extra.txt").write_text("unexpected\n")
                with self.assertRaisesRegex(ValueError, "missing, unexpected or changed"):
                    verify_inventory(self.root, self.digest)
                keys.write_bytes(original)
                (self.root / "extra.txt").unlink(missing_ok=True)

    def test_independent_inventory_digest_is_required(self):
        for digest in (None, "0" * 64):
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                verify_inventory(self.root, digest)

    def test_continuations_cannot_escape_the_input_folder(self):
        p = self.root / "results/main/manifest.json"
        p.write_text(json.dumps({"execution_workflow": "lean-v1", "continuation": {"parent": "../../../outside"}}))
        self.data["files"] = folder_inventory(self.root)
        self.save_inventory()
        with self.assertRaisesRegex(ValueError, "parent escapes"):
            verify_inventory(self.root, self.digest)

    def test_symlinks_and_credential_filenames_are_rejected(self):
        link = self.root / "linked"
        link.symlink_to(self.root / "keys.jsonl")
        with self.assertRaisesRegex(ValueError, "symlink"):
            verify_inventory(self.root, self.digest)
        link.unlink()
        (self.root / ".env").write_text("fictional excluded credential file\n")
        with self.assertRaisesRegex(ValueError, "excluded"):
            verify_inventory(self.root, self.digest)


class ConsequentialStatistics(unittest.TestCase):
    def test_complete_design_requires_both_decision_models_and_repeats(self):
        runs = [{"config_id": cid, "experiment_id": "E1", "replicate": 1, "role": "primary"}
                for cid in CORE_ROSTER]
        runs += [{"config_id": cid, "experiment_id": "E2", "replicate": r, "role": "secondary"}
                 for cid in ("J01", "D01") for r in (2, 3)]
        manifest = {"items": [None] * 1273, "configurations": [{"config_id": cid} for cid in CORE_ROSTER], "runs": runs}
        coverage = {"complete": True, "expected": 26733, "terminal": 26733, "unstarted": [], "indeterminate": []}
        check_complete_study(manifest, [None] * 26733, coverage)
        with self.assertRaisesRegex(ValueError, "repeats"):
            check_complete_study({**manifest, "runs": runs[:-1]}, [None] * 26733, coverage)
        with self.assertRaisesRegex(ValueError, "terminal attempts"):
            check_complete_study(manifest, [None] * 26733, {**coverage, "complete": False})

    def test_exact_paired_results_have_known_answers(self):
        result = paired_counts([1, 1, 0, 0], [1, 0, 0, 0])
        self.assertEqual((result["positive_discordant"], result["negative_discordant"], result["difference_pp"]), (1, 0, 25))
        self.assertEqual(exact_mcnemar(0, 0), 1.0)
        self.assertEqual(exact_mcnemar(10, 0), 2 / 2**10)

    def test_holm_preserves_order_and_uses_the_entire_declared_family(self):
        self.assertEqual(holm([.04, .01, .03], 3), [.06, .03, .06])
        self.assertEqual(holm([.001] + [.5] * 30, 31), [.031] + [1.] * 30)
        with self.assertRaises(ValueError):
            holm([.01] * 30, 31)


if __name__ == "__main__":
    unittest.main()
