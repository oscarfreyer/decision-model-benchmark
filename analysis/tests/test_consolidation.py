"""Manual fictional checks for first-valid selection and six-file evidence loading.

The integration fixture uses the existing mock-only retry fixture. No private
dataset, execution checkout, credentials or provider transport is used.
"""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from benchmarking.common import attempt_identity, fingerprint, read_package
from benchmarking.retry_common import read_retry_package
from analysis.dataset.consolidate import consolidate, publish, verify_raw_snapshot
from analysis.dataset.flat import (FILES, RawEvidence, attempt_id,
                          select_attempts, verified_flat_records)
from analysis.dataset.inputs import folder_inventory, sha256_file, verify_inventory


def trial(index, status, label=None, *, item="fictional"):
    record = {"identity": {"benchmark": "fictional", "split": "test", "item_id": item,
                           "config_id": "C13", "run_id": "fictional-primary"},
              "status": status, "selected_label": label}
    return {"attempt_id": attempt_id(record, index), "retry_index": index,
            "target_id": fingerprint(list(attempt_identity(record))), "record": record,
            "run": {"config_id": "C13", "run_id": "fictional-primary", "experiment_id": "E1",
                    "replicate": 1, "role": "primary"}}


class SelectionRules(unittest.TestCase):
    def test_valid_wrong_original_stays_selected_without_keys(self):
        result = select_attempts([trial(0, "valid", "B")])[0]
        self.assertEqual((result["outcome"], result["selected_label"], result["selected_retry_index"]),
                         ("original_valid", "B", 0))

    def test_first_valid_retry_is_also_final_cost_attempt(self):
        rows = [trial(0, "http_error"), trial(1, "truncated"), trial(2, "valid", "B")]
        result = select_attempts(list(reversed(rows)))[0]
        self.assertEqual(result["selected_attempt_id"], rows[2]["attempt_id"])
        self.assertEqual(result["cost_attempt_id"], rows[2]["attempt_id"])
        self.assertEqual(result["duration_attempt_ids"], [r["attempt_id"] for r in rows])
        self.assertEqual(result["selected_label"], "B")

    def test_exhausted_failure_links_last_terminal_attempt(self):
        rows = [trial(i, "http_error") for i in range(11)]
        result = select_attempts(rows)[0]
        self.assertEqual(result["outcome"], "exhausted")
        self.assertIsNone(result["selected_label"])
        self.assertEqual(result["cost_attempt_id"], rows[-1]["attempt_id"])
        self.assertEqual(len(result["duration_attempt_ids"]), 11)

    def test_duplicate_gapped_incomplete_and_post_valid_retries_fail(self):
        cases = ([trial(0, "valid", "A")] * 2,
                 [trial(0, "http_error"), trial(2, "valid", "B")],
                 [trial(0, "http_error"), trial(1, "http_error")],
                 [trial(0, "valid", "B"), trial(1, "valid", "A")],
                 [trial(0, "http_error"), trial(1, "valid", "B"), trial(2, "valid", "A")])
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                select_attempts(rows)

    def test_repeat_role_cannot_be_changed(self):
        rows = [trial(0, "http_error"), trial(1, "valid", "A")]
        rows[1]["run"]["role"] = "secondary"
        with self.assertRaisesRegex(ValueError, "role"):
            select_attempts(rows)


class EvidenceRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from benchmarking.tests.test_retries import RetryTests, http_error
        from benchmarking.run import mock_dispatch
        cls.fixture = RetryTests()
        cls.fixture.setUp()
        cls.addClassCleanup(cls.fixture.tearDown)

        def transport(request, model, question):
            # One original GLM target exhausts; the others recover with a valid
            # wrong answer. Validity and scoring must stay independent.
            if model["model_id"] == "z-ai/glm-5" and question.item_id.endswith("000003"):
                return http_error(503)
            raw = mock_dispatch(request, model, question)
            import base64
            body = json.loads(base64.b64decode(raw["body_base64"]))
            if model["adapter"] == "openai_decisions_gateway":
                body["answers"]["answer"]["choice"] = "B"
            else:
                body["choices"][0]["message"]["content"] = '{"answer":"B"}'
            raw["body_base64"] = base64.b64encode(json.dumps(body).encode()).decode()
            return raw

        cls.fixture.collect(transport=transport)
        cls.snapshot = cls.fixture.root / "raw-snapshot"
        payload = cls.snapshot / "payload"
        for path in [*cls.fixture.sources, cls.fixture.output]:
            shutil.copytree(path, payload / path.name,
                            ignore=shutil.ignore_patterns(".lock", ".DS_Store"))
        # Preserve unreferenced evidence with different component/string orders.
        for name in ("saved-evidence/item.txt", "saved-evidence-v2/item.txt"):
            path = payload / cls.fixture.sources[0].name / name
            path.parent.mkdir(parents=True)
            path.write_bytes(name.encode())
        shutil.copyfile(cls.fixture.questions, payload / "questions.jsonl")
        shutil.copyfile(cls.fixture.paths["evaluation_keys"], payload / "keys.jsonl")
        original_count = sum(len(read_package(p)[1]) for p in cls.fixture.sources)
        _, _, recovery = read_retry_package(cls.fixture.output)
        cls.snapshot.joinpath("fixture-utility.py").write_bytes(b"# Fictional inventory utility\n")
        inventory = {"schema_version": 1, "type": "raw_collection_inventory", "payload_root": "payload",
            "files": sorted(folder_inventory(payload), key=lambda row: row["path"]),
            "assembly_tool": {"path": "fixture-utility.py",
                              "sha256": sha256_file(cls.snapshot / "fixture-utility.py")},
            "coverage": {"original_attempts": original_count, "retry_attempts": recovery["terminal"],
                         "recovered_targets": recovery["recovered"], "exhausted_targets": recovery["exhausted"]}}
        cls.snapshot.joinpath("raw_inventory.json").write_text(json.dumps(inventory))
        cls.digest = sha256_file(cls.snapshot / "raw_inventory.json")
        cls.output = cls.fixture.root / "flat-fixture"
        cls.source_metadata = {str(p.relative_to(cls.snapshot)): (p.stat().st_mode, p.stat().st_mtime_ns)
                               for p in cls.snapshot.rglob("*") if p.is_file()}
        cls.receipt = consolidate(cls.snapshot, cls.digest, cls.output,
            retry_package=cls.fixture.output.name, questions="questions.jsonl", keys="keys.jsonl",
            allow_fictional=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "copy"
        shutil.copytree(self.output, self.root)

    def test_exact_six_files_and_byte_identical_questions_keys(self):
        self.assertEqual({p.name for p in self.root.iterdir()}, set(FILES))
        for name in ("questions.jsonl", "keys.jsonl"):
            self.assertEqual((self.root / name).read_bytes(), (self.snapshot / "payload" / name).read_bytes())
        data = verify_inventory(self.root, self.receipt["input_manifest_sha256"])
        self.assertEqual(data["counts"], self.receipt["counts"])
        verify_raw_snapshot(self.snapshot, self.digest)
        self.assertEqual(self.source_metadata,
            {str(p.relative_to(self.snapshot)): (p.stat().st_mode, p.stat().st_mtime_ns)
             for p in self.snapshot.rglob("*") if p.is_file()})

    def test_relocation_complete_records_and_all_attempt_duration_references(self):
        manifest, terminal, coverage, keys, questions = verified_flat_records(self.root)
        self.assertEqual(len(terminal), coverage["expected"])
        self.assertTrue(coverage["complete"])
        self.assertEqual(len(keys), len(questions))
        self.assertEqual({c["config_id"] for c in manifest["configurations"]},
                         {"C03", "C05", "C06", "C09", "C11", "C13", "C14", "C15", "C17", "C18", "C19", "C22", "C23", "C24", "C25", "J01", "D01"})
        exhausted = [r for r in terminal if r["analysis_selection"]["outcome"] == "exhausted"]
        self.assertTrue(exhausted)
        self.assertEqual(len(exhausted), self.receipt["counts"]["exhausted_targets"])
        for row in exhausted:
            self.assertEqual(row["analysis_selection"]["selected_retry_index"], 10)
            self.assertEqual(len(row["all_attempt_timings"]), 11)
            self.assertIsNone(row["selected_label"])

    def test_changed_extra_missing_and_subfolder_inputs_fail(self):
        for kind in ("changed", "missing", "extra", "folder"):
            with self.subTest(kind=kind):
                keys = self.root / "keys.jsonl"
                before = keys.read_bytes()
                if kind == "changed": keys.write_bytes(before + b" ")
                if kind == "missing": keys.unlink()
                if kind == "extra": (self.root / "extra.txt").touch()
                if kind == "folder": (self.root / "extra").mkdir()
                with self.assertRaises(ValueError):
                    verify_inventory(self.root, self.receipt["input_manifest_sha256"])
                keys.write_bytes(before)
                (self.root / "extra.txt").unlink(missing_ok=True)
                if (self.root / "extra").exists(): (self.root / "extra").rmdir()

    def test_notebook_response_tables_accept_flat_and_historical_questions(self):
        from analysis.dataset.inputs import load_input_folder
        from analysis.support.notebook import response_tables
        from benchmarking.common import Question as FlatQuestion
        from common import Question as HistoricalQuestion, Option as HistoricalOption

        # The saved-evidence reader and notebook handoff run unchanged. Only the
        # complete-study population gate is bypassed for this three-item fixture.
        with patch("analysis.dataset.inputs.check_complete_study"):
            manifest, terminal, coverage, keys, questions, _ = load_input_folder(
                self.root, self.receipt["input_manifest_sha256"])
        self.assertTrue(all(type(q) is FlatQuestion for q in questions))
        tables = response_tables(manifest, terminal, coverage, keys, questions)
        item_ids = [q.item_id for q in questions]
        self.assertEqual([r["item_id"] for r in tables["answers"]], item_ids)
        self.assertEqual([r["item_id"] for r in tables["questions"]], item_ids)
        self.assertEqual([r["config_id"] for r in tables["models"]],
                         [c["config_id"] for c in manifest["configurations"]])
        selected = {attempt_identity(row): row for row in terminal}
        answers = {row["item_id"]: row for row in tables["answers"]}
        statuses = {row["item_id"]: row for row in tables["statuses"]}
        details = {row["item_id"]: row for row in tables["questions"]}
        repeats = {cid: {row["item_id"]: row for row in rows}
                   for cid, rows in tables["secondary_by_configuration"].items()}
        for question in questions:
            ident = (question.benchmark, question.split, question.item_id)
            key = keys[ident]["reference_answer"]
            self.assertEqual(answers[question.item_id]["correct_answer"], key)
            self.assertEqual(details[question.item_id]["question"], question.stem)
            for option in question.options:
                self.assertEqual(details[question.item_id]["option_" + option.label], option.text)
            self.assertEqual(details[question.item_id]["correct_answer_text"],
                             next(o.text for o in question.options if o.label == key))
            for run in manifest["runs"]:
                row = selected[(*ident, run["config_id"], run["run_id"])]
                expected = (row["selected_label"] if row["status"] == "valid"
                            else "[" + row["status"] + "]")
                if run["role"] == "primary":
                    self.assertEqual(answers[question.item_id][run["config_id"]], expected)
                    self.assertEqual(statuses[question.item_id][run["config_id"]], row["status"])
                else:
                    self.assertEqual(repeats[run["config_id"]][question.item_id][run["run_id"]], expected)
        historical = [HistoricalQuestion(q.benchmark, q.item_id, q.split, q.stem,
                      tuple(HistoricalOption(o.label, o.text) for o in q.options)) for q in questions]
        self.assertEqual(response_tables(manifest, terminal, coverage, keys, historical), tables)
        missing_key = dict(keys)
        missing_key.pop(next(iter(missing_key)))
        with self.assertRaisesRegex(ValueError, "exact key coverage"):
            response_tables(manifest, terminal, coverage, missing_key, questions)
        with self.assertRaisesRegex(ValueError, "preparation order"):
            response_tables(manifest, terminal, coverage, keys, list(reversed(questions)))

    def test_final_sap_scoring_sensitivities_and_resource_handoff(self):
        from analysis.methods.pipeline import analyze
        manifest, terminal, coverage, keys, _ = verified_flat_records(self.root)
        items = [q["item_id"] for q in manifest["items"]]
        visual = {"policy": "fixed_nominated_108", "post_hoc": True,
                  "excluded_ids": [items[0]], "candidate_n": 1, "retained_n": 2,
                  "primary_n": 3, "study_id": manifest["study_id"],
                  "questions_sha256": manifest["bindings"]["questions"]}
        # Only population/pool size is mocked for the three-question fictional
        # study; production pool validation has a separate 1,273/108/1,165 check.
        with patch("analysis.dataset.visual_review.fixed_visual_exclusions", return_value=visual):
            summary = analyze(manifest, terminal, coverage, keys, seed=20261008,
                              methods={"normalization": "native_selected_normalized_distribution",
                                       "primary_family": "decision_vs_generative_31",
                                       "probability_semantics_verified": True,
                                       "visual_exclusion_policy": "fixed_nominated_108",
                                       "decision_models": {"D01": {"probability_semantics_verified": True}}})
        self.assertEqual(len(summary["paired"]["families"]["S1"]), 31)
        self.assertEqual(len(summary["paired"]["families"]["S2"]), 3)
        # Failed GLM originals and wrong recovered GLM answers both remain wrong.
        self.assertEqual(summary["primary"]["C22"]["planned_n"], 3)
        self.assertEqual(summary["primary"]["C22"]["correct"], 0)
        self.assertEqual(summary["first_attempt_sensitivity"]["C22"]["correct"], 0)
        filtered = summary["missing_visual_sensitivity"]
        self.assertEqual(filtered["excluded_ids"], [items[0]])
        self.assertEqual(filtered["common_n"], 2)
        self.assertEqual(len(filtered["accuracy"]), 21)
        self.assertTrue(all(r["n"] == 2 for r in filtered["accuracy"].values()))
        for source in (filtered["paired"], summary["complete_valid_sensitivity"]):
            for family in source["families"].values():
                self.assertTrue(all("p_exact" not in r and "p_holm" not in r for r in family))
        for cid, primary in summary["primary"].items():
            calls = sum(len(r["analysis_attempts"]) for r in terminal
                        if r["identity"]["config_id"] == cid and r["analysis_selection"]["run"]["role"] == "primary")
            self.assertEqual(primary["operations"]["latency"]["calls_n"], calls)
        self.assertEqual(set(summary["repeat_operations"]), {"J01:run2", "J01:run3", "D01:run2", "D01:run3"})
        self.assertNotIn("visual_review", summary)
        self.assertEqual(filtered["policy"], "fixed_nominated_108")
        # Save the actual fictional pipeline output, including native-vector and
        # bootstrap fields, then verify it can be reloaded without recalculation.
        from analysis.support.notebook import save_results, load_results
        _, path, digest = save_results(self.root.parent/"numerical-results", summary, {},
                                      {"fixture": True}, {"sha256": "a"*64})
        loaded = load_results(path, digest)
        self.assertEqual(loaded["summary"], summary)
        self.assertEqual(len(loaded["figure_data"]["accuracy"]), 17)
        self.assertEqual(len(loaded["figure_data"]["s1_paired_effects"]), 31)

    def test_temporary_parent_alias_preserves_package_identity(self):
        physical = self.root.parent / "physical"
        restored = physical / "restored"
        restored.mkdir(parents=True)
        alias = self.root.parent / "temporary-parent-alias"
        alias.symlink_to(physical, target_is_directory=True)
        # Model macOS /var -> /private/var in the temporary directory's parent.
        with patch("analysis.dataset.flat.tempfile.TemporaryDirectory") as temporary:
            temporary.return_value.__enter__.return_value = str(alias / "restored")
            _, terminal, coverage, _, _ = verified_flat_records(self.root)
        self.assertEqual(coverage["flat_counts"], self.receipt["counts"])
        self.assertEqual(len(terminal), self.receipt["counts"]["selected_outcomes"])

    def test_recomputed_local_manifest_cannot_hide_changed_selected_outcome(self):
        path = self.root / "selected_responses.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["cost_attempt_id"] = "0" * 64
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        data = json.loads((self.root / "input_manifest.json").read_text())
        data["files"] = folder_inventory(self.root)
        (self.root / "input_manifest.json").write_text(json.dumps(data))
        verify_inventory(self.root, sha256_file(self.root / "input_manifest.json"))
        with self.assertRaisesRegex(ValueError, "first-valid/final-attempt"):
            verified_flat_records(self.root)

    def test_changed_raw_artifact_cannot_be_hidden_by_a_new_flat_inventory(self):
        import base64
        path = self.root / "all_attempts.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        value = rows[0]["artifacts"]["request"]
        rows[0]["artifacts"]["request"] = base64.b64encode(base64.b64decode(value) + b" ").decode()
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        data = json.loads((self.root / "input_manifest.json").read_text())
        data["files"] = folder_inventory(self.root)
        (self.root / "input_manifest.json").write_text(json.dumps(data))
        verify_inventory(self.root, sha256_file(self.root / "input_manifest.json"))
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            verified_flat_records(self.root)

    def test_existing_destination_and_invalid_source_digest_create_nothing(self):
        with self.assertRaisesRegex(ValueError, "existing output"):
            consolidate(self.snapshot, self.digest, self.root)
        destination = self.root.parent / "new"
        with self.assertRaisesRegex(ValueError, "Raw inventory"):
            consolidate(self.snapshot, "0" * 64, destination)
        self.assertFalse(destination.exists())

    def test_publication_failure_removes_only_created_files(self):
        destination = self.root.parent / "failed"
        real_hash = sha256_file
        def changed_hash(path):
            return "0" * 64 if Path(path).parent == destination else real_hash(path)
        with patch("analysis.dataset.consolidate.sha256_file", side_effect=changed_hash):
            with self.assertRaisesRegex(ValueError, "Published file"):
                publish(self.root, destination)
        self.assertFalse(destination.exists())


class RawProofs(unittest.TestCase):
    def test_raw_inventory_uses_full_relative_filename_order(self):
        with tempfile.TemporaryDirectory() as folder:
            snapshot = Path(folder)
            payload = snapshot / "payload"
            for name in ("run/item.json", "run-v2/item.json"):
                path = payload / name
                path.parent.mkdir(parents=True)
                path.write_bytes(name.encode())
            utility = snapshot / "fixture-utility.py"
            utility.write_bytes(b"# Fictional inventory utility\n")
            files = sorted(folder_inventory(payload), key=lambda row: row["path"])
            self.assertEqual([row["path"] for row in files],
                             ["run-v2/item.json", "run/item.json"])
            inventory = {"schema_version": 1, "type": "raw_collection_inventory",
                "payload_root": "payload", "files": files,
                "assembly_tool": {"path": utility.name, "sha256": sha256_file(utility)}}
            manifest = snapshot / "raw_inventory.json"
            manifest.write_text(json.dumps(inventory))
            digest = sha256_file(manifest)
            evidence, _, _ = verify_raw_snapshot(snapshot, digest)
            self.assertEqual(evidence.inventory["files"], files)
            # Sorting must not weaken content-integrity checks.
            (payload / "run/item.json").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "missing, changed or unexpected"):
                verify_raw_snapshot(snapshot, digest)

    def test_invalid_embedded_bytes_path_escape_and_source_hash_fail(self):
        from analysis.dataset.flat import decode_bytes
        with self.assertRaises(ValueError):
            decode_bytes("not base64!")
        import hashlib
        payload = b"fictional exact bytes\r\n"
        inventory = json.dumps({"schema_version": 1, "type": "raw_collection_inventory", "payload_root": "payload",
            "files": [{"path": "source.txt", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}]}).encode()
        evidence = RawEvidence(inventory, hashlib.sha256(inventory).hexdigest())
        self.assertEqual(evidence.check("source.txt", payload), payload)
        for name, data in (("../source.txt", payload), ("source.txt", payload + b" ")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                evidence.check(name, data)


if __name__ == "__main__":
    unittest.main()
