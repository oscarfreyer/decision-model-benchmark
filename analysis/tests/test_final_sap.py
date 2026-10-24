"""Manual fictional/reference checks for final SAP methods; no research inputs."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from analysis.dataset.visual_review import (CANDIDATE_IDS, FIELDS, load_visual_review,
                                           review_agreement, validate_review_rows)
from analysis.methods.pipeline import paired_contrasts
from analysis.methods.statistics import confidence, newcombe_paired, operations, paired_counts
import numpy as np
from analysis.support.notebook import verify_notebook_release


def rating(item, a="retain", b="retain", final=None):
    adjudicate = a != b or a == "uncertain"
    final = final or ("retain" if adjudicate else a)
    return {"question_id": item, "OF_rating": a, "OF_reason": "Fictional independent review",
            "OF_date": "2026-10-08", "DG_rating": b, "DG_reason": "Fictional independent review",
            "DG_date": "2026-10-08", "MO_rating": final if adjudicate else "",
            "MO_reason": "Fictional adjudication" if adjudicate else "",
            "MO_date": "2026-10-08" if adjudicate else "", "final_decision": final}


class PairedIntervals(unittest.TestCase):
    def test_published_newcombe_example_and_reversed_orientation(self):
        # Independent worked reference [[10,5],[2,20]] from statsmodels docs:
        # https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.confint_proportions_paired.html
        cells = {"both_correct": 10, "positive_discordant": 5,
                 "negative_discordant": 2, "both_incorrect": 20}
        lo, hi = newcombe_paired(cells)
        self.assertAlmostEqual(lo, -6.43926050711, places=8)
        self.assertAlmostEqual(hi, 22.11160178958, places=8)
        reverse = newcombe_paired({**cells, "positive_discordant": 2, "negative_discordant": 5})
        self.assertAlmostEqual(reverse[0], -hi)
        self.assertAlmostEqual(reverse[1], -lo)

    def test_zero_discordance_and_extreme_margins_keep_score_uncertainty(self):
        for a, b in (([1]*10, [1]*10), ([0]*10, [0]*10), ([1, 0]*5, [1, 0]*5)):
            cells = paired_counts(a, b)
            self.assertEqual(sum(cells[k] for k in ("both_correct", "both_incorrect", "positive_discordant", "negative_discordant")), 10)
            lo, hi = newcombe_paired(cells)
            self.assertLess(lo, 0)
            self.assertGreater(hi, 0)
            self.assertEqual(cells["p_exact"], 1.)
        self.assertIsNone(newcombe_paired(paired_counts([], [])))

    def test_complete_valid_is_descriptive_and_preserves_undefined_draws(self):
        result = paired_contrasts({"A": np.array([1, 0]), "B": np.array([0, 1])},
                                  {"A": {"effort": "a"}, "B": {"effort": "b"}},
                                  [("S1", "A-B", "A", "B")], {"S1": 1}, seed=20261008,
                                  valid_matrix={"A": np.array([True, False]), "B": np.array([True, True])},
                                  inference=False)
        row = result["families"]["S1"][0]
        self.assertEqual((row["n"], row["difference_pp"]), (1, 100))
        self.assertNotIn("p_exact", row)
        self.assertNotIn("p_holm", row)
        self.assertIsNone(row["interval95_pp"])
        self.assertGreater(row["interval_undefined_resamples"], 0)


class ConfidenceReporting(unittest.TestCase):
    def test_constant_one_scores_brier_is_error_rate_with_degenerate_auroc(self):
        result = confidence([1]*4, [1, 0, 1, 0], calibration=True, seed=20261008)
        self.assertEqual(result["score_support"]["distinct_n"], 1)
        self.assertEqual(result["score_support"]["one_n"], 4)
        self.assertEqual(result["binary_brier"], .5)
        self.assertEqual(result["auroc"], .5)
        self.assertEqual(result["auroc_bootstrap"]["interval95"], [.5, .5])
        self.assertTrue(result["auroc_bootstrap"]["degenerate"])
        self.assertEqual(len(result["risk_coverage"]), 1)
        self.assertEqual(result["risk_coverage"][0]["coverage"], 1)
        self.assertEqual(result["reliability"][-1]["n"], 4)
        self.assertEqual(sum(b["n"] == 0 for b in result["reliability"]), 9)
        self.assertIsNotNone(result["reliability"][-1]["accuracy_wilson95"])

    def test_exact_bin_boundaries_and_probability_semantics_gate(self):
        scores = ["0", "0.1", "0.9", "1"]
        result = confidence(scores, [1, 1, 0, 0], calibration=True, seed=20261008)
        self.assertEqual([b["n"] for b in result["reliability"]], [1, 1, 0, 0, 0, 0, 0, 0, 0, 2])
        self.assertAlmostEqual(result["binary_brier"], (1+.81+.81+1)/4)
        gated = confidence(scores, [1, 1, 0, 0], seed=20261008)
        self.assertIsNone(gated["binary_brier"])
        self.assertIsNone(gated["reliability"])
        self.assertEqual(gated["auroc"], 0.)

    def test_stratified_auroc_single_class_and_empty_support(self):
        one_class = confidence([0, 1], [1, 1], calibration=True, seed=20261008)
        self.assertIsNone(one_class["auroc"])
        self.assertIsNone(one_class["auroc_bootstrap"]["interval95"])
        self.assertEqual(one_class["auroc_bootstrap"]["correct_n"], 2)
        empty = confidence([], [], calibration=True, seed=20261008)
        self.assertEqual(empty["score_support"]["distinct_n"], 0)
        self.assertIsNone(empty["binary_brier"])


class ReviewInput(unittest.TestCase):
    def test_kappa_uses_independent_ratings_and_counts_uncertain(self):
        pairs = [("retain", "retain")]*2 + [("exclude", "exclude")]*2 + [("retain", "exclude"), ("exclude", "retain")]
        rows = [rating(str(i), a, b) for i, (a, b) in enumerate(pairs)]
        result = review_agreement(rows)
        self.assertEqual(result["table"], [[2, 1, 0], [1, 2, 0], [0, 0, 0]])
        self.assertAlmostEqual(result["kappa"], 1/3)
        self.assertAlmostEqual(result["percentage_agreement"], 100*4/6)
        uncertain = review_agreement([rating("x", "uncertain", "uncertain")])
        self.assertEqual(uncertain["both_uncertain_n"], 1)
        self.assertIsNone(uncertain["kappa"])
        self.assertEqual(uncertain["kappa_reason"], "expected agreement is 1")

    def test_full_pool_and_adjudication_are_required(self):
        rows = [rating("x", "retain", "exclude", "exclude"), rating("y")]
        result = validate_review_rows(rows, ("x", "y"))
        self.assertEqual(result["excluded_ids"], ["x"])
        for changed in (rows[:-1], rows+[rows[0]],
                        [{**rows[0], "MO_reason": ""}, rows[1]],
                        [{**rows[0], "final_decision": "retain"}, rows[1]],
                        [rows[0], {**rows[1], "OF_date": "08/10/2026"}],
                        [rows[0], {**rows[1], "final_decision": "exclude"}]):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_review_rows(changed, ("x", "y"))

    def test_csv_requires_independent_digest_and_rejects_changed_bytes(self):
        # Real candidate IDs are identifiers only; all ratings are fictional.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"review.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader(); writer.writerows(rating(item) for item in CANDIDATE_IDS)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest = {"study_id": "fictional", "bindings": {"questions": "fictional"},
                        "items": [{"item_id": i} for i in CANDIDATE_IDS]}
            result = load_visual_review(path, digest, manifest)
            self.assertEqual(result["candidate_n"], 108)
            self.assertEqual(result["excluded_ids"], [])
            with self.assertRaises(ValueError):
                load_visual_review(path, None, manifest)
            path.write_bytes(path.read_bytes()+b"\n")
            with self.assertRaisesRegex(ValueError, "differs"):
                load_visual_review(path, digest, manifest)


class ResourceAccounting(unittest.TestCase):
    def test_final_attempt_cost_and_all_call_durations_have_distinct_bases(self):
        def call(cost, duration):
            return {"status": "valid", "billing": {"observed_usd": cost, "estimated_usd": cost},
                    "timing": {"request_wall_seconds": duration, "local_queue_wait_seconds": 0}, "usage": {}}
        earlier, final = call(9, 10), call(.2, 2)
        result = operations([final], 1, duration_records=[earlier, final])
        self.assertEqual(result["observed_usd"]["total"], .2)
        self.assertEqual(result["observed_usd"]["mean_per_attempt"], .2)
        self.assertEqual(result["latency"]["total_seconds"], 12)
        self.assertEqual(result["latency"]["calls_n"], 2)
        missing = operations([call(None, None)], 0)
        self.assertIsNone(missing["observed_usd"]["total"])
        self.assertIsNone(missing["latency"]["total_seconds"])


class AnalysisRelease(unittest.TestCase):
    def test_independent_human_release_binds_sources_methods_and_inputs(self):
        bindings = {"sap": {"version": "1.0", "sha256": "s"*64},
                    "analysis_code": {"analysis/Analysis.ipynb": "n"*64},
                    "executed_notebook_sha256": "n"*64,
                    "settings": {"seed": 20261008}, "input_manifest_sha256": "i"*64,
                    "visual_exclusions": {"policy": "fixed_nominated_108", "candidate_n": 108}, "environment": {"python": "fictional"}}
        record = {"schema_version": 1, "type": "flat_notebook_analysis_release",
                  "actor": "human", "approved": True, "reviewer": "Fictional reviewer",
                  "approved_at": "2026-10-08", "bindings": bindings}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"release.json"
            path.write_text(json.dumps(record))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            verify_notebook_release(path, digest, bindings)
            for field in ("sap", "analysis_code", "executed_notebook_sha256", "settings", "input_manifest_sha256", "visual_exclusions", "environment"):
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "differ from"):
                    verify_notebook_release(path, digest, {**bindings, field: None})
            with self.assertRaises(ValueError):
                verify_notebook_release(path, None, bindings)
            path.write_text(json.dumps({**record, "approved": False}))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "human approval"):
                verify_notebook_release(path, digest, bindings)


class FixedVisualNominations(unittest.TestCase):
    @staticmethod
    def manifest():
        # All input text and keys are fictional; only public nomination IDs recur.
        prefix = CANDIDATE_IDS[0].rsplit("-", 1)[0]+"-"
        return {"study_id": "fictional-study", "bindings": {"questions": "q"*64},
                "items": [{"item_id": prefix+f"{i:06d}"} for i in range(1, 1274)]}

    def test_exact_fixed_pool_counts_order_and_source_binding(self):
        from analysis.dataset.visual_review import fixed_visual_exclusions
        manifest = self.manifest()
        result = fixed_visual_exclusions(manifest)
        self.assertEqual(result["excluded_ids"], sorted(CANDIDATE_IDS))
        self.assertEqual(result["candidate_n"], 108)
        self.assertEqual(result["retained_n"], 1165)
        self.assertEqual(result["questions_sha256"], "q"*64)
        self.assertTrue(result["post_hoc"])
        manifest["items"].reverse()
        self.assertEqual(fixed_visual_exclusions(manifest), result)
        for removed in (CANDIDATE_IDS[0], manifest["items"][0]["item_id"]):
            missing = {**manifest, "items": [i for i in manifest["items"] if i["item_id"] != removed]}
            with self.assertRaises(ValueError):
                fixed_visual_exclusions(missing)
        changed = {**manifest, "items": [dict(i) for i in manifest["items"]]}
        candidate = next(i for i in changed["items"] if i["item_id"] == CANDIDATE_IDS[0])
        candidate["item_id"] = "fictional-nonnominated"
        with self.assertRaises(ValueError):
            fixed_visual_exclusions(changed)






if __name__ == "__main__":
    unittest.main()








