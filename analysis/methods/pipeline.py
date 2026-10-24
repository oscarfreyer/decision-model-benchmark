"""Explicit-path, offline consumer of the manifest-bound v3 attempt contract."""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
from fractions import Fraction
import json
from pathlib import Path
import sys

import numpy as np

from analysis.support.source_record import analysis_code_record
from common import (LEGACY_ROSTER, BUDGET_ROSTERS, KIMI_INSTANT_DECISION, SCHEMA_VERSION, attempt_identity, effort_pair_levels, file_hash, fingerprint, identity,
                    load_keys, load_questions, question_hash, read_package, strict_json,
                    validate_finish as validate_historical_finish,
                    validate_manifest as validate_historical_manifest, write_new)
from analysis.methods.statistics import (bootstrap_differences, confidence, fraction, holm, operations,
                                 paired_counts, newcombe_paired, pareto, probability_change, quantiles, wilson)

NORMALIZATION_POLICIES = ("preserve_native", "divide_by_sum", "native_selected_normalized_distribution")
COST_IMPUTATION_POLICY = "D01_run1_mean_two_refusals_20261009"
COST_IMPUTATION_ITEMS = {
    "medqa-us-evaluation-c3b905ccfa66152d-000032",
    "medqa-us-evaluation-c3b905ccfa66152d-000825",
}
COST_IMPUTATION_RUN = "medqa-luna-20261007-E1-D01-r1"
SONNET_COST_POLICY = "C09_run1_mean_one_refusal_20261009"
SONNET_COST_RUN = "medqa-main-20261006-E1-C09-r1"
SONNET_COST_ITEM = "medqa-us-evaluation-c3b905ccfa66152d-000742"
KIMI_DURATION_POLICY = "C23_run1_mean_one_interrupted_call_20261009"
KIMI_DURATION_RUN = "medqa-main-20261006-E1-C23-r1"
KIMI_DURATION_ITEM = "medqa-us-evaluation-c3b905ccfa66152d-000341"
KIMI_DURATION_ATTEMPT = "c174c07cd97788e773f8ccff7288b9b4d7af758e37f2bfb505f622f4f083666c"

PREPRINT_PAIRS = {"P03": ("C05", "C06"), "P07": ("C13", "C14"), "P09": ("C17", "C18")}


def validate_manifest(manifest):
    """Use the package's pure contract, retaining historical scheduler semantics."""
    if manifest.get("execution_workflow") == "analysis-reporting-ids-v1":
        from analysis.dataset.expanded import validate_view
        return validate_view(manifest)
    if manifest.get("execution_workflow") == "lean-analysis-joined-v1":
        from analysis.methods.core import validate_join
        return validate_join(manifest)
    if manifest.get("execution_workflow") in {"lean-v1", "lean-decisions-v1"}:
        from benchmarking.common import validate_manifest as validate_execution_manifest
        return validate_execution_manifest(manifest)
    return validate_historical_manifest(manifest)


def validate_finish(record, manifest, **kwargs):
    if manifest.get("execution_workflow") == "analysis-reporting-ids-v1":
        from analysis.dataset.expanded import validate_mapped_record
        return validate_mapped_record(record, manifest)
    if manifest.get("execution_workflow") == "lean-analysis-joined-v1":
        from analysis.methods.core import finish_source
        source = finish_source(record, manifest)
        return validate_finish(record, source, **kwargs)
    if manifest.get("execution_workflow") in {"lean-v1", "lean-decisions-v1"}:
        from benchmarking.common import validate_finish as validate_execution_finish
        return validate_execution_finish(record, manifest, **kwargs)
    return validate_historical_finish(record, manifest, **kwargs)


def expected_identities(manifest):
    """Generate identities after validation by the package-specific reader."""
    validate_manifest(manifest)
    return [(manifest['benchmark'], manifest['split'], item['item_id'], run['config_id'], run['run_id'])
            for run in manifest['runs'] for item in manifest['items']]


def family_sizes(configs, pairs):
    """Exact prespecified roster/pair identities, never a chosen subset family."""
    if set(configs) == set(LEGACY_ROSTER) and set(pairs) == {f"P{i:02d}" for i in range(1, 11)}:
        return {"S1": 24, "S2": 10}
    if any(set(configs) == set(roster) for roster in BUDGET_ROSTERS.values()) and set(pairs) == set(PREPRINT_PAIRS):
        return {"S1": 15, "S2": 3}
    raise ValueError("Paired families require exact historical 25 or budget-amended 16 roster")


def flat_record(record):
    """Flatten contract measurements for pure kernels; retain native vectors unchanged."""
    value = lambda section, field: record[section][field]["value"]
    usage = {k: value("usage", k) for k in ("input_tokens", "output_tokens", "reasoning_tokens", "final_tokens", "billed_tokens")}
    usage.update({k+"_missing_reason": record["usage"][k]["reason"] for k in ("input_tokens", "output_tokens", "reasoning_tokens", "final_tokens", "billed_tokens")})
    usage["accounting"] = {"definition": record["usage"]["accounting_definition"],
                           "reasoning_in_output": record["usage"]["reasoning_in_output"]}
    billing = {k: value("billing", k) for k in ("observed_usd", "estimated_usd")}
    billing.update(provenance=record["billing"]["rate_provenance"], rate_date=record["billing"]["rate_date"],
                   observed_usd_missing_reason=record["billing"]["observed_usd"]["reason"],
                   estimated_usd_missing_reason=record["billing"]["estimated_usd"]["reason"])
    probability = dict(record["probability"])
    for field, scalar in record.get("provider_scalars", {}).items():
        probability[field] = scalar["value"]
        probability[field+"_reason"] = scalar.get("reason")
    if "provider_confidence" in record:
        probability.update(provider_confidence=record["provider_confidence"]["value"],
                           provider_confidence_reason=record["provider_confidence"]["reason"])
    return {**record, "probability": probability, "usage": usage, "billing": billing,
            "timing": {**record["timing"], "request_wall_seconds": float(value("timing", "duration_seconds")) if value("timing", "duration_seconds") is not None else None,
                       "missing_reason": record["timing"]["duration_seconds"]["reason"],
                       "local_queue_wait_seconds": float(record["timing"]["queue_wait_seconds"]["value"]) if record["timing"].get("queue_wait_seconds", {}).get("value") is not None else None,
                       "queue_wait_missing_reason": record["timing"].get("queue_wait_seconds", {}).get("reason", "not_recorded_in_historical_package")}}


def review_file(path, checksum, manifest, *, kind, resolver=None):
    if not path:
        if checksum:
            raise ValueError("A review checksum requires its file")
        return None
    if not checksum or file_hash(path) != checksum:
        raise ValueError("Review manifest requires matching explicit SHA-256")
    result = strict_json(Path(path).read_text())
    if result.get("schema_version") != 2 or result.get("type") != kind:
        raise ValueError("Unsupported review manifest")
    if result.get("protocol_sha256") != manifest["bindings"]["protocol_record"]:
        raise ValueError("Review protocol binding mismatch")
    if result.get("accepted") is not True or not result.get("reviewer") or not result.get("accepted_at"):
        raise ValueError("Explicit dated human method review required")
    evidence = result.get("review_evidence")
    if not evidence or set(evidence) != {"path", "sha256"}:
        raise ValueError("Checksum-bound human review evidence required")
    evidence_path = (resolver.resolve(evidence["path"], evidence["sha256"], parent=Path(path).parent)
                     if resolver else Path(path).parent / evidence["path"])
    if file_hash(evidence_path) != evidence["sha256"]:
        raise ValueError("Human review evidence hash mismatch")
    result["input_sha256"] = checksum
    return result


def validated_exclusions(review, manifest):
    # A published flag or model error only nominates review. Apply exclusions only
    # after independent, source-backed adjudication bound to these exact questions.
    if review is None:
        return None
    if review.get("study_id") != manifest["study_id"] or review.get("questions_sha256") != manifest["bindings"]["questions"]:
        raise ValueError("Exclusions must bind this study and question file")
    for field in ("reviewer_assignment", "evidence_standard", "nomination_rule"):
        if not review.get(field):
            raise ValueError("Adjudication setup unresolved: " + field)
    if review.get("independent_source_review") is not True or review.get("model_identity_blinded") is not True or review.get("model_outcomes_blinded") is not True:
        raise ValueError("Independent source review, model-identity and correctness-pattern blinding required")
    if type(review.get("outcome_triggered_nomination")) is not bool or type(review.get("post_outcome_amendment")) is not bool:
        raise ValueError("Explicit outcome-triggered and post-outcome amendment disclosure required")
    allowed = {q["item_id"] for q in manifest["items"]}
    exclusions = review.get("exclusions")
    if not isinstance(exclusions, list) or len({x["item_id"] for x in exclusions}) != len(exclusions):
        raise ValueError("Invalid/duplicate exclusion identities")
    for exclusion in exclusions:
        if exclusion["item_id"] not in allowed or not exclusion.get("reason") or not exclusion.get("source_evidence"):
            raise ValueError("Exclusions require known IDs, reasons and source evidence")
        if exclusion.get("basis") != "independently_adjudicated_reference_defect":
            raise ValueError("Published flags/outcome errors alone cannot authorize exclusions")
    return {x["item_id"] for x in exclusions}


def vector_for_metrics(record, policy, provenance):
    # Convert an eligible native vector only for derived metrics; the saved raw
    # vector remains untouched. Real scores need an accepted normalization policy.
    if record["status"] != "valid":
        return None, "invalid_final_label"
    p = record["probability"]
    if not p["valid"]:
        return None, p["reason"]
    values = {k: Fraction(str(v)) for k, v in p["native_vector"].items()}
    total = sum(values.values())
    if policy is None:
        if provenance != "mock":
            return None, "normalization_policy_unaccepted"
        if total != 1:
            return None, "fictional_kernels_require_unit_sum"
        return values, None
    if policy == "preserve_native":
        return values, None
    if policy in ("divide_by_sum", "native_selected_normalized_distribution"):
        return {k: v/total for k, v in values.items()}, None
    raise ValueError("Unsupported normalization policy; use a human-reviewed supported policy")


def roster_check(manifest):
    if manifest.get("execution_workflow") == "analysis-reporting-ids-v1":
        from analysis.methods.expanded import PAIRS, ROSTER
        configs = {c["config_id"]: c for c in manifest["configurations"]}
        return configs, PAIRS, set(configs) == set(ROSTER)
    if manifest.get("execution_workflow") == "lean-analysis-joined-v1":
        from analysis.methods.core import CORE_ROSTER
        original = next(m for m in manifest["source_manifests"] if m["execution_workflow"] == "lean-v1")
        _, pairs, full = roster_check(original)
        configs = {c["config_id"]: c for c in manifest["configurations"]}
        if not full or set(configs) != CORE_ROSTER:
            raise ValueError("Current core roster requires exactly 17 configurations")
        d01 = configs["D01"]
        if (d01["adapter"] != "openai_decisions_gateway" or d01["pair_id"] is not None
                or d01["effort"].lower() != "native"):
            raise ValueError("D01 native decision identity mismatch")
        return configs, pairs, True
    configs = {c["config_id"]: c for c in manifest["configurations"]}
    members, pairs = {}, {}
    for config in configs.values():
        if config["pair_id"] is not None:
            members.setdefault(config["pair_id"], []).append(config)
    for pair_id, pair in members.items():
        if len(pair) != 2:
            raise ValueError("Incomplete effort pair")
        # Stable odd/even IDs bind contrast direction independently of native
        # vocabulary, including historical Low/High v3 packages.
        a, b = sorted(pair, key=lambda c: c["config_id"])
        effort_pair_levels(pair_id, a["effort"].lower(), b["effort"].lower())
        expected_lower = 2 * int(pair_id[1:]) - 1
        if (a["config_id"], b["config_id"]) != (f"C{expected_lower:02d}", f"C{expected_lower+1:02d}"):
            raise ValueError("Roster pair IDs/effort direction do not match accepted C01-C20 order")
        if (a["base_model"], a["model_id"], a["adapter"]) != (b["base_model"], b["model_id"], b["adapter"]):
            raise ValueError("Effort pairs must share base model/route adapter")
        pairs[pair_id] = {"lower": a["config_id"], "upper": b["config_id"]}
    if len(configs) != len(manifest["configurations"]):
        raise ValueError("Duplicate roster configuration identity")
    legacy = set(configs) == set(LEGACY_ROSTER)
    decision = manifest.get('roster_decision')
    preprint = any(set(configs) == set(roster) for roster in BUDGET_ROSTERS.values())
    full = legacy or preprint
    if legacy:
        for i in range(1, 21, 2):
            lower, upper = configs[f"C{i:02d}"], configs[f"C{i+1:02d}"]
            if lower["pair_id"] != f"P{(i+1)//2:02d}" or lower["pair_id"] != upper["pair_id"]:
                raise ValueError("Roster pair IDs/effort direction do not match accepted C01-C20 order")
        if any(configs[k]["pair_id"] is not None for k in ("C21", "C22", "C23", "C24", "J01")):
            raise ValueError("Standard/native entries must not form additional effort pairs")
        if manifest.get("roster_decision") is not None:
            raise ValueError("Historical roster cannot carry the preprint budget decision")
    elif preprint:
        if decision not in BUDGET_ROSTERS or set(configs) != set(BUDGET_ROSTERS[decision]):
            raise ValueError("Budget-amended roster requires its exact decision binding")
        for pair_id, (lower, upper) in PREPRINT_PAIRS.items():
            if any(configs[k]["pair_id"] != pair_id for k in (lower, upper)):
                raise ValueError("Budget-amended roster pair IDs mismatch")
            expected_upper = "xhigh" if pair_id == "P09" else "high"
            if (configs[lower]["effort"].lower(), configs[upper]["effort"].lower()) != ("low", expected_upper):
                raise ValueError("Budget-amended roster requires its accepted native effort labels")
        paired_ids = {cid for pair in PREPRINT_PAIRS.values() for cid in pair}
        if any(configs[k]["pair_id"] is not None for k in set(configs)-paired_ids):
            raise ValueError("Single retained arms must have null pair membership")
        singleton_efforts = {"C03": "low", "C09": "low", "C11": "low", "C15": "low", "C19": "none",
                            "C21": "standard", "C22": "native", "C23": "native", "C24": "standard", "J01": "native"}
        if decision == KIMI_INSTANT_DECISION:
            singleton_efforts.pop('C21')
            singleton_efforts['C25'] = 'none'
            if (configs['C25']['model_id'], configs['C25']['base_model']) != ('moonshotai/kimi-k2.5', 'moonshotai/kimi-k2.5'):
                raise ValueError('Kimi Instant replacement requires K2.5 identity')
        if any(configs[cid]["effort"].lower() != effort for cid, effort in singleton_efforts.items()):
            raise ValueError("Single retained arms require their accepted native effort labels")
    elif manifest.get("roster_decision") is not None:
        raise ValueError("Budget decision cannot bind an arbitrary roster subset")
    if full:
        family_sizes(configs, pairs)
    if manifest["provenance"] == "definitive" and (not full or len(manifest["items"]) != 1273 or manifest["split"] != "evaluation"):
        raise ValueError("Definitive preprint requires original 1273 test items and exact accepted 16/25 roster")
    return configs, pairs, full


def paired_family(matrix, configs, pairs, *, seed, valid_matrix=None, inference=True):
    """Only named families; no all-pairs or pooling of replicates."""
    # Column direction is JEV minus comparator (S1), upper minus lower (S2):
    # High/Low, Xhigh/Low or High/None, retaining native effort labels.
    # Primary zeros include completed failures; valid_matrix restricts only the
    # separately labelled sensitivity to pairs with two valid final answers.
    contrasts = [("S1", k, "J01", k) for k in sorted(configs) if k != "J01"]
    contrasts += [("S2", k, pair["upper"], pair["lower"]) for k, pair in sorted(pairs.items())]
    sizes = family_sizes(configs, pairs)
    return paired_contrasts(matrix, configs, contrasts, sizes, seed=seed, valid_matrix=valid_matrix, inference=inference)


def paired_contrasts(matrix, configs, contrasts, sizes, *, seed, valid_matrix=None, inference=True):
    """Evaluate explicitly specified families using shared question bootstrap draws."""
    if any(sum(c[0] == family for c in contrasts) != size for family, size in sizes.items()):
        raise ValueError("Paired family coverage differs from prespecified roster")
    tables = {family: [] for family in sizes}
    differences = np.column_stack([matrix[a]-matrix[b] for _, _, a, b in contrasts])
    eligibility = None if valid_matrix is None else np.column_stack([valid_matrix[a]&valid_matrix[b] for _, _, a, b in contrasts])
    boot = bootstrap_differences(differences, seed=seed, eligibility=eligibility)
    for i, (family, contrast_id, a, b) in enumerate(contrasts):
        mask = np.ones(len(matrix[a]), dtype=bool) if valid_matrix is None else valid_matrix[a]&valid_matrix[b]
        row = {"contrast_id": contrast_id, "positive_configuration": a, "negative_configuration": b,
               "positive_effort": configs[a]["effort"], "negative_effort": configs[b]["effort"],
               **paired_counts(matrix[a][mask], matrix[b][mask]),
               "denominator": "all_original_key_questions" if valid_matrix is None else "paired_complete_valid"}
        if boot:
            row.update(interval95_pp=boot["intervals_pp"][i], interval_degenerate=bool(boot["degenerate"][i]),
                       bootstrap_constant=boot["constant_distribution"][i],
                       interval_undefined_resamples=boot.get("undefined_resamples", [0]*len(contrasts))[i])
        row["newcombe95_pp"] = newcombe_paired(row) if inference else None
        if not inference:
            row.pop("p_exact")
        tables[family].append(row)
    if not inference:
        return {"families": tables, "bootstrap": {k: v for k, v in boot.items() if k not in ("intervals_pp", "degenerate", "constant_distribution")},
                "multiplicity": "descriptive sensitivity; no hypothesis tests"}
    for family, rows in tables.items():
        # Preserve each full prespecified family. An undefined contrast does not
        # license shrinking the family or reporting a partial Holm adjustment.
        if all(row["p_exact"] is not None for row in rows):
            adjusted = holm([row["p_exact"] for row in rows], sizes[family])
            for row, p in zip(rows, adjusted):
                row.update(p_holm=p, reject_at_005=p <= .05)
        else:
            for row in rows:
                row.update(p_holm=None, reject_at_005=None, adjustment_gate="undefined contrast; family cannot be adjusted")
    return {"families": tables, "bootstrap": {k: v for k, v in (boot or {}).items() if k not in ("intervals_pp", "degenerate", "constant_distribution")},
            "multiplicity": "Holm separately within "+" and ".join(f"{k}={v}" for k, v in sizes.items())+"; no global multiplicity control or equivalence claim"}


def change_distribution(values):
    """Descriptive absolute changes with exact zero counts and a whole-tie ECDF."""
    values = list(values)
    counts = Counter(values)
    cumulative, ecdf = 0, []
    for value, n in sorted(counts.items()):
        cumulative += n
        ecdf.append({"change": float(value), "count": n, "cumulative_n": cumulative,
                     "fraction": cumulative/len(values)})
    return {"eligible_n": len(values), "unchanged_n": counts.get(0, 0),
            "unchanged_rate": fraction(counts.get(0, 0), len(values)),
            "mean": float(sum(values)/len(values)) if values else None,
            **quantiles(float(v) for v in values), "ecdf": ecdf}


def stability(manifest, runs, records, keys, coverage, policy, *, config_id="J01",
              vector_reader=vector_for_metrics):
    """Question-matched repeats for one decision service; defaults preserve JEV."""
    service_runs = [r for r in runs if r["config_id"] == config_id]
    by_replicate = {r["replicate"]: r for r in service_runs}
    if len(by_replicate) != len(service_runs):
        raise ValueError("Duplicate decision-service replicate")
    if set(by_replicate) != {1, 2, 3}:
        return {"gate": ("E2 requires planned JEV replicates 2 and 3 in addition to primary run1" if config_id == "J01"
                         else f"{config_id} stability requires planned replicates 1, 2 and 3")}
    items = [q["item_id"] for q in manifest["items"]]
    missing = {tuple(x): "unstarted" for x in coverage["unstarted"]}
    missing.update({tuple(x): "indeterminate" for x in coverage["indeterminate"]})
    key = lambda item, rep: (manifest["benchmark"], manifest["split"], item, config_id, by_replicate[rep]["run_id"])
    def state(item, rep):
        ident = key(item, rep)
        row = records.get(ident)
        if row is None:
            return coverage.get("missing_outcome_reasons", {}).get(fingerprint(list(ident)), missing[ident])
        if row["status"] != "valid":
            return row["status"]
        return "correct" if row["selected_label"] == keys[ident[:3]]["reference_answer"] else "incorrect"
    per_run, pairs = {}, []
    for rep, run in sorted(by_replicate.items()):
        rows = [records[key(item, rep)] for item in items if key(item, rep) in records]
        correct = sum(state(item, rep) == "correct" for item in items)
        complete = len(rows) == len(items)
        per_run[str(rep)] = {"run_id": run["run_id"], "role": run["role"], "complete": complete,
                             "accuracy": fraction(correct, len(items)) if complete else None,
                             "wilson95": wilson(correct, len(items)) if complete else None,
                             "correct_observed": correct, "planned_n": len(items), "terminal_n": len(rows),
                             "valid_output_rate": fraction(sum(r["status"] == "valid" for r in rows), len(items)) if complete else None,
                             "state_counts": dict(Counter(state(item, rep) for item in items)),
                             "service_dates": sorted({r["timing"]["started_at"][:10] for r in rows})}
    for a, b in ((1, 2), (1, 3), (2, 3)):
        # Transitions include every planned item; answer agreement uses valid
        # pairs only. Probability changes have their own stricter eligibility.
        valid_pairs, agreements, changes, selected_changes = 0, 0, [], []
        excluded = Counter()
        transitions = Counter()
        for item in items:
            transitions[state(item, a)+" -> "+state(item, b)] += 1
            x, y = records.get(key(item, a)), records.get(key(item, b))
            if x is None or y is None or x["status"] != "valid" or y["status"] != "valid":
                excluded["missing_or_invalid_label_pair"] += 1
                continue
            valid_pairs += 1; agreements += x["selected_label"] == y["selected_label"]
            vx, rx = vector_reader(x, policy, manifest["provenance"])
            vy, ry = vector_reader(y, policy, manifest["provenance"])
            if vx is None or vy is None:
                excluded[rx or ry] += 1
            elif sum(vx.values()) != 1 or sum(vy.values()) != 1:
                excluded["accepted_native_policy_not_unit_sum"] += 1
            else:
                change = probability_change(vx, vy, x["selected_label"])
                if policy == "native_selected_normalized_distribution":
                    label = x["selected_label"]
                    change["earlier_selected_probability_change"] = abs(
                        Fraction(str(x["probability"]["native_vector"][label]))-
                        Fraction(str(y["probability"]["native_vector"][label])))
                changes.append(change)
                selected_changes.append(abs(Fraction(str(x["probability"]["native_vector"][x["selected_label"]]))-
                                            Fraction(str(y["probability"]["native_vector"][y["selected_label"]]))))
        pairs.append({"earlier_run": a, "later_run": b, "valid_pair_n": valid_pairs, "agreements": agreements,
                      "answer_agreement": fraction(agreements, valid_pairs), "transitions": dict(sorted(transitions.items())),
                      "vector_eligible_n": len(changes), "vector_exclusions": dict(excluded),
                      "total_variation": {"mean": float(np.mean([x["total_variation"] for x in changes])) if changes else None,
                                          **quantiles(x["total_variation"] for x in changes)},
                      "earlier_selected_probability_change": {"mean": float(np.mean([x["earlier_selected_probability_change"] for x in changes])) if changes else None,
                                                               **quantiles(x["earlier_selected_probability_change"] for x in changes)},
                      "selected_probability_change": {**change_distribution(selected_changes),
                          "exclusions": dict(excluded)},
                      "confidence_change_distribution": {
                          "earlier_selected_probability": change_distribution(
                              [x["earlier_selected_probability_change"] for x in changes]),
                          "total_variation": change_distribution([x["total_variation"] for x in changes])},
                      "uncertainty": "descriptive; stability interval method not specified"})
    triples = [[records.get(key(item, rep)) for rep in (1, 2, 3)] for item in items]
    valid = [rows for rows in triples if all(r is not None and r["status"] == "valid" for r in rows)]
    agreements = sum(len({r["selected_label"] for r in rows}) == 1 for rows in valid)
    accuracies = [r["accuracy"] for r in per_run.values() if r["accuracy"] is not None]
    trajectories = Counter(tuple(r["selected_label"] if r is not None and r["status"] == "valid"
                                 else "No valid answer" for r in rows) for rows in triples)
    selected_triples, vector_triples = [], []
    for rows in valid:
        vectors = [vector_reader(r, policy, manifest["provenance"])[0] for r in rows]
        if all(v is not None and sum(v.values()) == 1 for v in vectors):
            native = [{k: Fraction(str(v)) for k, v in r["probability"]["native_vector"].items()} for r in rows]
            selected_triples.append(len({v[r["selected_label"]] for r, v in zip(rows, native)}) == 1)
            vector_triples.append(native[0] == native[1] == native[2])
    triple_agreement = lambda matches: {"eligible_n": len(matches), "unchanged_n": sum(matches),
        "rate": fraction(sum(matches), len(matches)), "excluded_n": len(items)-len(matches)}
    return {"planned_n": len(items), "runs": per_run, "pairs": pairs,
            "answer_flows": {"planned_n": len(items), "run_order": [1, 2, 3],
                "trajectories": [{"labels": list(labels), "n": n} for labels, n in sorted(trajectories.items())]},
            "confidence_three_way": {"selected_probability": triple_agreement(selected_triples),
                "native_vector": triple_agreement(vector_triples)},
            "three_way": {"valid_triple_n": len(valid), "agreements": agreements, "rate": fraction(agreements, len(valid))},
            "descriptive_accuracy_mean": float(np.mean(accuracies)) if len(accuracies) == 3 else None,
            "descriptive_accuracy_range": [min(accuracies), max(accuracies)] if len(accuracies) == 3 else None,
            "interpretation": "dependent question-matched runs; stable wrong answers remain errors; E2 never replaces E1"}


def failure_summary(rows):
    n = len(rows)
    return {"n": n, "status_counts": dict(Counter(r["status"] for r in rows)),
            "failure_reasons": dict(Counter(r.get("failure_reason") or "unspecified" for r in rows if r["status"] != "valid")),
            "truncated_n": sum(r["status"] == "truncated" for r in rows),
            "truncated_rate": fraction(sum(r["status"] == "truncated" for r in rows), n)}


def run_operations(rows, correct, *, complete_ledger=True, indeterminate_n=0, unstarted_n=0, selected_final=False):
    """Final-attempt billing, full-call durations and separate collection segments."""
    calls, grouped = [], {}
    originals = []
    for row in rows:
        originals.append(flat_record(row["original_attempt"]) if "original_attempt" in row else row)
        history = row.get("analysis_attempts")
        entries = history if history is not None else [{"record": None, "retry_index": 0, "source_package": "original_saved_package"}]
        for entry in entries:
            call = flat_record(entry["record"]) if entry["record"] is not None else row
            calls.append(call)
            provider = call.get("scheduler", {}).get("provider") or call.get("route", {}).get("returned_route") or "not_recorded"
            group = (entry["source_package"], "retry" if entry["retry_index"] else "original", str(provider))
            grouped.setdefault(group, []).append(call)
    op = operations(rows, correct, complete_ledger=complete_ledger, indeterminate_n=indeterminate_n,
                    unstarted_n=unstarted_n, duration_records=rows if selected_final else calls)
    op["cost_basis"] = "selected final attempt per original query"
    op["duration_basis"] = "all preserved original/retry calls" if any("analysis_attempts" in r for r in rows) else "saved original calls"
    if selected_final:
        op["duration_basis"] = "same selected final attempt per question as cost"
        op["all_call_latency"] = operations([], 0, duration_records=calls)["latency"]
    op["original_failures"], op["final_failures"] = failure_summary(originals), failure_summary(rows)
    op["retry_calls_n"] = sum(len(r.get("analysis_attempts", []))-1 for r in rows if "analysis_attempts" in r)
    op["recovered_queries_n"] = sum(r.get("analysis_selection", {}).get("outcome") == "recovered" for r in rows)
    op["exhausted_queries_n"] = sum(r.get("analysis_selection", {}).get("outcome") == "exhausted" for r in rows)
    from datetime import datetime
    op["segments"] = []
    for (package, phase, provider), segment in sorted(grouped.items()):
        duration = operations([], 0, complete_ledger=complete_ledger, duration_records=segment)
        starts = [r["timing"]["started_at"] for r in segment]
        finishes = [r["timing"]["finished_at"] for r in segment if r["timing"]["request_wall_seconds"] is not None]
        op["segments"].append({"source_package": package, "phase": phase, "provider": provider,
                               "first_start": min(starts, key=datetime.fromisoformat),
                               "last_measured_finish": max(finishes, key=datetime.fromisoformat) if finishes else None,
                               "unmeasured_finish_n": len(segment)-len(finishes),
                               "service_dates": sorted({s[:10] for s in starts}),
                               "latency": duration["latency"], "queue_wait": duration["local_queue_wait"],
                               "failures": failure_summary(segment)})
    if selected_final:
        final_groups = {}
        for row in rows:
            selection = row.get('analysis_selection', {})
            selected_index = selection.get('selected_retry_index', 0)
            history = row.get('analysis_attempts', [])
            entry = next((a for a in history if a['retry_index'] == selected_index), {})
            package = entry.get('source_package', row.get('source_package', 'unrecorded'))
            provider = row.get('scheduler', {}).get('provider') or row.get('route', {}).get('returned_route') or 'unrecorded'
            final_groups.setdefault((str(package), str(provider)), []).append(row)
        op['selected_final_segments'] = [{'source_package': package, 'provider': provider,
            'service_dates': sorted({r['timing']['started_at'][:10] for r in group}),
            'resources': operations(group, 0, complete_ledger=complete_ledger),
            'failures': failure_summary(group)} for (package, provider), group in sorted(final_groups.items())]
    op["service_dates"] = sorted({r["timing"]["started_at"][:10] for r in calls})
    op["backoff"] = {"available": False, "reason": "No separately measured backoff duration in the saved attempt contract"}
    return op


def apply_primary_cost_imputation(summaries, rows_by_config, policy):
    """Add a labelled cost basis without changing observed receipts or responses."""
    if policy != COST_IMPUTATION_POLICY:
        raise ValueError("Unsupported cost-imputation policy")
    row = summaries.get("D01")
    rows = rows_by_config.get("D01", [])
    if (row is None or row["run_id"] != COST_IMPUTATION_RUN or not row["complete"]
            or row["planned_n"] != 1273 or len(rows) != 1273):
        raise ValueError("Cost imputation requires the complete approved D01 primary run")
    if (len({r["identity"]["item_id"] for r in rows}) != 1273
            or any(r["identity"]["config_id"] != "D01"
                   or r["identity"]["run_id"] != COST_IMPUTATION_RUN for r in rows)):
        raise ValueError("Cost-imputation rows disagree with the approved primary population")
    missing = [r for r in rows if r["billing"]["observed_usd"] is None]
    if ({r["identity"]["item_id"] for r in missing} != COST_IMPUTATION_ITEMS
            or len(missing) != 2
            or any(r["status"] != "http_error"
                   or r["billing"].get("observed_usd_missing_reason") != "provider_charge_unavailable"
                   or r.get("analysis_selection", {}).get("outcome") != "exhausted"
                   or r["analysis_selection"].get("selected_retry_index") != 10 for r in missing)):
        raise ValueError("Cost imputation requires exactly the two approved exhausted refusals")
    donors = [r for r in rows if r["billing"]["observed_usd"] is not None]
    values = [Decimal(str(r["billing"]["observed_usd"])) for r in donors]
    if (len(values) != 1271 or any(r["status"] != "valid" for r in donors)
            or any(not v.is_finite() or v < 0 for v in values)):
        raise ValueError("Cost-imputation donors require 1271 valid observed final-attempt charges")
    # Fractions preserve the exact decimal subtotal and the unrounded mean.
    subtotal = sum((Fraction(v) for v in values), Fraction(0))
    mean = subtotal/len(values)
    observed = row["operations"]["observed_usd"]
    if (observed["observed_n"] != 1271 or observed["missing_n"] != 2
            or observed["observed_subtotal"] != float(subtotal)):
        raise ValueError("Observed cost summary disagrees with imputation donors")
    for result in summaries.values():
        receipt = result["operations"]["observed_usd"]
        result["operations"]["analysis_usd"] = {
            **receipt, "basis": "observed final-attempt billing",
            "imputed_n": 0, "imputed_subtotal": 0.0,
            "unresolved_n": receipt["missing_n"],
        }
    total = subtotal + len(missing)*mean
    adjusted = row["operations"]["analysis_usd"]
    adjusted.update(basis="observed billing plus approved mean imputation",
                    complete=True, total=float(total), mean_per_attempt=float(total/1273),
                    cost_per_correct=float(total/row["correct"]) if row["correct"] else None,
                    cost_per_correct_reason=None if row["correct"] else "zero correct answers",
                    imputed_n=2, imputed_subtotal=float(2*mean), unresolved_n=0,
                    imputation={"policy": policy, "donor_n": len(values),
                                "donor_subtotal": float(subtotal), "mean_usd": float(mean),
                                "mean_usd_exact": str(mean),
                                "items": [{"item_id": r["identity"]["item_id"],
                                           "cost_attempt_id": r["analysis_selection"]["cost_attempt_id"],
                                           "assigned_usd": float(mean)}
                                          for r in sorted(missing, key=lambda r: r["identity"]["item_id"])],
                                "timing": "post hoc after result inspection; approved 2026-10-09"})
    return {"policy": policy, "scope": "D01 primary run only",
            "frontier_basis": "analysis_usd", "repeat_costs": "unchanged observed billing"}


def apply_sonnet_cost_imputation(summaries, rows_by_config, policy):
    """One approved C09 final charge; keep raw receipts and repeat costs intact."""
    rows = rows_by_config.get("C09", [])
    result = summaries.get("C09")
    if (policy != SONNET_COST_POLICY or result is None or not result["complete"]
            or result["planned_n"] != 1273 or result["run_id"] != SONNET_COST_RUN
            or len(rows) != 1273 or len({r["identity"]["item_id"] for r in rows}) != 1273
            or any(r["identity"]["config_id"] != "C09" or r["identity"]["run_id"] != SONNET_COST_RUN for r in rows)):
        raise ValueError("Sonnet cost imputation requires the approved complete C09 primary run")
    missing = [r for r in rows if r["billing"]["observed_usd"] is None]
    if (len(missing) != 1 or missing[0]["identity"]["item_id"] != SONNET_COST_ITEM
            or missing[0]["status"] != "refused"
            or missing[0]["billing"].get("observed_usd_missing_reason") != "provider_charge_unavailable"
            or missing[0].get("analysis_selection", {}).get("outcome") != "exhausted"
            or missing[0]["analysis_selection"].get("selected_retry_index") != 10):
        raise ValueError("Sonnet cost imputation requires the single approved exhausted refusal")
    donors = [r for r in rows if r["billing"]["observed_usd"] is not None]
    values = [Decimal(str(r["billing"]["observed_usd"])) for r in donors]
    if (len(values) != 1272 or any(r["status"] != "valid" for r in donors)
            or any(not v.is_finite() or v < 0 for v in values)):
        raise ValueError("Sonnet cost donors require 1272 valid nonnegative final charges")
    subtotal = sum((Fraction(v) for v in values), Fraction(0))
    mean = subtotal/len(values)
    receipt = result["operations"]["observed_usd"]
    if (receipt["observed_n"] != 1272 or receipt["missing_n"] != 1
            or receipt["observed_subtotal"] != float(subtotal)):
        raise ValueError("Sonnet observed summary disagrees with imputation donors")
    for row in summaries.values():
        raw = row["operations"]["observed_usd"]
        row["operations"].setdefault("analysis_usd", {**raw, "basis": "observed final-attempt billing",
            "imputed_n": 0, "imputed_subtotal": 0.0, "unresolved_n": raw["missing_n"]})
    total = subtotal+mean
    result["operations"]["analysis_usd"].update(
        basis="observed billing plus approved mean imputation", complete=True,
        total=float(total), mean_per_attempt=float(total/1273),
        cost_per_correct=float(total/result["correct"]) if result["correct"] else None,
        cost_per_correct_reason=None if result["correct"] else "zero correct answers",
        imputed_n=1, imputed_subtotal=float(mean), unresolved_n=0,
        imputation={"policy": policy, "donor_n": len(values), "donor_subtotal": float(subtotal),
            "mean_usd": float(mean), "mean_usd_exact": str(mean),
            "items": [{"item_id": SONNET_COST_ITEM,
                "cost_attempt_id": missing[0]["analysis_selection"]["cost_attempt_id"], "assigned_usd": float(mean)}],
            "timing": "post hoc after result inspection; approved 2026-10-09"})
    return {"policy": policy, "scope": "C09 primary run only", "frontier_basis": "analysis_usd",
            "repeat_costs": "unchanged observed billing"}


def apply_kimi_duration_imputation(summaries, rows_by_config, policy):
    """Impute one original C23 call from all observed original/retry durations."""
    result = summaries.get("C23")
    rows = rows_by_config.get("C23", [])
    if (policy != KIMI_DURATION_POLICY or result is None or not result["complete"]
            or result["planned_n"] != 1273 or result["run_id"] != KIMI_DURATION_RUN
            or len(rows) != 1273 or len({r["identity"]["item_id"] for r in rows}) != 1273
            or any(r["identity"]["config_id"] != "C23" or r["identity"]["run_id"] != KIMI_DURATION_RUN for r in rows)):
        raise ValueError("Kimi duration imputation requires the approved complete C23 primary run")
    calls = []
    for row in rows:
        history = row.get("analysis_attempts", [])
        attempt_ids = row.get("analysis_selection", {}).get("duration_attempt_ids", [])
        if len(history) != len(attempt_ids):
            raise ValueError("Kimi duration history and attempt IDs disagree")
        calls.extend({**entry, "attempt_id": aid} for aid, entry in zip(attempt_ids, history))
    if (len(calls) != 1275 or len({c["attempt_id"] for c in calls}) != len(calls)
            or any(c["record"]["identity"]["config_id"] != "C23"
                   or c["record"]["identity"]["run_id"] != KIMI_DURATION_RUN for c in calls)):
        raise ValueError("Kimi duration imputation requires all 1275 unique primary calls")
    missing = [c for c in calls if c["record"]["timing"]["duration_seconds"]["value"] is None]
    if (len(missing) != 1 or missing[0]["attempt_id"] != KIMI_DURATION_ATTEMPT
            or missing[0]["retry_index"] != 0
            or missing[0]["record"]["identity"]["item_id"] != KIMI_DURATION_ITEM
            or missing[0]["record"]["status"] != "transport_error"
            or missing[0]["record"].get("failure_reason") != "IncompleteRead"
            or missing[0]["record"]["timing"]["duration_seconds"]["reason"] != "interrupted_response_timing_unavailable"):
        raise ValueError("Kimi duration imputation requires the approved interrupted original call")
    values = [Decimal(str(c["record"]["timing"]["duration_seconds"]["value"]))
              for c in calls if c["record"]["timing"]["duration_seconds"]["value"] is not None]
    if len(values) != 1274 or any(not v.is_finite() or v < 0 for v in values):
        raise ValueError("Kimi duration donors require 1274 finite nonnegative call durations")
    subtotal = sum((Fraction(v) for v in values), Fraction(0))
    mean = subtotal/len(values)
    raw = result["operations"]["latency"]
    if raw["missing_n"] != 1 or raw["calls_n"] != 1275 or raw["n"] != 1274:
        raise ValueError("Kimi duration summary disagrees with observed calls")
    for row in summaries.values():
        latency = row["operations"]["latency"]
        row["operations"]["analysis_latency"] = {**latency, "basis": "observed all-call durations",
            "imputed_n": 0, "unresolved_n": latency["missing_n"]}
    adjusted = result["operations"]["analysis_latency"]
    adjusted.update(quantiles([float(v) for v in values]+[float(mean)]))
    adjusted.update(basis="observed all-call durations plus approved mean imputation",
        complete=True, total_seconds=float(subtotal+mean), imputed_n=1, unresolved_n=0,
        imputation={"policy": policy, "donor_n": len(values), "mean_seconds": float(mean),
            "mean_seconds_exact": str(mean), "donor_subtotal_seconds": float(subtotal),
            "assigned_seconds": float(mean), "attempt_id": KIMI_DURATION_ATTEMPT,
            "item_id": KIMI_DURATION_ITEM, "retry_index": 0,
            "timing": "post hoc after result inspection; approved 2026-10-09"})
    return {"policy": policy, "scope": "one original C23 primary call",
            "frontier_basis": "analysis_latency", "repeat_durations": "unchanged observed durations"}


def analyze(manifest, terminal, coverage, keys, *, seed, methods=None, exclusions_review=None,
            visual_review=None, field_review_path=None, input_hashes=None):
    validate_manifest(manifest)
    expanded = manifest.get("execution_workflow") == "analysis-reporting-ids-v1"
    if expanded:
        from analysis.methods.expanded import RESOURCE_POLICY
        if (seed != 20261008 or not methods or methods.get('resource_imputation') != RESOURCE_POLICY
                or methods.get('normalization') != 'native_selected_normalized_distribution'
                or any(k in methods for k in ('cost_imputation','sonnet_cost_imputation','duration_imputation'))):
            raise ValueError('SAP 1.8 requires accepted seed, common final-attempt resource policy and native probabilities')
        if manifest['provenance'] == 'definitive':
            from analysis.dataset.expanded import decision_settings
            expected_decision_settings = decision_settings(
                manifest, field_review_path=field_review_path, input_hashes=input_hashes)
            if methods.get('decision_models') != expected_decision_settings:
                raise ValueError('Decision field-review settings differ from verified source evidence')
    configs, pairs, full_roster = roster_check(manifest)
    if set(keys) != {(manifest["benchmark"], manifest["split"], q["item_id"]) for q in manifest["items"]}:
        raise ValueError("Analysis keys must cover exactly the planned questions")
    if methods and methods.get("normalization") not in NORMALIZATION_POLICIES:
        raise ValueError("Unsupported reviewed normalization policy")
    if methods and "bootstrap_seed" in methods and methods["bootstrap_seed"] != seed:
        raise ValueError("Bootstrap seed differs from reviewed method")
    policy = methods.get("normalization") if methods else None
    exclusions = validated_exclusions(exclusions_review, manifest)
    records = {}
    expected = set(expected_identities(manifest))
    item_labels = {q["item_id"]: q["labels"] for q in manifest["items"]}
    for record in terminal:
        ident = attempt_identity(record)
        if ident in records:
            raise ValueError("Duplicate terminal identity")
        validate_finish(record, manifest, allowed_ids=expected, item_labels=item_labels)
        records[ident] = flat_record(record)
    unstarted, indeterminate = {tuple(x) for x in coverage["unstarted"]}, {tuple(x) for x in coverage["indeterminate"]}
    # These three disjoint sets must exhaust the plan. Missing ledger rows cannot
    # silently become incorrect responses or disappear from the planned denominator.
    if set(records)|unstarted|indeterminate != expected or set(records)&(unstarted|indeterminate) or unstarted&indeterminate:
        raise ValueError("Coverage and terminal ledger disagree")
    if coverage["expected"] != len(expected) or coverage["terminal"] != len(records) or coverage["complete"] != (not unstarted and not indeterminate):
        raise ValueError("Coverage counts/completion disagree with attempt identities")
    runs = manifest["runs"]
    primary_runs = [r for r in runs if r["role"] == "primary"]
    if len({r["config_id"] for r in primary_runs}) != len(primary_runs):
        raise ValueError("One primary response per configuration required")
    if any(r["replicate"] != 1 or r["experiment_id"] not in ("E0", "E1") for r in primary_runs):
        raise ValueError("Primary analysis requires E0/E1 replicate1")
    if len({r["experiment_id"] for r in primary_runs}) > 1:
        raise ValueError("Do not mix development E0 and E1 primary experiments")
    if any(r["role"] == "secondary" and (r["experiment_id"] != "E2" or r["config_id"] not in ({f"D{i:02d}" for i in range(1, 14)} if expanded else ({"J01", "D01"} if "D01" in configs else {"J01"}))) for r in runs):
        raise ValueError("Secondary preprint runs require E2 decision-service repeats")
    items = [q["item_id"] for q in manifest["items"]]
    matrix, valid_matrix, summaries, primary_rows = {}, {}, {}, {}
    all_complete = len(primary_runs) == len(configs)
    for run in primary_runs:
        ids = [(manifest["benchmark"], manifest["split"], item, run["config_id"], run["run_id"]) for item in items]
        rows = [records[i] for i in ids if i in records]
        primary_rows[run["config_id"]] = rows
        complete = len(rows) == len(ids)
        all_complete &= complete
        correct = sum(row["status"] == "valid" and row["selected_label"] == keys[attempt_identity(row)[:3]]["reference_answer"] for row in rows)
        n_valid = sum(row["status"] == "valid" for row in rows)
        # Completed failures count as incorrect on original keys. Incomplete runs
        # expose observed counts, but withhold full-run accuracy and intervals.
        summaries[run["config_id"]] = {"run_id": run["run_id"], "complete": complete, "planned_n": len(ids), "terminal_n": len(rows),
                                      "correct_observed": correct, "correct": correct if complete else None,
                                      "accuracy": fraction(correct, len(ids)) if complete else None,
                                      "wilson95": wilson(correct, len(ids)) if complete else None,
                                      "valid_n": n_valid, "valid_output_rate": fraction(n_valid, len(ids)) if complete else None,
                                      "unstarted_n": len(set(ids)&unstarted), "indeterminate_n": len(set(ids)&indeterminate),
                                      "operations": run_operations(rows, correct, complete_ledger=complete,
                                                               indeterminate_n=len(set(ids)&indeterminate), unstarted_n=len(set(ids)&unstarted), selected_final=expanded)}
        if complete:
            valid_matrix[run["config_id"]] = np.array([records[i]["status"] == "valid" for i in ids])
            matrix[run["config_id"]] = np.array([int(records[i]["status"] == "valid" and records[i]["selected_label"] == keys[i[:3]]["reference_answer"]) for i in ids])
    out = {"schema_version": SCHEMA_VERSION, "study_id": manifest["study_id"], "provenance": manifest["provenance"],
           "bindings": manifest["bindings"], "coverage": coverage, "primary_complete": all_complete,
           "study_complete": coverage["complete"], "primary": summaries,
           "method": {"repeatability_scope": "probability-only-20261012", "seed": seed, "resamples": 10000, "primary_key_rule": "all original reference keys retained",
                      "normalization": policy or "unaccepted", "method_review_sha256": methods.get("input_sha256") if methods else None,
                      "question_unit_assumption": "question-level inference; two known shared-context pairs disclosed in SAP"}}
    if expanded:
        from analysis.methods.expanded import impute_resources
        out['method']['resource_imputation'] = impute_resources(summaries, primary_rows)
        out['reporting_status'] = 'complete expanded benchmark' if coverage['complete'] else 'INCOMPLETE expanded benchmark'
        out['configuration_namespace'] = manifest['configuration_namespace']
        out['missing_configurations'] = manifest.get('missing_configurations', [])
    cost_policy = methods.get("cost_imputation") if methods else None
    if cost_policy is not None:
        out["method"]["cost_imputation"] = apply_primary_cost_imputation(summaries, primary_rows, cost_policy)
    sonnet_policy = (methods or {}).get("sonnet_cost_imputation")
    if sonnet_policy is not None:
        out["method"]["sonnet_cost_imputation"] = apply_sonnet_cost_imputation(summaries, primary_rows, sonnet_policy)
    duration_policy = (methods or {}).get("duration_imputation")
    if duration_policy is not None:
        out["method"]["duration_imputation"] = apply_kimi_duration_imputation(summaries, primary_rows, duration_policy)
    core = "D01" in configs and not expanded
    reviewed_family = not core or not methods or methods.get("primary_family", "decision_vs_generative_31") == "decision_vs_generative_31"
    if core:
        from analysis.methods.core import core_paired
        paired = core_paired
        out["source_packages"] = manifest["source_packages"]
        out["method"]["primary_family"] = "decision_vs_generative_31" if reviewed_family else None
    else:
        paired = paired_family
    if expanded:
        from analysis.methods.expanded import paired
        reviewed_family = (methods or {}).get('primary_family') == 'decision_related_273'
        out['method']['primary_family'] = 'decision_related_273'
        out['source_packages'] = manifest['source_packages']
    if all_complete and full_roster and reviewed_family:
        out["paired"] = paired(matrix, configs, pairs, seed=seed)
        out["complete_valid_sensitivity"] = paired(matrix, configs, pairs, seed=seed, valid_matrix=valid_matrix, inference=False)
        for family in out["complete_valid_sensitivity"]["families"].values():
            for row in family:
                mask = valid_matrix[row["positive_configuration"]]&valid_matrix[row["negative_configuration"]]
                row["excluded_ids"] = [item for item, eligible in zip(items, mask) if not eligible]
        for row in out["paired"]["families"]["S2"]:
            upper, lower = summaries[row["positive_configuration"]], summaries[row["negative_configuration"]]
            for basis in (("analysis_usd", "observed_usd", "estimated_usd") if expanded else ("observed_usd", "estimated_usd")):
                a, b = upper["operations"][basis]["mean_per_attempt"], lower["operations"][basis]["mean_per_attempt"]
                row[basis+"_mean_change"] = a-b if a is not None and b is not None else None
            a, b = (v["operations"].get("analysis_latency", v["operations"]["latency"]) for v in (upper, lower))
            row["median_latency_change_seconds"] = a["median"]-b["median"] if a["complete"] and b["complete"] else None
            row["failure_diagnostics"] = {cid: {"original": summaries[cid]["operations"]["original_failures"],
                                               "final": summaries[cid]["operations"]["final_failures"]}
                                          for cid in (row["positive_configuration"], row["negative_configuration"])}
        if exclusions is not None:
            # One common item mask applies to every configuration and contrast.
            # Original-key primary results above are retained unchanged.
            mask = np.array([item not in exclusions for item in items])
            out["reference_exclusion_sensitivity"] = {
                "common_n": int(mask.sum()), "excluded_ids": sorted(exclusions), "review_sha256": exclusions_review.get("input_sha256"),
                "outcome_triggered_nomination": exclusions_review["outcome_triggered_nomination"],
                "post_outcome_amendment": exclusions_review["post_outcome_amendment"],
                "audit_scope": "reviewed nominated candidates; empty exclusions do not certify a full clinical audit",
                "accuracy": {k: {"correct": int(v[mask].sum()), "n": int(mask.sum()), "accuracy": fraction(int(v[mask].sum()), int(mask.sum()))} for k, v in matrix.items()},
                "paired": paired({k: v[mask] for k, v in matrix.items()}, configs, pairs, seed=seed, inference=False)}
    else:
        out["paired"] = {"gate": (("SAP 1.8 requires primary_family='decision_related_273'" if expanded else "Current SAP requires primary_family='decision_vs_generative_31'") if not reviewed_family
                                   else "Complete primary coverage and exact accepted roster required; missing attempts are not failures")}
    if "reference_exclusion_sensitivity" not in out and (not core or exclusions_review is not None):
        out["reference_exclusion_sensitivity"] = {"gate": "No adjudicated reference-defect sensitivity supplied; primary original keys retained" if exclusions is None else "Complete primary coverage required"}
    out["frontiers"] = primary_frontiers(summaries)
    out["collection_conditions"] = coverage.get("collection_conditions", {})
    first_attempt = {}
    repeat_operations = {}
    for run in runs:
        ids = [(manifest["benchmark"], manifest["split"], item, run["config_id"], run["run_id"]) for item in items]
        rows = [records[i] for i in ids if i in records]
        original = [flat_record(r["original_attempt"]) if "original_attempt" in r else r for r in rows]
        correct = sum(r["status"] == "valid" and r["selected_label"] == keys[attempt_identity(r)[:3]]["reference_answer"] for r in original)
        complete = len(rows) == len(ids)
        key = run["config_id"] if run["role"] == "primary" else f"{run['config_id']}:run{run['replicate']}"
        first_attempt[key] = {"correct": correct if complete else None, "n": len(ids),
                              "accuracy": fraction(correct, len(ids)) if complete else None,
                              "wilson95": wilson(correct, len(ids)) if complete else None,
                              "original_failures": failure_summary(original),
                              "change_from_primary_pp": 100*(correct-summaries[run["config_id"]]["correct"])/len(ids)
                              if complete and run["role"] == "primary" and summaries[run["config_id"]]["correct"] is not None else None}
        if run["role"] == "secondary":
            selected_correct = sum(r["status"] == "valid" and r["selected_label"] == keys[attempt_identity(r)[:3]]["reference_answer"] for r in rows)
            repeat_operations[key] = run_operations(rows, selected_correct, complete_ledger=complete,
                unstarted_n=len(set(ids)&unstarted), indeterminate_n=len(set(ids)&indeterminate), selected_final=expanded)
    out["first_attempt_sensitivity"] = first_attempt
    out["repeat_operations"] = repeat_operations
    out["missing_visual_sensitivity"] = {"gate": "Select a versioned visual-exclusion policy"}
    visual_policy = (methods or {}).get("visual_exclusion_policy")
    visual_metadata = None
    if visual_policy == "fixed_nominated_108":
        from analysis.dataset.visual_review import fixed_visual_exclusions
        if visual_review is not None:
            raise ValueError("SAP v1.1 uses fixed nominations; do not supply an adjudicated review CSV")
        visual_metadata = fixed_visual_exclusions(manifest)
        excluded = set(visual_metadata["excluded_ids"])
        out["method"]["visual_exclusion_policy"] = visual_policy
    elif visual_policy is not None:
        raise ValueError("Unknown visual-exclusion policy")
    elif visual_review is not None:
        # Retained for historical/reviewed full-paper consumers, outside SAP v1.1.
        from analysis.dataset.visual_review import validate_review_rows
        if (visual_review.get("study_id") != manifest["study_id"]
                or visual_review.get("questions_sha256") != manifest["bindings"]["questions"]
                or not visual_review.get("input_sha256")):
            raise ValueError("Visual review must bind the current study/questions and approved CSV")
        validated = validate_review_rows(visual_review["rows"])
        excluded = set(validated["excluded_ids"])
        out["visual_review"] = {**validated, "candidate_n": len(visual_review["rows"]), "input_sha256": visual_review["input_sha256"]}
        visual_metadata = {"policy": "adjudicated_review", "review_sha256": visual_review["input_sha256"]}
    if visual_metadata is not None:
        if all_complete and full_roster and reviewed_family:
            mask = np.array([item not in excluded for item in items], dtype=bool)
            accuracy = {}
            for run in runs:
                ids = [(manifest["benchmark"], manifest["split"], item, run["config_id"], run["run_id"])
                       for item in items if item not in excluded]
                complete = all(i in records for i in ids)
                n = len(ids)
                correct = sum(i in records and records[i]["status"] == "valid" and records[i]["selected_label"] == keys[i[:3]]["reference_answer"] for i in ids)
                key = run["config_id"] if run["role"] == "primary" else f"{run['config_id']}:run{run['replicate']}"
                accuracy[key] = {"n": n, "correct": correct if complete else None,
                                 "accuracy": fraction(correct, n) if complete else None,
                                 "wilson95": wilson(correct, n) if complete else None}
            if visual_policy == "fixed_nominated_108" and int(mask.sum()) != visual_metadata["retained_n"]:
                raise ValueError("Fixed visual sensitivity has an unexpected retained denominator")
            out["missing_visual_sensitivity"] = {"common_n": int(mask.sum()), "excluded_ids": sorted(excluded),
                **visual_metadata, "accuracy": accuracy,
                "paired": paired({k: v[mask] for k, v in matrix.items()}, configs, pairs, seed=seed, inference=False)}
    if expanded:
        from analysis.methods.expanded import secondary
        out['decision_models'] = secondary(manifest, runs, records, keys, coverage, methods or {}, seed)
        return out
    jev_primary = next((r for r in primary_runs if r["config_id"] == "J01"), None)
    if jev_primary:
        out["confidence"] = confidence_for_run(
            manifest, jev_primary, records, keys, policy, methods={**(methods or {}), "bootstrap_seed": seed},
            complete=summaries["J01"]["complete"])
    out["stability"] = stability(manifest, runs, records, keys, coverage, policy)
    if core:
        from analysis.methods.decisions import probability_tolerance, secondary_decisions
        d01_source = next(m for m in manifest["source_manifests"] if m["execution_workflow"] == "lean-decisions-v1")
        d01_methods = methods.get("decision_models", {}).get("D01", {}) if methods else {}
        saved_tolerance = d01_source.get("probability_tolerance")
        tolerance = d01_methods.get("sum_tolerance", saved_tolerance)
        if probability_tolerance(tolerance) != probability_tolerance(saved_tolerance):
            raise ValueError("D01 analysis tolerance must match the frozen collection tolerance")
        d01 = secondary_decisions(manifest, runs, records, keys, coverage,
                                  sum_tolerance=tolerance,
                                  bootstrap_seed=seed,
                                  probability_semantics_verified=d01_methods.get("probability_semantics_verified") is True)
        out["decision_models"] = {
            "J01": {"confidence": out["confidence"], "stability": out["stability"]}, "D01": d01}
        out["method"]["decision_models"] = {"D01": {"sum_tolerance": d01["sum_tolerance"],
            "collection_sum_tolerance": saved_tolerance,
            "probability_semantics_verified": d01_methods.get("probability_semantics_verified") is True}}
    return out


def primary_frontiers(summaries):
    """Compute cost bases separately using only supplied primary summaries."""
    frontiers = {}
    # Retain observed-only and token-estimated bases beside the labelled amendment.
    bases = ["observed_usd", "estimated_usd"]
    if any("analysis_usd" in v["operations"] for v in summaries.values()):
        if not all("analysis_usd" in v["operations"] for v in summaries.values()):
            raise ValueError("Amended cost frontier requires one common basis across the roster")
        bases.append("analysis_usd")
    for basis in bases:
        latency_basis = "analysis_latency" if basis == "analysis_usd" else "latency"
        points = {k: {"accuracy": v["accuracy"], "cost": v["operations"][basis]["mean_per_attempt"] if v["complete"] else None,
                      "latency": v["operations"].get(latency_basis, v["operations"]["latency"])["median"] if v["complete"] and v["operations"].get(latency_basis, v["operations"]["latency"])["complete"] else None} for k, v in summaries.items()}
        frontiers[basis] = {"accuracy_cost": pareto(points, {"accuracy": True, "cost": False}),
                            "three_dimensional": pareto(points, {"accuracy": True, "cost": False, "latency": False}),
                            "basis": basis, "currency": "USD"}
        if "accuracy_latency" not in frontiers:
            frontiers["accuracy_latency"] = pareto(points, {"accuracy": True, "latency": False})
    if any("analysis_latency" in v["operations"] for v in summaries.values()):
        if not all("analysis_latency" in v["operations"] for v in summaries.values()):
            raise ValueError("Amended duration frontier requires one common basis across the roster")
        frontiers["observed_accuracy_latency"] = frontiers["accuracy_latency"]
        points = {k: {"accuracy": v["accuracy"], "latency": v["operations"]["analysis_latency"]["median"]
                     if v["complete"] and v["operations"]["analysis_latency"]["complete"] else None}
                  for k, v in summaries.items()}
        frontiers["accuracy_latency"] = pareto(points, {"accuracy": True, "latency": False})
    return frontiers


def confidence_for_run(manifest, run, records, keys, policy, *, methods=None,
                       complete=False, vector_reader=vector_for_metrics):
    """Use one run's returned-choice probability, keeping calibration gated."""
    scores, correct, excluded, vector_diagnostics = [], [], Counter(), []
    for item in manifest["items"]:
        ident = (manifest["benchmark"], manifest["split"], item["item_id"],
                 run["config_id"], run["run_id"])
        row = records.get(ident)
        if row is None:
            excluded["missing_attempt"] += 1
            continue
        native = row["probability"].get("native_vector")
        try:
            total = sum((Fraction(str(v)) for v in native.values()), Fraction(0)) if isinstance(native, dict) and native else None
        except (ValueError, ZeroDivisionError):
            total = None
        vector, reason = vector_reader(row, policy, manifest["provenance"])
        vector_diagnostics.append({"item_id": item["item_id"], "native_vector": native,
                                   "sum_exact": str(total) if total is not None else None,
                                   "deviation_exact": str(abs(total-1)) if total is not None else None,
                                   "eligible": vector is not None, "exclusion_reason": reason})
        if vector is None:
            excluded[reason] += 1
            continue
        score = (Fraction(str(row["probability"]["native_vector"][row["selected_label"]]))
                 if policy == "native_selected_normalized_distribution" else vector[row["selected_label"]])
        scores.append(score)
        correct.append(row["selected_label"] == keys[ident[:3]]["reference_answer"])
    return {**confidence(scores, correct, calibration=bool(methods and methods.get("probability_semantics_verified") is True),
                         seed=methods.get("bootstrap_seed") if methods else None),
            "vector_diagnostics": vector_diagnostics,
            "planned_n": len(manifest["items"]), "exclusions": dict(excluded),
            "primary_run_id": run["run_id"],
            "score_policy": ("native_selected_probability" if policy == "native_selected_normalized_distribution"
                             else policy or "unit_sum_fictional_only"),
            "complete_primary_run": complete}


def load_study(package, questions_path, keys_path):
    # Read only explicit packages, never discover historical/private runs. Join
    # original keys here, after inference, using benchmark/split/item identity.
    packages = [Path(package)] if isinstance(package, (str, Path)) else [Path(p) for p in package]
    if not packages:
        raise ValueError("At least one explicit package required")
    if len({path.resolve() for path in packages}) != len(packages):
        raise ValueError("duplicate saved package path")
    workflows = [strict_json((path / 'manifest.json').read_text()).get('execution_workflow')
                 for path in packages]
    if any(workflow in {'lean-v1', 'lean-decisions-v1'} for workflow in workflows):
        from analysis.methods.core import join_studies, reject_repeated_sources
        reject_repeated_sources(packages)
    # Historical reproduction stays isolated from the benchmarking namespace.
    # Lean packages validate each execution segment through their pure reader.
    def load_explicit_package(path, workflow):
        if workflow in {'lean-v1', 'lean-decisions-v1'}:
            from benchmarking.common import read_package as read_execution_package
            return read_execution_package(path)
        return read_package(path)

    loaded = [load_explicit_package(path, workflow) for path, workflow in zip(packages, workflows)]
    manifest, terminal, coverage = loaded[0]
    core_join = any(m.get("execution_workflow") == 'lean-decisions-v1' for m, _, _ in loaded)
    if core_join:
        manifest, terminal, coverage = join_studies(loaded, packages)
        validate_manifest(manifest)
    elif len(loaded) > 1:
        # Independently verified immutable ledgers; do not manufacture a merged raw package.
        match_fields = ("schema_version", "execution_workflow", "study_id", "provenance", "benchmark", "split", "bindings", "items", "configurations", "key_manifest", "schedule", "roster_decision")
        for other, _, _ in loaded[1:]:
            if any(other.get(field) != manifest.get(field) for field in match_fields):
                raise ValueError("Saved packages have incompatible study/input/config/method bindings")
        manifest = {**manifest, "runs": [run for m, _, _ in loaded for run in m["runs"]]}
        # The virtual combined plan has the same canonical order as a single E1+E2 plan.
        config_order = {c["config_id"]: i for i, c in enumerate(manifest["configurations"])}
        manifest["runs"].sort(key=lambda r: ({"E0": 0, "E1": 1, "E2": 2}[r["experiment_id"]],
                                            config_order[r["config_id"]], r["replicate"]))
        validate_manifest(manifest)
        terminal = [row for _, rows, _ in loaded for row in rows]
        coverage = {"expected": sum(c["expected"] for _, _, c in loaded), "terminal": len(terminal),
                    "unstarted": [i for _, _, c in loaded for i in c["unstarted"]],
                    "indeterminate": [i for _, _, c in loaded for i in c["indeterminate"]],
                    "complete": all(c["complete"] for _, _, c in loaded), "last_event_sha256": None,
                    "source_chain_heads": [c["last_event_sha256"] for _, _, c in loaded]}
    if file_hash(questions_path) != manifest["bindings"]["questions"]:
        raise ValueError("Question file binding mismatch")
    questions = load_questions(questions_path)
    # Manifest items keep preparation order. The separate schedule binds execution order.
    planned = {x["item_id"]: (x["content_sha256"], tuple(x["labels"])) for x in manifest["items"]}
    actual = {q.item_id: (question_hash(q), q.labels) for q in questions}
    if [q['item_id'] for q in manifest['items']]!=[q.item_id for q in questions]:
        raise ValueError('Manifest preparation order mismatch')
    if planned != actual or any(q.benchmark != manifest["benchmark"] or q.split != manifest["split"] for q in questions):
        raise ValueError("Question identity/content/option order mismatch")
    key_manifest = manifest.get("key_manifest")
    if not key_manifest or key_manifest.get("sha256") != file_hash(keys_path):
        raise ValueError("Isolated original keys require matching key_manifest.sha256")
    keys = load_keys(keys_path, questions)
    return manifest, terminal, coverage, keys


def markdown_tables(summary):
    lines = ["Fictional offline results" if summary["provenance"] == "mock" else "Offline saved-attempt analysis",
             "", f"Primary coverage complete: {summary['primary_complete']}. Entire planned study complete: {summary['study_complete']}.",
             "", "| Configuration | Correct / planned | Accuracy | Wilson 95% | Valid / planned | Billed USD | Median wall seconds |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    cost_basis = "analysis_usd" if "analysis_usd" in summary["frontiers"] else "observed_usd"
    if cost_basis == "analysis_usd":
        lines[4] = lines[4].replace("Billed USD", "Analysis USD with approved mean imputation")
        lines[3] = "\nCost basis: observed final-attempt billing plus approved primary within-run mean imputations; observed receipts and missing counts are retained.\n"
    if summary.get("method", {}).get("duration_imputation"):
        lines[3] += "Duration basis: observed all-call durations plus one approved C23 within-run mean assignment; raw missingness is retained.\n"
    fmt = lambda v: f"{v:.4f}" if v is not None else "unavailable"
    for name, row in summary["primary"].items():
        ci = row["wilson95"]
        lines.append(f"| {name} | {row['correct'] if row['complete'] else 'incomplete'}/{row['planned_n']} | {fmt(row['accuracy'])} | {', '.join(fmt(v) for v in ci) if ci else 'unavailable'} | {row['valid_n']}/{row['planned_n']} | {fmt(row['operations'][cost_basis]['total'])} | {fmt(row['operations'].get('analysis_latency', row['operations']['latency'])['median'])} |")
    lines += ["", "| Configuration | Terminal / started | Analysis USD per question | Observed billing missing n | Estimated USD per attempt | Wall Q1 / Q3 / p95 seconds | Analysis 3D nondominated |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    frontier = summary["frontiers"][cost_basis]["three_dimensional"]
    for name, row in summary["primary"].items():
        op = row["operations"]; lat = op.get("analysis_latency", op["latency"])
        nondominated = "excluded: "+", ".join(frontier["excluded"][name]["missing_coordinates"]) if name in frontier["excluded"] else ("yes" if name in frontier["nondominated"] else "no")
        lines.append(f"| {name} | {op['terminal_n']} / {op['attempted_n']} | {fmt(op[cost_basis]['mean_per_attempt'])} | {op['observed_usd']['missing_n']} | {fmt(op['estimated_usd']['mean_per_attempt'])} | {' / '.join(fmt(lat[k]) for k in ('q1', 'q3', 'p95'))} | {nondominated} |")
    if "families" in summary["paired"]:
        lines += ["", "| Family / contrast | Positive minus negative | Difference pp | Bootstrap 95% pp | Newcombe 95% pp | Both correct / + / − / both incorrect | Exact P | Holm P |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for family, rows in summary["paired"]["families"].items():
            for row in rows:
                lines.append(f"| {family}/{row['contrast_id']} | {row['positive_configuration']} ({row['positive_effort']}) − {row['negative_configuration']} ({row['negative_effort']}) | {fmt(row['difference_pp'])} | {', '.join(fmt(v) for v in row['interval95_pp']) if row['interval95_pp'] else 'undefined'} | {', '.join(fmt(v) for v in row['newcombe95_pp']) if row['newcombe95_pp'] else 'undefined'} | {row['both_correct']} / {row['positive_discordant']} / {row['negative_discordant']} / {row['both_incorrect']} | {fmt(row['p_exact'])} | {fmt(row['p_holm'])} |")
    conf = summary.get("confidence")
    if conf:
        high = conf["high_confidence"]
        lines += ["", f"JEV run-1 confidence eligible n: {conf['eligible_n']} / {conf['planned_n']}; score policy: {conf['score_policy']}.",
                  f"Correctness AUROC: {fmt(conf['auroc'])}"+(" ("+conf['auroc_reason']+")" if conf['auroc_reason'] else "")+".",
                  f"High-confidence errors at ≥0.90: {high['errors']} / {high['n']}; rate {fmt(high['error_rate'])}."]
        if conf["calibration_gate"]:
            lines.append("Calibration gate: "+conf["calibration_gate"]+".")
        else:
            lines += ["", "| Probability bin | n | Mean selected probability | Observed accuracy |", "| --- | --- | --- | --- |"]
            for b in conf["reliability"]:
                lines.append(f"| [{b['lower']:.1f}, {b['upper']:.1f}{']' if b['upper_inclusive'] else ')'} | {b['n']} | {fmt(b['mean_probability'])} | {fmt(b['observed_accuracy'])} |")
        if conf["exclusions"]:
            lines.append("Confidence exclusions: "+", ".join(f"{reason}={n}" for reason, n in conf["exclusions"].items())+".")
    stability = summary["stability"]
    if "runs" in stability:
        lines += ["", "| JEV run | Complete | Correct / planned | Accuracy | Valid output rate | Service dates |", "| --- | --- | --- | --- | --- | --- |"]
        for rep, row in stability["runs"].items():
            lines.append(f"| {rep} | {row['complete']} | {row['correct_observed']} / {row['planned_n']} | {fmt(row['accuracy'])} | {fmt(row['valid_output_rate'])} | {', '.join(row['service_dates'])} |")
        lines += ["", "| JEV run pair | Agreements / valid pairs | Agreement | Eligible vectors | Mean / median TV | Mean / median earlier-choice probability change |", "| --- | --- | --- | --- | --- | --- |"]
        for row in stability["pairs"]:
            tv, change = row["total_variation"], row["earlier_selected_probability_change"]
            lines.append(f"| {row['earlier_run']}–{row['later_run']} | {row['agreements']} / {row['valid_pair_n']} | {fmt(row['answer_agreement'])} | {row['vector_eligible_n']} | {fmt(tv['mean'])} / {fmt(tv['median'])} | {fmt(change['mean'])} / {fmt(change['median'])} |")
        triple = stability["three_way"]
        lines += ["", f"Three-way agreement: {triple['agreements']} / {triple['valid_triple_n']} valid triples; rate {fmt(triple['rate'])}. Stable wrong answers remain accuracy errors."]
    if "gate" in summary["paired"]:
        lines += ["", "Paired inference gate: " + summary["paired"]["gate"] + "."]
    d01 = summary.get("decision_models", {}).get("D01")
    if d01:
        conf = d01["confidence"]
        lines += ["", f"D01 run-1 confidence eligible n: {conf['eligible_n']} / {conf['planned_n']}; native-score policy: {conf['score_policy']}.",
                  f"D01 correctness AUROC: {fmt(conf['auroc'])}; high-confidence errors: {conf['high_confidence']['errors']} / {conf['high_confidence']['n']}.",
                  "D01 sum tolerance: " + (d01["sum_tolerance"] or "unreviewed") + "."]
        if conf["calibration_gate"]:
            lines.append("D01 calibration gate: " + conf["calibration_gate"] + ".")
        if conf["exclusions"]:
            lines.append("D01 confidence exclusions: " + ", ".join(f"{k}={v}" for k, v in conf["exclusions"].items()) + ".")
        if "runs" in d01["stability"]:
            lines += ["", "| D01 run | Complete | Correct / planned | Accuracy | Valid output rate |", "| --- | --- | --- | --- | --- |"]
            for rep, row in d01["stability"]["runs"].items():
                lines.append(f"| {rep} | {row['complete']} | {row['correct_observed']} / {row['planned_n']} | {fmt(row['accuracy'])} | {fmt(row['valid_output_rate'])} |")
    for cid, diagnostics in summary.get("decision_models", {}).items():
        conf = diagnostics["confidence"]
        auc_ci = conf["auroc_bootstrap"]["interval95"]
        lines += ["", f"{cid} AUROC 95%: {', '.join(fmt(v) for v in auc_ci) if auc_ci else 'unavailable'}; binary Brier: {fmt(conf['binary_brier'])}.",
                  f"{cid} score support: {conf['score_support']['distinct_n']} distinct; {conf['score_support']['zero_n']} at 0, {conf['score_support']['one_n']} at 1."]
    if "visual_review" in summary:
        irr = summary["visual_review"]["irr"]
        lines += ["", f"OF/DG pre-adjudication IRR: paired n={irr['paired_n']}; agreement={fmt(irr['percentage_agreement'])}%; Cohen κ={fmt(irr['kappa'])}; both uncertain n={irr['both_uncertain_n']}."]
    lines += ["", "Partial billing is an observed subtotal in JSON, never a complete billed total. Request-wall latency includes routing/scheduling.",
              "Probability/calibration policy and optional adjudication status remain explicit in summary.json. Question-level inference assumptions are disclosed in the SAP.",
              "Holm controls each named family separately. Nonsignificance does not establish equivalence."]
    return "\n".join(lines)+"\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True, action="append", help="Saved directory; repeat for compatible E1 and E2 ledgers")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--keys", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, help="Recorded fixed nonnegative bootstrap seed")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing results never overwritten")
    for name in ("methods", "exclusions"):
        parser.add_argument("--"+name, type=Path)
        parser.add_argument("--"+name+"-sha256")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--config", type=Path, help="Preserved original config TOML; required for definitive reproduction")
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--freeze-sha256")
    parser.add_argument("--freeze-review", type=Path)
    parser.add_argument("--freeze-review-sha256")
    parser.add_argument("--analysis-review", type=Path, help="Later human analysis release for schema-3 execution freezes")
    parser.add_argument("--analysis-review-sha256")
    parser.add_argument("--artifact-map", type=Path, help="Optional schema-1 checksum to local path JSON map")
    args = parser.parse_args(argv)
    try:
        if args.seed < 0:
            raise ValueError("Seed must be nonnegative")
        manifest, terminal, coverage, keys = load_study(args.package, args.questions, args.keys)
        from analysis.support.reproduction import ArtifactResolver
        resolver = ArtifactResolver(args.artifact_map)
        reproduction = {}
        frozen_methods = None
        if manifest["provenance"] == "definitive":
            # Offline provenance validation never uses live resume/registry state.
            from analysis.support.readiness import verify_definitive_analysis
            frozen_methods = verify_definitive_analysis(
                manifest, seed=args.seed, config=args.config, freeze_path=args.freeze_manifest,
                freeze_sha256=args.freeze_sha256, review_path=args.freeze_review,
                review_sha256=args.freeze_review_sha256, resolver=resolver, reproduction_record=reproduction,
                analysis_review_path=args.analysis_review, analysis_review_sha256=args.analysis_review_sha256)
        methods = review_file(args.methods, args.methods_sha256, manifest, kind="analysis_methods", resolver=resolver)
        if frozen_methods is not None:
            if methods is not None and methods["input_sha256"] != frozen_methods["input_sha256"]:
                raise ValueError("Supplied methods differ from the frozen accepted method")
            methods = frozen_methods
        exclusions = review_file(args.exclusions, args.exclusions_sha256, manifest, kind="reference_exclusions", resolver=resolver)
        summary = analyze(manifest, terminal, coverage, keys, seed=args.seed, methods=methods, exclusions_review=exclusions)
        summary["inputs"] = {"packages": [{"manifest_sha256": file_hash(package/"manifest.json"), "events_sha256": file_hash(package/"events.jsonl")} for package in args.package],
                             "analysis_manifest_sha256": fingerprint(manifest),
                             "questions_sha256": file_hash(args.questions), "keys_sha256": file_hash(args.keys),
                             "analysis_code": analysis_code_record(),
                             "numpy_version": np.__version__}
        args.output.mkdir(parents=True, exist_ok=False)
        write_new(args.output/"summary.json", summary)
        if reproduction:
            write_new(args.output/"reproduction_environment.json", reproduction)
        (args.output/"tables.md").write_text(markdown_tables(summary))
        if args.figures:
            from analysis.support.figures import draw_figures
            draw_figures(summary, args.output)
        print(f"Offline analysis: {args.output}/summary.json; primary complete={summary['primary_complete']}; study complete={summary['study_complete']}")
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print("Analysis blocked: "+str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
