"""Offline native-vector confidence and stability for D01.

The notebook consumes independently verified schema-4 packages joined with the
original study. Probability precision and calibration semantics remain explicit.
Historical exploratory kernels remain available for preserved results only.
"""
from decimal import Decimal
from fractions import Fraction

import numpy as np

from common import BUDGET_ROSTERS, KIMI_INSTANT_DECISION
from analysis.methods.pipeline import (confidence_for_run, paired_contrasts,
                               primary_frontiers, stability)

D01 = "D01"
COMPARATORS = ("J01", "C05", "C06")


def probability_tolerance(value):
    """Require an explicit rational tolerance; never inherit JEV's 0.02."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("D01 sum tolerance must be a finite number in [0,1]")
    try:
        tolerance = Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        raise ValueError("D01 sum tolerance must be a finite number in [0,1]") from None
    if not 0 <= tolerance <= 1:
        raise ValueError("D01 sum tolerance must be a finite number in [0,1]")
    return tolerance


def native_vector(record, *, sum_tolerance, labels=("A", "B", "C", "D")):
    """Validate native precision independently of any JEV-derived valid flag."""
    tolerance = probability_tolerance(sum_tolerance)
    if record["status"] != "valid":
        return None, "invalid_final_label"
    if tolerance is None:
        return None, "d01_sum_tolerance_unreviewed"
    p = record["probability"].get("native_vector")
    if (len(labels) != 4 or len(set(labels)) != 4 or not isinstance(p, dict)
            or set(p) != set(labels)):
        return None, "label_coverage"
    try:
        if any(isinstance(v, bool) or not isinstance(v, (str, int, float, Decimal, Fraction)) for v in p.values()):
            return None, "nonfinite_or_range"
        values = {k: Fraction(str(v)) for k, v in p.items()}
        if any(not 0 <= v <= 1 for v in values.values()):
            return None, "nonfinite_or_range"
    except (ValueError, ZeroDivisionError):
        return None, "nonfinite_or_range"
    total = sum(values.values())
    if total <= 0:
        return None, "nonpositive_sum"
    if abs(total - 1) > tolerance:
        return None, "sum_outside_tolerance"
    if record["selected_label"] not in values:
        return None, "selected_label_unavailable"
    return values, None


def secondary_decisions(manifest, runs, records, keys, coverage, *, sum_tolerance,
                        probability_semantics_verified=False, bootstrap_seed=None):
    """Apply all JEV secondary kernels to verified, flattened D01 inputs.

    Records retain a label-to-value native_vector and a separate provider_confidence
    scalar. The reader supplies those fields after validating the source/artifact
    hashes, identities and coverage, without repairing raw values.
    """
    tolerance = probability_tolerance(sum_tolerance)
    selected_runs = [r for r in runs if r["config_id"] == D01]
    primary = [r for r in selected_runs if r["replicate"] == 1 and r["role"] == "primary"]
    if len(primary) != 1:
        raise ValueError("D01 confidence requires exactly one primary run 1")
    if any((r["replicate"] == 1 and r["role"] != "primary")
           or (r["replicate"] in (2, 3) and r["role"] != "secondary")
           or r["replicate"] not in (1, 2, 3) for r in selected_runs):
        raise ValueError("D01 runs 2 and 3 must be secondary")

    def reader(row, policy, provenance):
        values, reason = native_vector(row, sum_tolerance=tolerance)
        if values is None:
            return None, reason
        total = sum(values.values())
        return {k: v / total for k, v in values.items()}, None

    policy = "native_selected_normalized_distribution"
    complete = all((manifest["benchmark"], manifest["split"], item["item_id"],
                    D01, primary[0]["run_id"]) in records for item in manifest["items"])
    return {"configuration": D01,
            "sum_tolerance": str(tolerance) if tolerance is not None else None,
            "confidence": confidence_for_run(
                manifest, primary[0], records, keys, policy, complete=complete,
                methods={"probability_semantics_verified": probability_semantics_verified,
                         "bootstrap_seed": bootstrap_seed},
                vector_reader=reader),
            "stability": stability(manifest, selected_runs, records, keys, coverage, policy,
                                   config_id=D01,
                                   vector_reader=reader)}


def paired_decisions(matrix, configs, *, question_ids, seed, valid_matrix=None):
    """S-D=3 on run-1 original-key outcomes, with explicit ordered question IDs."""
    required = {D01, *COMPARATORS}
    if set(matrix) != required or set(configs) != required or set(question_ids) != required:
        raise ValueError("S-D requires exactly D01, J01, C05 and C06")
    ids = tuple(question_ids[D01])
    if len(set(ids)) != len(ids) or any(tuple(question_ids[k]) != ids for k in required):
        raise ValueError("S-D requires identical unique question identities and order")
    outcomes = {}
    for cid, values in matrix.items():
        values = np.asarray(values)
        if values.ndim != 1 or len(values) != len(ids) or not np.isin(values, [0, 1]).all():
            raise ValueError("S-D outcomes require complete aligned binary run-1 responses")
        outcomes[cid] = values.astype(int)
    if valid_matrix is not None:
        if set(valid_matrix) != required:
            raise ValueError("S-D sensitivity requires all four validity arrays")
        for values in valid_matrix.values():
            a = np.asarray(values)
            if a.ndim != 1 or len(a) != len(ids) or not np.isin(a, [0, 1]).all():
                raise ValueError("S-D validity must align with the original questions")
        valid_matrix = {k: np.asarray(v, dtype=bool) for k, v in valid_matrix.items()}
    result = paired_contrasts(outcomes, configs,
                             [("S-D", k, D01, k) for k in COMPARATORS], {"S-D": 3},
                             seed=seed, valid_matrix=valid_matrix)
    result["interpretation"] = "exploratory post-start extension; D01 minus comparator; run 1 only"
    return result


def decisions_frontiers(primary):
    """Return original and extended frontiers without overwriting either view."""
    original = set(BUDGET_ROSTERS[KIMI_INSTANT_DECISION])
    if set(primary) != original | {D01}:
        raise ValueError("Extended frontier requires the original 16 configurations plus D01")
    return {"original": primary_frontiers({k: v for k, v in primary.items() if k in original}),
            "extended": primary_frontiers(primary),
            "interpretation": "D01 run 1; later service dates and scheduling qualify latency comparisons"}
