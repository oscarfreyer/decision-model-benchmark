"""Pure statistical kernels for final SAP v1.0; no file/network access."""
from __future__ import annotations

from collections import Counter
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math

import numpy as np
from scipy.stats import binomtest, norm, rankdata


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def quantiles(values):
    values = list(values)
    if not values:
        return {"n": 0, "median": None, "q1": None, "q3": None, "iqr": None, "p95": None}
    q1, median, q3, p95 = np.quantile(values, [.25, .5, .75, .95], method="linear")
    return {"n": len(values), "median": float(median), "q1": float(q1),
            "q3": float(q3), "iqr": float(q3-q1), "p95": float(p95)}


def wilson(correct, n):
    if not isinstance(correct, int) or not isinstance(n, int) or not 0 <= correct <= n:
        raise ValueError("Wilson requires integer counts 0 <= correct <= n")
    if not n:
        return None
    z = float(norm.ppf(.975))
    p = correct / n
    scale = 1 + z*z/n
    centre = (p + z*z/(2*n))/scale
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/scale
    return [max(0., centre-half), min(1., centre+half)]


def exact_mcnemar(positive, negative):
    # Only discordant matched answers contribute. Under equal accuracy their
    # direction is binomial with p=0.5; no discordance gives P=1, not equivalence.
    if any(type(x) is not int or x < 0 for x in (positive, negative)):
        raise ValueError("Discordant counts must be nonnegative integers")
    n = positive + negative
    return float(binomtest(positive, n, .5, alternative="two-sided").pvalue) if n else 1.


def holm(pvalues, family_size):
    # Scientific family selection belongs to the roster-validated caller. This
    # kernel verifies the supplied family is complete and applies Holm's formula.
    if type(family_size) is not int or family_size < 1 or len(pvalues) != family_size:
        raise ValueError("Holm requires the complete declared family of P values")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues):
        raise ValueError("P values must be finite in [0,1]")
    order = sorted(range(family_size), key=lambda i: (pvalues[i], i))
    adjusted, running = [0.] * family_size, 0.
    for rank, i in enumerate(order):
        # Step-down factors decrease with rank; the cumulative maximum makes
        # adjusted P values monotone before restoring the original contrast order.
        running = max(running, (family_size-rank)*pvalues[i])
        adjusted[i] = min(1., running)
    return adjusted


def paired_counts(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 1 or b.shape != a.shape or not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError("Paired outcomes must be aligned one-dimensional binary arrays")
    positive, negative = int(((a == 1)&(b == 0)).sum()), int(((a == 0)&(b == 1)).sum())
    return {"n": len(a), "difference_pp": 100*fraction(positive-negative, len(a)) if len(a) else None,
            "both_correct": int(((a == 1)&(b == 1)).sum()),
            "both_incorrect": int(((a == 0)&(b == 0)).sum()),
            "positive_discordant": positive, "negative_discordant": negative,
            "p_exact": exact_mcnemar(positive, negative) if len(a) else None}


def newcombe_paired(cells):
    """Newcombe (1998) method 10, first minus second, percentage points.

    Positive covariance uses Newcombe's n/2 correction. Reference implementation
    and worked example: statsmodels.stats.proportion.confint_proportions_paired.
    https://doi.org/10.1002/(SICI)1097-0258(19981130)17:22<2635::AID-SIM954>3.0.CO;2-C
    """
    a, b, c, d = (cells[k] for k in
                  ("both_correct", "positive_discordant", "negative_discordant", "both_incorrect"))
    if any(type(v) is not int or v < 0 for v in (a, b, c, d)):
        raise ValueError("Paired cells must be nonnegative integers")
    n = a+b+c+d
    if not n:
        return None
    p, q = (a+b)/n, (a+c)/n
    lp, up = wilson(a+b, n)
    lq, uq = wilson(a+c, n)
    product = (a+b)*(c+d)*(a+c)*(b+d)
    covariance = a*d-b*c
    if covariance > 0:
        covariance = max(covariance-n/2, 0)
    phi = covariance/math.sqrt(product) if product else 0.
    lower = math.sqrt(max(0., (p-lp)**2+(uq-q)**2-2*phi*(p-lp)*(uq-q)))
    upper = math.sqrt(max(0., (up-p)**2+(q-lq)**2-2*phi*(up-p)*(q-lq)))
    return [100*max(-1., p-q-lower), 100*min(1., p-q+upper)]


def bootstrap_differences(differences, *, seed, resamples=10000, eligibility=None):
    """Rows are ordered questions, columns contrasts; shared sampled IDs for every column."""
    d = np.asarray(differences)
    if d.ndim != 2 or not np.isin(d, [-1, 0, 1]).all():
        raise ValueError("Bootstrap requires a question-by-contrast {-1,0,1} matrix")
    if type(seed) is not int or seed < 0 or resamples != 10000:
        raise ValueError("Record a nonnegative integer seed and use exactly 10,000 resamples")
    n, contrasts = d.shape
    mask = np.ones_like(d, dtype=bool) if eligibility is None else np.asarray(eligibility, dtype=bool)
    if mask.shape != d.shape:
        raise ValueError("Sensitivity eligibility must align with questions/contrasts")
    if not n:
        return {"intervals_pp": [None]*contrasts, "degenerate": [True]*contrasts,
                "constant_distribution": [None]*contrasts,
                "undefined_resamples": [resamples]*contrasts,
                "reason": "no paired questions", "seed": seed, "resamples": resamples,
                "rng": "numpy.PCG64", "percentile_method": "linear"}
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty((resamples, contrasts))
    digest = hashlib.sha256()
    for start in range(0, resamples, 256):
        # Batch draws bound memory; each resample uses the same question indices
        # across all contrasts, retaining pairing and their shared question sample.
        stop = min(resamples, start+256)
        ids = rng.integers(0, n, size=(stop-start, n), dtype=np.int32)
        digest.update(ids.astype("<i4").tobytes())
        # Reuse these exact draws in bounded contrast blocks for the 273-pair family.
        for column in range(0, contrasts, 16):
            last = min(contrasts, column+16)
            sampled_mask = mask[:, column:last][ids]
            denominator = sampled_mask.sum(axis=1)
            numerator = (d[:, column:last][ids]*sampled_mask).sum(axis=1)
            draws[start:stop, column:last] = np.divide(100*numerator, denominator,
                out=np.full_like(numerator, np.nan, dtype=float), where=denominator > 0)
    undefined = np.isnan(draws).sum(axis=0)
    intervals, degenerate, constant = [], [], []
    for i in range(contrasts):
        if undefined[i]:
            # A sensitivity draw with zero eligible pairs is undefined. Do not
            # discard such draws and silently change the percentile procedure.
            intervals.append(None); degenerate.append(True); constant.append(None)
        else:
            lo, hi = np.quantile(draws[:, i], [.025, .975], method="linear")
            intervals.append([float(lo), float(hi)]); degenerate.append(lo == hi)
            constant.append(bool(np.all(draws[:, i] == draws[0, i])))
    return {"intervals_pp": intervals, "degenerate": degenerate, "constant_distribution": constant,
            "undefined_resamples": undefined.tolist(),
            "seed": seed, "resamples": resamples, "rng": "numpy.PCG64",
            "percentile_method": "linear", "sampled_question_indices_sha256": digest.hexdigest()}


def auroc_interval(scores, correct, *, seed):
    """Stratified percentile bootstrap, preserving exact score ties and class n."""
    positives = [s for s, c in zip(scores, correct) if c]
    negatives = [s for s, c in zip(scores, correct) if not c]
    result = {"seed": seed, "resamples": 10000, "rng": "numpy.PCG64",
              "percentile_method": "linear", "correct_n": len(positives),
              "incorrect_n": len(negatives), "interval95": None, "degenerate": None,
              "constant_distribution": None}
    if not positives or not negatives:
        return {**result, "reason": "single correctness class" if scores else "no eligible scores"}
    if type(seed) is not int or seed < 0:
        return {**result, "reason": "bootstrap seed required"}
    # Integer order codes retain exact rational ties, including values whose
    # binary floating-point representations would otherwise collapse together.
    codes = {s: i for i, s in enumerate(sorted(set(scores)))}
    pos, neg = np.array([codes[s] for s in positives]), np.array([codes[s] for s in negatives])
    np_, nn = len(pos), len(neg)
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty(10000)
    for start in range(0, 10000, 128):
        size = min(128, 10000-start)
        sampled = np.concatenate((pos[rng.integers(np_, size=(size, np_))],
                                  neg[rng.integers(nn, size=(size, nn))]), axis=1)
        ranks = rankdata(sampled, method="average", axis=1)
        draws[start:start+size] = (ranks[:, :np_].sum(axis=1)-np_*(np_+1)/2)/(np_*nn)
    lo, hi = np.quantile(draws, [.025, .975], method="linear")
    return {**result, "interval95": [float(lo), float(hi)], "degenerate": bool(lo == hi),
            "constant_distribution": bool(np.all(draws == draws[0])), "reason": None}


def confidence(scores, correct, *, calibration=False, seed=None):
    """Scores are eligible selected-option probabilities, never concentration."""
    if len(scores) != len(correct):
        raise ValueError("Confidence scores and correctness must align")
    try:
        ds = [s if isinstance(s, Fraction) else Fraction(str(s)) for s in scores]
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("Confidence requires finite rational scores") from exc
    if any(not 0 <= s <= 1 for s in ds) or any(c not in (0, 1, False, True) for c in correct):
        raise ValueError("Confidence requires finite scores in [0,1] and binary correctness")
    n, positives = len(ds), sum(correct)
    negatives = n-positives
    if positives and negatives:
        ranks = rankdata(ds, method="average")
        auc = (sum(r for r, c in zip(ranks, correct) if c)-positives*(positives+1)/2)/(positives*negatives)
        auc_reason = None
    else:
        auc, auc_reason = None, "single correctness class" if n else "no eligible scores"
    bins = []
    # Exact rational thresholds avoid moving two-decimal boundary values because
    # of floating-point rounding. Bins are [lower, upper), except the last includes 1.
    for i in range(10):
        indices = [j for j, s in enumerate(ds) if min(int(s*10), 9) == i]
        bins.append({"lower": i/10, "upper": (i+1)/10, "upper_inclusive": i == 9,
                     "n": len(indices), "mean_probability": float(sum((ds[j] for j in indices), Fraction(0))/len(indices)) if indices else None,
                     "correct": int(sum(correct[j] for j in indices)),
                     "accuracy_wilson95": wilson(int(sum(correct[j] for j in indices)), len(indices)),
                     "observed_accuracy": fraction(sum(correct[j] for j in indices), len(indices))})
    high = [i for i, s in enumerate(ds) if s >= Fraction(9, 10)]
    errors = sum(not correct[i] for i in high)
    curve, retained, failures = [], 0, 0
    grouped = {}
    for score, c in zip(ds, correct):
        entry = grouped.setdefault(score, [0, 0]); entry[0] += 1; entry[1] += not c
    for score in sorted(grouped, reverse=True):
        # Retain all tied scores at once; arbitrary ordering within a tie would
        # imply an unattainable confidence threshold and distort risk/coverage.
        group_n, group_errors = grouped[score]
        retained += group_n; failures += group_errors
        curve.append({"threshold": float(score), "threshold_exact": str(score), "retained": retained, "errors": failures,
                      "coverage": retained/n, "risk": failures/retained})
    return {"eligible_n": n, "auroc": float(auc) if auc is not None else None, "auroc_reason": auc_reason,
            "auroc_bootstrap": auroc_interval(ds, correct, seed=seed),
            "score_support": {"distinct_n": len(set(ds)), "minimum": float(min(ds)) if ds else None,
                              "maximum": float(max(ds)) if ds else None,
                              "zero_n": ds.count(Fraction(0)), "one_n": ds.count(Fraction(1)),
                              "correct_n": int(positives), "incorrect_n": int(negatives)},
            "binary_brier": float(sum(((s-int(c))**2 for s, c in zip(ds, correct)), Fraction(0))/n) if calibration and n else None,
            "binary_brier_reason": None if calibration and n else ("no eligible scores" if not n else "verified probability semantics required"),
            "reliability": bins if calibration else None,
            "calibration_gate": None if calibration else "verified probability semantics required",
            "high_confidence": {"threshold": .9, "n": len(high), "errors": errors, "error_rate": fraction(errors, len(high))},
            "risk_coverage": curve, "tie_rule": "whole ties", "coverage_denominator": n}


def pareto(points, dimensions):
    """Each dimension maps to maximize/minimize. Missing coordinates excluded locally."""
    included, excluded = {}, {}
    for name, point in points.items():
        absent = [d for d in dimensions if point.get(d) is None]
        if absent:
            excluded[name] = {"missing_coordinates": absent}; continue
        if any(not math.isfinite(point[d]) for d in dimensions):
            raise ValueError("Frontier coordinates must be finite or explicitly missing")
        included[name] = point
    # Dominance requires no worse in every coordinate and better in at least one.
    # Exact ties therefore remain jointly nondominated; no weighted score is used.
    dominates = lambda a, b: (all((a[d] >= b[d] if maximize else a[d] <= b[d]) for d, maximize in dimensions.items())
                             and any(a[d] != b[d] for d in dimensions))
    dominated_by = {name: [other for other, p in included.items() if other != name and dominates(p, point)]
                    for name, point in included.items()}
    return {"nondominated": [name for name in included if not dominated_by[name]],
            "dominated_by": dominated_by, "excluded": excluded, "dimensions": dimensions}


def operations(records, correct, *, complete_ledger=True, indeterminate_n=0, unstarted_n=0,
               duration_records=None):
    """Selected attempts for cost; supplied full call history for durations."""
    n = len(records)
    out = {"attempted_n": n+indeterminate_n, "terminal_n": n, "indeterminate_n": indeterminate_n, "unstarted_n": unstarted_n,
           "complete_ledger": complete_ledger, "status_counts": dict(Counter(r["status"] for r in records))}
    for field in ("observed_usd", "estimated_usd"):
        values = [r["billing"].get(field) for r in records]
        observed = [Decimal(str(x)) for x in values if x is not None]
        subtotal = sum(observed, Decimal(0))
        complete = complete_ledger and n > 0 and len(observed) == n
        # A known subtotal is useful audit evidence, but incomplete receipts or
        # an incomplete attempt ledger cannot yield a total or mean billed cost.
        out[field] = {"observed_n": len(observed), "missing_n": n-len(observed),
                      "observed_subtotal": float(subtotal) if observed else None, "complete": complete,
                      "total": float(subtotal) if complete else None,
                      "mean_per_attempt": float(subtotal/n) if complete else None,
                      "cost_per_correct": float(subtotal/correct) if complete and correct else None,
                      "cost_per_correct_reason": None if complete and correct else ("zero correct answers" if complete else "incomplete billing/attempt ledger"),
                      "missing_reasons": dict(Counter(r["billing"].get(field+"_missing_reason", "unspecified") for r, x in zip(records, values) if x is None)),
                      "provenance": sorted({str(r["billing"].get("provenance")) for r in records}),
                      "rate_dates": sorted({str(r["billing"].get("rate_date")) for r in records})}
    duration_records = records if duration_records is None else duration_records
    times = [r["timing"].get("request_wall_seconds") for r in duration_records]
    out["latency"] = quantiles(t for t in times if t is not None)
    measured = [t for t in times if t is not None]
    duration_complete = complete_ledger and len(times) > 0 and len(measured) == len(times)
    out["latency"].update(missing_n=sum(t is None for t in times), complete=duration_complete,
                          calls_n=len(times), observed_sum_seconds=sum(measured) if measured else None,
                          total_seconds=sum(measured) if duration_complete else None,
                          missing_reasons=dict(Counter(r["timing"].get("missing_reason", "unspecified") for r, t in zip(duration_records, times) if t is None)))
    waits = [r["timing"].get("local_queue_wait_seconds") for r in duration_records]
    out["local_queue_wait"] = {**quantiles(t for t in waits if t is not None),
                               "missing_n": sum(t is None for t in waits),
                               "complete": complete_ledger and len(waits) > 0 and all(t is not None for t in waits),
                               "missing_reasons": dict(Counter(r["timing"].get("queue_wait_missing_reason", "not_recorded_in_historical_package") for r, t in zip(duration_records, waits) if t is None)),
                               "definition": "from phase-ready queue entry to transport start; separate from request wall duration"}
    # Contradictory native counters remain audit observations. They cannot
    # establish total generated/billed tokens; monetary receipts stand alone.
    inconsistent = [r for r in records if isinstance(r["usage"].get("accounting"), dict)
                    and r["usage"]["accounting"].get("reasoning_in_output") is True
                    and r["usage"].get("reasoning_tokens") is not None
                    and r["usage"].get("output_tokens") is not None
                    and r["usage"]["reasoning_tokens"] > r["usage"]["output_tokens"]]
    out["token_counter_consistency"] = {
        "inconsistent_n": len(inconsistent),
        "reason": "reasoning_exceeds_output_claimed_to_include_reasoning" if inconsistent else None,
        "attempt_identities": [r.get("identity") for r in inconsistent],
        "interpretation": "native reported counters; no universal billed-token inference"}
    out["tokens"] = {}
    for field in ("input_tokens", "reasoning_tokens", "final_tokens", "output_tokens", "billed_tokens"):
        values = [r["usage"].get(field) for r in records]
        observed = [x for x in values if x is not None]
        out["tokens"][field] = {"observed_subtotal": sum(observed) if observed else None,
                               "missing_n": n-len(observed), "total": sum(observed) if complete_ledger and n and len(observed) == n else None,
                               "missing_reasons": dict(Counter(r["usage"].get(field+"_missing_reason", "unspecified") for r, v in zip(records, values) if v is None))}
        if inconsistent and field in {"output_tokens", "reasoning_tokens", "billed_tokens"}:
            out["tokens"][field].update(total=None, total_reason="inconsistent_native_token_counters")
    # Billed/output fields stand alone: never add reasoning to output tokens.
    out["token_accounting"] = sorted({str(r["usage"].get("accounting")) for r in records})
    out["service_dates"] = sorted({r["timing"].get("started_at", "unavailable")[:10] for r in records})
    routing = Counter(json.dumps({k: r.get("route", {}).get(k) for k in ("requested_model", "returned_model", "requested_route", "returned_route", "effective_controls")}, sort_keys=True) for r in records)
    out["routing"] = [{**json.loads(key), "terminal_n": count} for key, count in sorted(routing.items())]
    return out


def probability_change(earlier, later, earlier_label):
    """Requires normalized vectors supplied by an accepted policy, not native repair."""
    if set(earlier) != set(later) or earlier_label not in earlier:
        raise ValueError("Probability stability requires the same supplied labels")
    a, b = {k: Fraction(v) for k, v in earlier.items()}, {k: Fraction(v) for k, v in later.items()}
    if sum(a.values()) != 1 or sum(b.values()) != 1:
        raise ValueError("Probability stability requires normalized vectors")
    return {"total_variation": float(sum(abs(a[k]-b[k]) for k in a)/2),
            "earlier_selected_probability_change": float(abs(a[earlier_label]-b[earlier_label]))}
