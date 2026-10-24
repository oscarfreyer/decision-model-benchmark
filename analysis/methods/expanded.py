"""SAP 1.8 reporting families and common selected-final-attempt resources."""
from fractions import Fraction
from itertools import combinations

GENERATIVE = tuple(f"G{i:02d}" for i in range(1, 16))
DECISIONS = tuple(f"D{i:02d}" for i in range(1, 14))
ROSTER = GENERATIVE + DECISIONS
PAIRS = {"P03": {"lower": "G02", "upper": "G03"},
         "P07": {"lower": "G06", "upper": "G07"},
         "P09": {"lower": "G09", "upper": "G10"}}
WORKFLOW = "analysis-reporting-ids-v1"
RESOURCE_POLICY = "primary_selected_final_within_configuration_mean"


def contrasts():
    return ([("S1", f"{a}-{b}", a, b) for a in DECISIONS for b in GENERATIVE]
            + [("S1", f"{a}-{b}", a, b) for a, b in combinations(DECISIONS, 2)]
            + [("S2", k, p["upper"], p["lower"]) for k, p in PAIRS.items()])


def paired(matrix, configs, pairs, *, seed, valid_matrix=None, inference=True):
    from analysis.methods.pipeline import paired_contrasts
    if set(matrix) != set(ROSTER) or set(configs) != set(ROSTER) or pairs != PAIRS:
        raise ValueError("SAP 1.8 requires all 28 configurations for the 273/3 families")
    return paired_contrasts(matrix, configs, contrasts(), {"S1": 273, "S2": 3},
                            seed=seed, valid_matrix=valid_matrix, inference=inference)


def impute_resources(summaries, rows_by_config):
    """Assign separately from raw data; donors include zero and terminal failures."""
    from analysis.methods.statistics import quantiles
    for cid, result in summaries.items():
        rows = rows_by_config[cid]
        if any(r['identity']['config_id'] != cid or r['identity']['run_id'] != result['run_id'] for r in rows):
            raise ValueError("Resource donors cross configuration/run boundaries")
        for metric, section, field, observed_key, analysis_key in (
            ('cost', 'billing', 'observed_usd', 'observed_usd', 'analysis_usd'),
            ('duration', 'timing', 'request_wall_seconds', 'latency', 'analysis_latency')):
            observed = result['operations'][observed_key]
            value = lambda r: (r['timing']['duration_seconds']['value']
                if metric == 'duration' and 'duration_seconds' in r['timing'] else r[section][field])
            values = [Fraction(str(value(r))) for r in rows if value(r) is not None]
            if any(v < 0 for v in values):
                raise ValueError('Negative resource donor')
            missing = [r for r in rows if value(r) is None]
            subtotal = sum(values, Fraction(0))
            mean = subtotal / len(values) if values else None
            assign = mean is not None and result['complete']
            assignments = [{"identity": r['identity'], "source_identity": r.get('source_identity', r['identity']),
                "attempt_id": r.get('analysis_selection', {}).get('selected_attempt_id'),
                "assigned_value": float(mean), "assigned_value_exact": str(mean)} for r in missing] if assign else []
            complete = result['complete'] and bool(rows) and (not missing or assign)
            adjusted = {**observed, 'complete': complete,
                'basis': 'selected final attempts; observed plus within-configuration primary-run mean imputation',
                'imputed_n': len(assignments), 'unresolved_n': len(missing)-len(assignments),
                'imputation': {'policy': RESOURCE_POLICY, 'donor_n': len(values),
                    'observed_subtotal_exact': str(subtotal), 'donor_mean_exact': str(mean) if mean is not None else None,
                    'donor_mean': float(mean) if mean is not None else None, 'assignments': assignments}}
            total = subtotal + len(assignments)*(mean or 0)
            if metric == 'cost':
                adjusted.update(total=float(total) if complete else None,
                    mean_per_attempt=float(total/result['planned_n']) if complete else None,
                    imputed_subtotal=float(total-subtotal),
                    cost_per_correct=float(total/result['correct']) if complete and result['correct'] else None,
                    cost_per_correct_reason=None if complete and result['correct'] else
                        'zero correct answers' if complete else 'incomplete resource basis')
            else:
                adjusted.update(quantiles([float(v) for v in values]
                    + [a['assigned_value'] for a in assignments]))
                adjusted.update(total_seconds=float(total) if complete else None)
            result['operations'][analysis_key] = adjusted
    return {'policy': RESOURCE_POLICY, 'duration_basis': 'same selected final attempt as cost',
            'repeat_resources': 'observed only'}


def secondary(manifest, runs, records, keys, coverage, methods, seed):
    from analysis.methods.decisions import native_vector
    from analysis.methods.pipeline import confidence_for_run, stability
    out = {}
    for cid in DECISIONS:
        config_methods = methods.get('decision_models', {}).get(cid, {})
        reviewed = config_methods.get('probability_semantics_verified') is True
        def reader(row, policy, provenance):
            if not reviewed:
                return None, 'probability_fields_unreviewed'
            values, reason = native_vector(row, sum_tolerance='0.02')
            return ({k: v/sum(values.values()) for k, v in values.items()}, None) if values else (None, reason)
        primary = next(r for r in runs if r['config_id'] == cid and r['role'] == 'primary')
        policy = 'native_selected_normalized_distribution'
        conf = confidence_for_run(manifest, primary, records, keys, policy,
            methods={'probability_semantics_verified': reviewed, 'bootstrap_seed': seed},
            complete=all((manifest['benchmark'], manifest['split'], q['item_id'], cid, primary['run_id']) in records for q in manifest['items']),
            vector_reader=reader)
        repeat = stability(manifest, runs, records, keys, coverage, policy,
                           config_id=cid, vector_reader=reader)
        if not all(r['complete'] for r in repeat.get('runs', {}).values()):
            repeat.pop('answer_flows', None)
        out[cid] = {'confidence': conf, 'stability': repeat, 'sum_tolerance': '0.02',
                    'probability_semantics_verified': reviewed}
    return out
