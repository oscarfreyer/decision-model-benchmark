"""Separate, descriptive failure recovery. Original primary analysis stays intact."""
import argparse
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import sys

from benchmarking.common import (ROOT, attempt_identity, file_hash, load_keys,
    load_questions, run_environment_record, write_new)
from benchmarking.retry_common import read_retry_package, source_context


def measurements(records, section, field, *, unresolved=0):
    values = [r[section][field]['value'] for r in records]
    known = [Decimal(str(value)) for value in values if value is not None]
    missing = len(values)-len(known)+unresolved
    return {'total': str(sum(known)) if known and not missing else None,
            'known_sum': str(sum(known)) if known else None,
            'missing': missing, 'observed': len(known)}


def recovery_report(package, keys_path):
    """First-valid selection uses only status/index. Keys score after selection."""
    package = Path(package)
    before = {name: file_hash(package/name) for name in ('manifest.json', 'events.jsonl', 'provenance.json')}
    manifest, attempts, coverage = read_retry_package(package)
    if file_hash(keys_path) != manifest['bindings']['keys']:
        raise ValueError('recovery requires exact original-key bytes')
    questions = load_questions(package/'source/questions.jsonl')
    keys = load_keys(keys_path, questions)
    originals, source_manifests, _ = source_context(package, manifest)
    grouped = defaultdict(list)
    for attempt in attempts:
        grouped[attempt['target_id']].append(attempt)
    rows = []; runs = {}; categories = defaultdict(lambda: {'original_failures': 0, 'recovered': 0, 'exhausted': 0, 'retry_starts': 0})
    for tid, original in originals.items():
        ident = attempt_identity(original); source = source_manifests[tid]
        run = next(r for r in source['runs'] if r['run_id'] == ident[-1])
        trials = sorted(grouped[tid], key=lambda a: a['retry_index'])
        first_valid = next((a for a in trials if a['record']['status'] == 'valid'), None)
        selected = original if original['status'] == 'valid' or first_valid is None else first_valid['record']
        answer = keys[ident[:3]]['reference_answer']
        original_correct = original['status'] == 'valid' and original['selected_label'] == answer
        recovered_correct = selected['status'] == 'valid' and selected['selected_label'] == answer
        starts = coverage['started_by_target'].get(tid, 0)
        unresolved = original['status'] != 'valid' and first_valid is None and starts < 10
        if any(t == tid for t, _ in coverage['indeterminate']):
            unresolved = True
        all_records = [original]+[a['record'] for a in trials]
        row = {'identity': original['identity'], 'experiment_id': run['experiment_id'],
            'role': run['role'], 'replicate': run['replicate'], 'original_status': original['status'],
            'original_answer': original['selected_label'], 'reference_answer': answer,
            'original_correct': original_correct, 'recovery_status': selected['status'],
            'recovery_answer': selected['selected_label'], 'recovery_correct': recovered_correct,
            'selected_retry_index': first_valid['retry_index'] if first_valid else None,
            'retry_starts': starts, 'retry_finishes': len(trials), 'pending_recovery': unresolved,
            'all_attempt_charge_usd': measurements(all_records, 'billing', 'observed_usd', unresolved=starts-len(trials)),
            'all_attempt_request_seconds': measurements(all_records, 'timing', 'duration_seconds', unresolved=starts-len(trials))}
        rows.append(row)
        summary = runs.setdefault(ident[-1], {'config_id': ident[3], 'role': run['role'],
            'replicate': run['replicate'], 'denominator': 0, 'original_correct': 0,
            'recovery_correct': 0, 'original_failures': 0, 'final_failures': 0,
            'retry_starts': 0, 'pending_recovery': 0, 'records': [], 'unresolved_starts': 0})
        summary['denominator'] += 1; summary['original_correct'] += original_correct
        summary['recovery_correct'] += recovered_correct
        summary['original_failures'] += original['status'] != 'valid'
        summary['final_failures'] += selected['status'] != 'valid'
        summary['retry_starts'] += starts; summary['pending_recovery'] += unresolved
        summary['unresolved_starts'] += starts-len(trials)
        summary['records'].extend(all_records)
        if original['status'] != 'valid':
            category = categories[(ident[3], original['status'])]
            category['original_failures'] += 1; category['recovered'] += first_valid is not None
            category['exhausted'] += starts == 10 and not first_valid and not unresolved
            category['retry_starts'] += starts
    for summary in runs.values():
        records = summary.pop('records')
        summary['original_accuracy'] = summary['original_correct']/summary['denominator']
        summary['recovery_accuracy'] = summary['recovery_correct']/summary['denominator'] if not summary['pending_recovery'] else None
        summary['all_attempt_charge_usd'] = measurements(records, 'billing', 'observed_usd', unresolved=summary['unresolved_starts'])
        summary['all_attempt_request_seconds'] = measurements(records, 'timing', 'duration_seconds', unresolved=summary['unresolved_starts'])
        if summary['pending_recovery']:
            summary['final_failures'] = None
    for run_id, summary in runs.items():
        original_run = [r for r in originals.values() if r['identity']['run_id'] == run_id]
        retry_run = [a['record'] for a in attempts if a['record']['identity']['run_id'] == run_id]
        summary['original_request_window'] = window(original_run)
        summary['retry_request_window'] = window(retry_run)
    after = {name: file_hash(package/name) for name in before}
    if before != after or file_hash(keys_path) != manifest['bindings']['keys']:
        raise ValueError('retry inputs changed during recovery analysis')
    return {'analysis_role': 'secondary_recovery', 'complete': coverage['complete'],
        'source_bindings': manifest['sources'], 'retry_bindings': before,
        'retry_policy': manifest['policy'],
        'analysis_code': {name: file_hash(ROOT/name) for name in
            ('analysis/support/recovery.py', 'benchmarking/retry_common.py', 'benchmarking/common.py')},
        'analysis_environment': run_environment_record(),
        'keys_sha256': manifest['bindings']['keys'], 'coverage': coverage, 'runs': runs,
        'failures_by_model_category': [{'config_id': cid, 'original_status': status, **values}
            for (cid, status), values in sorted(categories.items())], 'responses': rows}


def window(records):
    if not records:
        return None
    # The reviewed incomplete-response timestamp is a resolution date, not a
    # response finish. Keep it out of the observed response window.
    observed = [r for r in records if 'recovery_review' not in r]
    if not observed:
        return None
    return {'first_started_at': min((r['timing']['started_at'] for r in observed), key=datetime.fromisoformat),
            'last_finished_at': max((r['timing']['finished_at'] for r in observed), key=datetime.fromisoformat)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--keys', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError('new recovery output directory required')
        report = recovery_report(args.package, args.keys)
        if not report['complete']:
            raise ValueError('complete recovery/exhaustion required for export')
        args.output.mkdir(parents=True, exist_ok=False)
        write_new(args.output/'recovery.json', report)
        print(str(args.output/'recovery.json'))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print('Recovery analysis blocked: '+str(exc), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
