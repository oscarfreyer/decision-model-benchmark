"""Read-only, namespace-aware view over historical inputs and explicit new packages.

Raw records, run IDs and source files retain their original identities. Only the
verified in-memory analysis view uses reporting IDs. Never discover input paths.
"""
from copy import deepcopy
from pathlib import Path
from collections import defaultdict

from common import ROOT, attempt_identity, fingerprint
from analysis.methods.expanded import ROSTER, DECISIONS, WORKFLOW
from analysis.dataset.inputs import load_input_folder, folder_inventory


def mapped_configuration(config, namespace):
    from benchmarking.configuration_ids import canonical_configuration
    canonical = canonical_configuration(namespace, config['config_id'],
        model_id=config['model_id'], effort=config['effort'])
    return {**config, 'config_id': canonical['config_id'], 'source_config_id': config['config_id'],
            'source_namespace': namespace, 'display_name': canonical['display_name']}


def map_record(record, mapping, namespace):
    source = deepcopy(record)
    cid = mapping[record['identity']['config_id']]
    return {**deepcopy(record), 'identity': {**record['identity'], 'config_id': cid},
            'source_identity': deepcopy(record['identity']), 'source_namespace': namespace,
            'source_record': source, 'source_record_sha256': fingerprint(source)}


def select_extension(records, manifest, package):
    """Keep pending recovery absent; failures become outcomes only at attempt 10."""
    grouped = defaultdict(list)
    for record in records:
        grouped[attempt_identity(record)].append(record)
    selected = []
    for ident, trials in grouped.items():
        trials.sort(key=lambda r: r['attempt_index'])
        indices = [r['attempt_index'] for r in trials]
        if indices != list(range(len(trials))) or len(trials) > 11:
            raise ValueError('Duplicate or noncontiguous extension attempts')
        valid = [r for r in trials if r['status'] == 'valid']
        if valid and valid[0] is not trials[-1]:
            raise ValueError('Attempt after first valid response')
        if not valid and indices[-1] != 10:
            continue
        final = valid[0] if valid else trials[-1]
        outcome = 'original_valid' if final['attempt_index'] == 0 else 'recovered' if valid else 'exhausted'
        selected.append({**final, 'original_attempt': trials[0],
            'analysis_selection': {'outcome': outcome, 'selected_retry_index': final['attempt_index'],
                'selected_attempt_id': final['attempt_id'], 'cost_attempt_id': final['attempt_id'],
                'duration_attempt_id': final['attempt_id'],
                'duration_attempt_ids': [r['attempt_id'] for r in trials]},
            'analysis_attempts': [{'record': r, 'retry_index': r['attempt_index'],
                'source_package': package} for r in trials]})
    return selected


def assemble_view(sources):
    """Sources are independently verified (manifest, selected records, coverage, namespace)."""
    from benchmarking.configuration_ids import canonical_configuration, CURRENT_NAMESPACE
    first = sources[0][0]
    configs, runs, records, source_packages = {}, [], [], []
    conditions, unresolved, pending = {}, set(), set()
    for manifest, rows, coverage, namespace in sources:
        if any(manifest[k] != first[k] for k in ('benchmark', 'split', 'items', 'provenance')):
            raise ValueError('Expanded inputs disagree on question order or provenance')
        if any(manifest['bindings'][k] != first['bindings'][k] for k in ('questions', 'keys')):
            raise ValueError('Expanded question/key hashes differ')
        mapping = {}
        for config in manifest['configurations']:
            mapped = mapped_configuration(config, namespace)
            cid = mapped['config_id']
            if cid in configs:
                raise ValueError('Duplicate reporting configuration across source namespaces')
            mapping[config['config_id']] = cid
            configs[cid] = mapped
        for run in manifest['runs']:
            runs.append({**run, 'config_id': mapping[run['config_id']],
                         'source_config_id': run['config_id'], 'source_namespace': namespace})
        records.extend(map_record(row, mapping, namespace) for row in rows)
        if manifest.get('execution_workflow') == 'decision-extension-v1':
            from benchmarking.extension_contract import targets
            by_query = targets(manifest)
            for query in coverage.get('indeterminate', []):
                unresolved.add(attempt_identity({'identity': by_query[query]['identity']}))
            for query in coverage.get('pending', []):
                if query not in coverage.get('unstarted', []):
                    pending.add(attempt_identity({'identity': by_query[query]['identity']}))
        for ident in coverage.get('indeterminate', []):
            # Source readers may expose attempt-level metadata; missing selected
            # outcomes below remain unresolved regardless of retry representation.
            if isinstance(ident, (list, tuple)) and len(ident) == 5 and ident[3] in mapping:
                unresolved.add((*ident[:3], mapping[ident[3]], ident[4]))
        source_packages.append({'manifest': manifest, 'manifest_sha256': fingerprint(manifest),
                                'source_namespace': namespace})
        conditions.update(coverage.get('collection_conditions', {}))
        conditions[fingerprint(manifest)] = {'campaign': manifest.get('campaign', namespace),
            'schedule': manifest.get('schedule'), 'execution': manifest.get('execution'),
            'source_packages': manifest.get('source_packages')}
    absent = sorted(set(ROSTER)-set(configs))
    for cid in absent:
        c = canonical_configuration(CURRENT_NAMESPACE, cid)
        configs[cid] = {**c, 'base_model': c['model_id'], 'pair_id': None,
                       'adapter': 'uncollected', 'source_namespace': CURRENT_NAMESPACE}
    expected_runs = {(cid, rep) for cid in ROSTER for rep in ((1, 2, 3) if cid in DECISIONS else (1,))}
    actual = [(r['config_id'], r['replicate']) for r in runs]
    if len(set(actual)) != len(actual) or not set(actual) <= expected_runs:
        raise ValueError('Duplicated or unexpected model/run role')
    for run in runs:
        if (run['role'], run['experiment_id']) != (('primary', 'E1') if run['replicate'] == 1 else ('secondary', 'E2')):
            raise ValueError('Invalid primary/repeat role')
    for cid, rep in sorted(expected_runs-set(actual)):
        runs.append({'config_id': cid, 'replicate': rep, 'role': 'primary' if rep == 1 else 'secondary',
                     'experiment_id': 'E1' if rep == 1 else 'E2',
                     'run_id': f'uncollected:{cid}:run{rep}', 'uncollected': True})
    if len({r['run_id'] for r in runs}) != len(runs):
        raise ValueError('Source run IDs collide')
    view = {**first, 'execution_workflow': WORKFLOW, 'configuration_namespace': CURRENT_NAMESPACE,
            'configurations': [configs[cid] for cid in ROSTER],
            'runs': sorted(runs, key=lambda r: (r['replicate'], ROSTER.index(r['config_id']))),
            'source_packages': source_packages, 'missing_configurations': absent}
    expected = {(view['benchmark'], view['split'], q['item_id'], r['config_id'], r['run_id'])
                for r in runs for q in view['items']}
    identities = [attempt_identity(r) for r in records]
    if len(set(identities)) != len(identities) or not set(identities) <= expected:
        raise ValueError('Duplicate or unexpected selected outcomes')
    missing = expected-set(identities)
    unresolved &= missing
    coverage = {'expected': len(expected), 'terminal': len(records), 'complete': not missing,
                'unstarted': [list(i) for i in sorted(missing-unresolved)],
                'indeterminate': [list(i) for i in sorted(unresolved)],
                'collection_conditions': conditions, 'last_event_sha256': None,
                'missing_configurations': absent, 'pending_outcomes': len(missing),
                'pending_recovery': [list(i) for i in sorted(pending & missing)],
                'missing_outcome_reasons': {fingerprint(list(i)): 'indeterminate' if i in unresolved else 'pending_recovery' if i in pending else 'unstarted' for i in missing}}
    return view, records, coverage


def validate_view(manifest):
    if manifest.get("public_derivative") == "structured-results-v1":
        from analysis.public_release import validate_public_manifest
        return validate_public_manifest(manifest)
    from benchmarking.configuration_ids import CURRENT_NAMESPACE
    if manifest.get('configuration_namespace') != CURRENT_NAMESPACE:
        raise ValueError('Unknown reporting namespace')
    if [c['config_id'] for c in manifest['configurations']] != list(ROSTER):
        raise ValueError('Expanded view requires the ordered 28-model roster')
    for package in manifest['source_packages']:
        if fingerprint(package['manifest']) != package['manifest_sha256']:
            raise ValueError('Source manifest binding changed')
    expected = {(cid, rep) for cid in ROSTER for rep in ((1, 2, 3) if cid in DECISIONS else (1,))}
    actual = [(r['config_id'], r['replicate']) for r in manifest['runs']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('Expanded run roster changed')
    return manifest


def validate_mapped_record(record, manifest):
    if manifest.get("public_derivative") == "structured-results-v1":
        from analysis.public_release import validate_public_record
        return validate_public_record(record, manifest)
    from benchmarking.configuration_ids import canonical_configuration
    source = record['source_record']
    if fingerprint(source) != record['source_record_sha256']:
        raise ValueError('Source record changed after verification')
    raw_id = source['identity']
    cid = canonical_configuration(record['source_namespace'], raw_id['config_id'])['config_id']
    if record['identity'] != {**raw_id, 'config_id': cid}:
        raise ValueError('Mapped identity disagrees with source namespace')
    extras = {'source_record', 'source_record_sha256', 'source_identity', 'source_namespace'}
    if {k: v for k, v in record.items() if k not in extras | {'identity'}} != {k: v for k, v in source.items() if k != 'identity' and k not in extras}:
        raise ValueError('Mapped response differs from verified source')


def load_expanded_inputs(root, digest, extension_packages=(), *, repository_root=ROOT):
    """Resolve extension paths from the repository, retaining absolute paths/digests."""
    from benchmarking.configuration_ids import ORIGINAL_NAMESPACE, LUNA_NAMESPACE, CURRENT_NAMESPACE
    from benchmarking.common import read_package
    legacy, terminal, coverage, keys, questions, hashes = load_input_folder(root, digest)
    sources = []
    for original in legacy['source_manifests']:
        namespace = ORIGINAL_NAMESPACE if original['execution_workflow'] == 'lean-v1' else LUNA_NAMESPACE
        run_ids = {r['run_id'] for r in original['runs']}
        sources.append((original, [r for r in terminal if r['identity']['run_id'] in run_ids], coverage, namespace))
    base = Path(repository_root).resolve()
    paths = [(base / Path(selection['path'])).resolve() for selection in extension_packages]
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate extension package')
    for path, selection in zip(paths, extension_packages):
        before = folder_inventory(path)
        if fingerprint(before) != selection['inventory_sha256']:
            raise ValueError('Extension package differs from independent inventory digest')
        manifest, attempts, extension_coverage = read_package(path)
        if manifest.get('execution_workflow') != 'decision-extension-v1':
            raise ValueError('Select an explicit decision-extension-v1 package')
        if manifest.get('config_namespace') != CURRENT_NAMESPACE:
            raise ValueError('Extension namespace mismatch')
        if extension_coverage.get('drift'):
            raise ValueError('Extension identity/schema drift requires resolution before analysis')
        if manifest.get('provenance') == 'development' or manifest.get('purpose') == 'development':
            raise ValueError('Development collection cannot enter evaluation analysis')
        rows = select_extension(attempts, manifest, path.name)
        if len(rows) != extension_coverage['terminal']:
            raise ValueError('Selected outcome count differs from verified collection ledger')
        sources.append((manifest, rows, extension_coverage, CURRENT_NAMESPACE))
        if before != folder_inventory(path):
            raise ValueError('Extension package changed while loading')
        hashes['extension:'+selection['inventory_sha256']] = selection['inventory_sha256']
    view, records, coverage = assemble_view(sources)
    return view, records, coverage, keys, questions, hashes


def decision_settings(manifest, *, field_review_path=None, input_hashes=None):
    """Carry reviewed field evidence into the bound method settings."""
    if manifest.get("public_derivative") == "structured-results-v1":
        return deepcopy(manifest["reviewed_decision_settings"])
    settings = {}
    for config in manifest['configurations']:
        cid = config['config_id']
        if cid not in DECISIONS:
            continue
        evidence = config.get('probability_semantics', {})
        historical = cid in ('D01', 'D02') and config.get('source_config_id') is not None
        fields = config.get('scalar_semantics', {})
        settings[cid] = {'sum_tolerance': '0.02',
            'probability_semantics_verified': historical or (isinstance(evidence, dict) and evidence.get('reviewed') is True and bool(evidence.get('evidence', '').strip())),
            'field_review_evidence': evidence if not historical else 'accepted historical JEV/Luna field review',
            'scalar_semantics': fields,
            'scalar_fields': list(fields) if fields else ['concentration'] if cid == 'D01' else ['provider_confidence']}
    if field_review_path is not None:
        _apply_field_review(settings, manifest, field_review_path, input_hashes or {})
    return settings


def _apply_field_review(settings, manifest, path, input_hashes):
    """Apply analysis evidence only to its exact verified collection package."""
    from common import strict_json
    raw = Path(path).read_bytes()
    review = strict_json(raw.decode('utf-8'))
    if review.get('schema_version') != 1 or not all(
            isinstance(review.get(k), str) and review[k].strip()
            for k in ('review_id', 'reviewed_at', 'reviewer', 'source_record')):
        raise ValueError('Invalid decision field-review metadata')
    packages = [p for p in manifest.get('source_packages', [])
                if p['manifest'].get('execution_workflow') == 'decision-extension-v1']
    if not packages:
        return  # Historical-only exploration retains the original eligibility.
    if len(packages) != 1:
        raise ValueError('Decision field review requires one bound extension package')
    package = packages[0]
    bound = package['manifest']
    if (package['manifest_sha256'] != review.get('package_manifest_sha256') or
            fingerprint(bound) != review.get('package_manifest_sha256')):
        raise ValueError('Decision field review belongs to a different manifest')
    inventory = review.get('package_inventory_sha256')
    if not isinstance(inventory, str) or input_hashes.get('extension:'+inventory) != inventory:
        raise ValueError('Decision field review lacks its verified package inventory')
    models = review.get('models', {})
    if set(models) != set(bound['models']):
        raise ValueError('Decision field-review roster differs from its package')
    configs = {c['config_id']: c for c in manifest['configurations']}
    import hashlib
    review_hash = hashlib.sha256(raw).hexdigest()
    proposed = {}
    for cid, evidence in models.items():
        model = bound['models'][cid]
        semantics = evidence.get('probability_semantics', {})
        if (cid not in settings or evidence.get('source_model_sha256') != fingerprint(model)
                or evidence.get('model_id') != model['model_id']
                or configs[cid]['model_id'] != model['model_id']
                or evidence.get('returned_model') != model['expected_returned_model']
                or evidence.get('provider') != model['route']
                or evidence.get('surface') != model['surface']
                or evidence.get('labels') != list('ABCD')
                or evidence.get('vector_field') != 'answers.answer.probabilities'
                or evidence.get('selected_label_field') != 'answers.answer.choice'
                or type(semantics.get('reviewed')) is not bool
                or semantics.get('meaning') != 'native_label_indexed_choice_probability'
                or not isinstance(semantics.get('evidence'), str)
                or not semantics['evidence'].strip()
                or not evidence.get('sources') or not evidence.get('development_evidence')
                or set(evidence.get('scalar_semantics', {})) != set(model['scalar_semantics'])):
            raise ValueError('Incomplete or mismatched decision field review: '+cid)
        for field, meaning in evidence['scalar_semantics'].items():
            if meaning == 'unreviewed':
                continue
            scalar_review = evidence.get('scalar_review_evidence', {}).get(field, {})
            if (scalar_review.get('reviewed') is not True or scalar_review.get('meaning') != meaning
                    or not isinstance(scalar_review.get('evidence'), str)
                    or not scalar_review['evidence'].strip() or not scalar_review.get('source')):
                raise ValueError('Missing scalar field-review evidence: '+cid+' '+field)
        proposed[cid] = {**settings[cid],
            'probability_semantics_verified': semantics['reviewed'],
            'field_review_evidence': {
                **{k: review[k] for k in ('review_id', 'reviewed_at', 'reviewer',
                    'source_record', 'package_manifest_sha256', 'package_inventory_sha256')},
                'review_file_sha256': review_hash, 'model': evidence},
            'scalar_semantics': dict(evidence['scalar_semantics'])}
    settings.update(proposed)
