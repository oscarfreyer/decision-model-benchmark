"""Current twelve-question E0 prerequisites; no calls or keys."""
from datetime import datetime
from pathlib import Path
import re

from common import BUDGET_ROSTERS, budget_roster_decision, file_hash, fingerprint, operational_blockers, strict_json, native_effort_request_valid, unknown_region_accepted, token_limit_request_valid
from benchmarking import adapters

PURPOSE = 'twelve_question_parallel_verification'
QUESTION_COUNT = 12
PARALLEL_DECISION = 'E0-parallel-12-20261006'
PROVIDER_LIMITS_DECISION = 'provider-concurrency-overrides-20261006'
SCOPE_FIELDS = {'purpose', 'expected_items', 'questions_sha256', 'keys_sha256', 'preparation_manifest_sha256'}
def endpoint_settings_binding(model):
    return fingerprint({k:v for k,v in model.items() if k not in
        {'control_evidence','control_evidence_sha256','display_name','effort','request_settings'}})


def development_blockers(config, manifest=None):
    """Allow collection of live evidence, never undocumented requests or an E1/E2 bypass."""
    errors=operational_blockers(config)
    if 'D-10' not in config.get('accepted_decisions',[]):
        errors.append('accepted D-10 development verification decision required')
    scope=config.get('development_validation',{})
    if PARALLEL_DECISION not in config.get('accepted_decisions',[]):
        errors.append('accepted twelve-question E0 decision required')
    count=QUESTION_COUNT
    if set(scope)!=SCOPE_FIELDS or scope.get('purpose')!=PURPOSE or type(scope.get('expected_items')) is not int or scope['expected_items']!=QUESTION_COUNT:
        errors.append('checksum-bound accepted development scope required')
    execution=config.get('execution',{})
    configurable_limits=PROVIDER_LIMITS_DECISION in config.get('accepted_decisions',[])
    if ((configurable_limits and execution.get('scheduler_version')!=2)
            or (not configurable_limits and (execution.get('concurrency')!=4
                or execution.get('provider_concurrency')!=1 or execution.get('provider_concurrency_overrides')))):
        errors.append('accepted twelve-question stage requires accepted configurable provider limits or the historical four-worker/provider-one settings')
    decision=budget_roster_decision(config)
    if decision is None or tuple(config.get('model_roster',[]))!=BUDGET_ROSTERS[decision]:
        errors.append('accepted twelve-question scope requires retained sixteen-configuration budget roster')
    for field in SCOPE_FIELDS-{'purpose','expected_items'}:
        if not isinstance(scope.get(field),str) or not re.fullmatch('[0-9a-f]{64}',scope[field]):
            errors.append('development scope digest required: '+field)
    if manifest is not None:
        if (manifest.get('provenance')!='development' or manifest.get('benchmark')!='medqa'
                or manifest.get('split')!='development' or len(manifest.get('items',[]))!=count
                or len(manifest.get('runs',[]))!=len(config['model_roster']) or any(r.get('experiment_id')!='E0' for r in manifest.get('runs',[]))):
            errors.append(f'D-10 permits only {count} development questions and the exact configured E0 roster')
        for field,binding in (('questions_sha256','questions'),('keys_sha256','keys')):
            if scope.get(field)!=manifest.get('bindings',{}).get(binding):
                errors.append('development package outside approved scope: '+field)
        if scope.get('preparation_manifest_sha256')!=manifest.get('key_manifest',{}).get('preparation_sha256'):
            errors.append('development preparation scope mismatch')
    for cid,model in config.get('models',{}).items():
        expected_adapter='jev_gateway' if cid=='J01' else 'openrouter'
        if model.get('adapter')!=expected_adapter:
            errors.append(cid+': selected OpenRouter adapter required')
        for field in ('route','version','endpoint','control_evidence','control_evidence_sha256'):
            if not isinstance(model.get(field),str) or not model[field].strip():
                errors.append(cid+': '+field+' pending')
        endpoint=model.get('endpoint')
        allowed={adapters.JEV_DECISIONS_ENDPOINT} if cid=='J01' else adapters.OPENROUTER_ENDPOINTS
        if endpoint not in allowed:
            errors.append(cid+': unapproved transport endpoint')
        if not model.get('control_evidence'):
            if not model.get('region'): errors.append(cid+': region pending')
            continue
        try:
            path=Path(model['control_evidence']);evidence=strict_json(path.read_text())
            if file_hash(path)!=model.get('control_evidence_sha256'):
                raise ValueError('documentary prerequisite hash mismatch')
            if not model.get('region') and not unknown_region_accepted(config,model,evidence):
                raise ValueError('region pending; bound human unknown-region exception required')
            if (evidence.get('schema_version')!=2 or evidence.get('type')!='endpoint_prerequisites'
                    or evidence.get('evidence_stage')!='documentation' or not evidence.get('reviewer')
                    or evidence.get('settings_sha256')!=endpoint_settings_binding(model)):
                raise ValueError('reviewed setting-bound documentary prerequisites required')
            datetime.fromisoformat(evidence['reviewed_at'])
            sources=evidence.get('sources')
            if not isinstance(sources,list) or not sources:
                raise ValueError('documentary source URLs and access dates required')
            for source in sources:
                if not source.get('url','').startswith('https://') or not source.get('role'):
                    raise ValueError('documentary source URL/role missing')
                datetime.fromisoformat(source['accessed_at'])
            if manifest is not None and evidence.get('protocol_sha256')!=manifest['bindings']['protocol_record']:
                raise ValueError('documentary protocol binding mismatch')
            for field in ('route_version_documented','request_schema_documented','parameter_support_documented'):
                if evidence.get(field) is not True:
                    raise ValueError('documented prerequisite required: '+field)
            settings=model.get('request_settings',{})
            if evidence.get('request_settings_by_effort',{}).get(model['effort'])!=settings:
                raise ValueError('request settings not bound to documentation')
            limit=evidence.get('token_limit_control',{})
            if cid=='J01':
                if settings or limit!={'kind':'native_choice','generated_text_limit_applicable':False} or evidence.get('decisions_mapping_documented') is not True:
                    raise ValueError('documented native Decisions mapping required; no decoding overrides')
                if not isinstance(model.get('provider_slug'),str) or not model['provider_slug'].strip():
                    raise ValueError('JEV OpenRouter provider_slug required')
            else:
                if (not token_limit_request_valid(config,model,limit) or set(settings)-adapters.ALLOWED_CONTROLS):
                    raise ValueError('documented total-token control and supported settings required')
                if model.get('pair_id'):
                    if (not native_effort_request_valid(model)
                            or evidence.get('native_effort_documented') is not True):
                        raise ValueError('documented native effort mapping required')
        except (OSError,ValueError,KeyError,TypeError,AttributeError) as exc:
            errors.append(cid+': '+str(exc))
    return errors
