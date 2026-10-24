"""Native bounded-choice wire contract, shared by Decisions and System One.

The documented envelopes match; each model/route must still pass development
verification. No fallback surface, generative controls or probability repair.
"""
import base64
from decimal import Decimal
import json
import os
import time
import urllib.error
import urllib.request
from benchmarking import adapters
from benchmarking.common import empty_probability, measurement, native_probability

ENDPOINTS = {
    'decisions': 'https://openrouter.ai/api/alpha/decisions',
    'systemone': 'https://openrouter.ai/api/v1/systemone',
}
SCHEMA = 'openrouter-native-choice-v1'


def validate_model(model):
    if model.get('surface') not in ENDPOINTS or model.get('endpoint') != ENDPOINTS[model['surface']]:
        raise ValueError('unsupported native surface/endpoint')
    if model.get('response_schema') != SCHEMA:
        raise ValueError('unverified response schema: implement and review an explicit adapter')
    if model.get('request_settings'):
        raise ValueError('native requests cannot contain generative controls')
    for key in ('model_id', 'provider_slug', 'route', 'expected_returned_model'):
        if not isinstance(model.get(key), str) or not model[key].strip():
            raise ValueError('native route requires '+key)


def build_request(question, model, prompt):
    validate_model(model)
    question.validate()
    from benchmarking.extension_contract import request_payload
    return request_payload(question, model, prompt)


def scalar(value, semantics):
    if value is None:
        return {**measurement(), 'semantics': semantics}
    try:
        if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
            raise ValueError()
        if not Decimal(str(value)).is_finite():
            raise ValueError()
    except (ArithmeticError, ValueError):
        return {**measurement(reason='nonfinite_or_invalid'), 'semantics': semantics}
    return {**measurement(str(value)), 'semantics': semantics}


def parse(raw, model, labels):
    """Preserve exact wire values; a bad vector never invalidates a valid label."""
    result = {'status': 'invalid', 'selected_label': None, 'failure_reason': 'malformed_response',
              'probability': empty_probability(), 'provider_scalars': {}, 'usage': {},
              'observed_usd': None, 'returned_model': None, 'returned_route': None,
              'provider_request_id': raw.get('headers', {}).get('x-request-id'),
              'effective_controls': None, 'finish_reason': None, 'final': None, 'reasoning': None}
    semantics = model.get('scalar_semantics', {})
    result['provider_scalars'] = {path: scalar(None, meaning) for path, meaning in semantics.items()}
    if raw.get('transport_error'):
        result.update(status='timeout' if raw['transport_error'] == 'TimeoutError' else 'transport_error', failure_reason=raw['transport_error'])
        return result
    # Non-JSON HTTP failures still participate in persisted 429/Retry-After policy.
    if not 200 <= raw['status'] < 300:
        result.update(status='http_error', failure_reason='provider_http_error')
    try:
        _, body = adapters.decode(raw)
        if not isinstance(body, dict):
            return result
        result.update(returned_model=body.get('model'), returned_route=body.get('provider'),
                      provider_request_id=body.get('id', result['provider_request_id']),
                      effective_controls=body.get('effective_controls'), finish_reason=body.get('finish_reason'))
        usage = body.get('usage') or {}
        if not isinstance(usage, dict):
            result.update(status='drift', failure_reason='usage_schema_changed')
            return result
        result['usage'] = usage
        result['observed_usd'] = usage.get('cost')
        if result['status'] == 'http_error' or body.get('error'):
            result.update(status='http_error', failure_reason='provider_http_or_error_response')
            return result
        if (body.get('model') != model['expected_returned_model'] or body.get('provider') != model['route']):
            result.update(status='drift', failure_reason='returned_model_or_provider_mismatch')
            return result
        if body.get('effective_controls') not in (None, {}):
            result.update(status='drift', failure_reason='unexpected_effective_controls')
            return result
        if any(body.get(k) for k in ('choices', 'messages', 'tool_calls', 'function_call', 'reasoning')):
            result.update(status='drift', failure_reason='unexpected_generated_or_tool_output')
            return result
        answers = body.get('answers')
        if not isinstance(answers, dict) or set(answers) != {'answer'}:
            result.update(status='drift', failure_reason='native_answer_envelope_changed')
            return result
        answer = answers['answer']
        result['final'] = adapters.json_text(answer)
        if not isinstance(answer, dict):
            result.update(status='drift', failure_reason='native_answer_schema_changed')
            return result
        # Capture all numeric provider scalars recursively, outside probabilities and usage.
        # The complete response also preserves nonnumeric/unknown fields verbatim.
        def collect(value, path):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key not in ('probabilities', 'usage'):
                        collect(child, path+'.'+key if path else key)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    collect(child, path+'.'+str(index))
            elif path in semantics or isinstance(value, (int, Decimal)) and not isinstance(value, bool):
                result['provider_scalars'][path] = scalar(value, semantics.get(path, 'unreviewed'))
        collect(body, '')
        if answer.get('type') == 'refusal':
            result.update(status='refused', failure_reason='native_refusal')
            return result
        if answer.get('type') != 'choice':
            result.update(status='drift', failure_reason='native_answer_type_changed')
            return result
        label = answer.get('choice')
        if not isinstance(label, str) or label not in labels:
            return result
        vector = answer.get('probabilities')
        if isinstance(vector, dict):
            vector = {k: str(v) if isinstance(v, (int, Decimal)) and not isinstance(v, bool) else v
                      for k, v in vector.items()}
        result['probability'] = native_probability(vector, label, labels,
            semantics=model.get('probability_semantics', {}).get('meaning', 'unreviewed'),
            sum_tolerance='0.02', decisions=True)
        result.update(status='valid', selected_label=label, failure_reason=None)
    except (ValueError, TypeError, KeyError, AttributeError, UnicodeError):
        pass
    return result


def dispatch(request, model, timeout):
    """One POST with a total response deadline, no redirects/retries/fallbacks."""
    validate_model(model)
    key = os.environ.get('OPENROUTER_API_KEY')
    if not key:
        raise ValueError('OPENROUTER_API_KEY unavailable')
    req = urllib.request.Request(model['endpoint'], data=json.dumps(request,ensure_ascii=False).encode(),
        headers={'Content-Type':'application/json','Authorization':'Bearer '+key},method='POST')
    opener = urllib.request.build_opener(adapters.NoRedirect())
    deadline = time.monotonic()+timeout
    try:
        response = opener.open(req,timeout=timeout)
    except urllib.error.HTTPError as exc:
        response = exc
    chunks = []; failure = None
    with response:
        try:
            while True:
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError()
                # CPython HTTPResponse.read1 performs at most one socket read.
                raw_response = response.fp if isinstance(response,urllib.error.HTTPError) else response
                if raw_response.isclosed():
                    break
                sock = raw_response.fp.raw._sock
                sock.settimeout(remaining)
                chunk = raw_response.read1(65536)
                if not chunk:
                    if raw_response.length not in (None, 0):
                        failure = 'IncompleteRead'
                    break
                chunks.append(chunk)
        except TimeoutError:
            failure = 'TimeoutError'
        except (OSError, adapters.IncompleteRead) as exc:
            if isinstance(exc,adapters.IncompleteRead):
                chunks.append(exc.partial)
            failure = type(exc).__name__
        result = {'status':response.status,
                  'body_base64':base64.b64encode(b''.join(chunks)).decode('ascii'),
                  'headers':{k.lower():v for k,v in response.headers.items()
                             if k.lower() in ('x-request-id','retry-after')}}
        if failure:
            result['transport_error'] = failure
        return result
