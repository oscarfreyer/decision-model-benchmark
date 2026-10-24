import base64
from decimal import Decimal
from http.client import IncompleteRead
import json
import os
import socket
import urllib.error
import urllib.request
from benchmarking.common import (strict_json, native_probability, empty_probability, measurement,
    OPENROUTER_ENDPOINTS, TYPESAFE_ENDPOINT, JEV_DECISIONS_ENDPOINT, validate_transport)
ALLOWED_CONTROLS = {'max_tokens','max_completion_tokens','temperature','top_p','seed','reasoning','verbosity'}

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl): return None

def build_request(question, model, prompt, *, mock=False):
    validate_transport(model)
    if model['adapter'] in {'jev','jev_gateway','openai_decisions_gateway'}:
        if model.get('request_settings'): raise ValueError('native Choice decoding overrides unverified')
        request={'model': model['model_id'] if model['adapter'] in {'jev_gateway','openai_decisions_gateway'} else model['version'] or model['model_id'], 'state':prompt,
            'questions':{'answer':{'type':'choice','instructions':'Select exactly one supplied option label.',
                'criteria':{o.label:o.text for o in question.options}}}}
        if model['adapter'] in {'jev_gateway','openai_decisions_gateway'}:
            slug=model.get('provider_slug')
            if not isinstance(slug,str) or not slug.strip(): raise ValueError('Decisions OpenRouter provider_slug required')
            request['provider']={'order':[slug],'only':[slug],'allow_fallbacks':False,'require_parameters':True}
        return request
    if model['adapter']!='openrouter': raise ValueError('unsupported adapter; local execution excluded')
    controls=model.get('request_settings',{})
    if set(controls)-ALLOWED_CONTROLS: raise ValueError('unsupported controls: '+str(sorted(set(controls)-ALLOWED_CONTROLS)))
    route=model['route'] or ('fictional-provider' if mock else '')
    if not route: raise ValueError('exact provider endpoint route required')
    slug=model.get('provider_slug',route)
    if not isinstance(slug,str) or not slug.strip(): raise ValueError('OpenRouter provider_slug required')
    provider={'order':[slug],'only':[slug],'allow_fallbacks':False,'require_parameters':True}
    ignored=model.get('provider_ignore',[])
    if not isinstance(ignored,list) or any(not isinstance(x,str) or not x.strip() for x in ignored):
        raise ValueError('provider_ignore must contain endpoint slugs')
    if ignored: provider['ignore']=ignored
    schema={'type':'object','properties':{'answer':{'type':'string','enum':list(question.labels)}},'required':['answer'],'additionalProperties':False}
    return {'model':model['model_id'],'messages':[{'role':'user','content':prompt}], 'stream':False,
        'provider':provider,
        'response_format':{'type':'json_schema','json_schema':{'name':'option_answer','strict':True,'schema':schema}},
        **controls}

def dispatch(request, model, timeout):
    """One HTTP request; no redirects, SDK retries, provider fallback or auxiliary calls."""
    validate_transport(model)
    direct=model['adapter']=='jev'; endpoint=model['endpoint']
    key=os.environ.get('TYPESAFE_API_KEY' if direct else 'OPENROUTER_API_KEY')
    if not key: raise ValueError('required credential unavailable')
    req=urllib.request.Request(endpoint,data=json.dumps(request,ensure_ascii=False).encode(),
        headers={'Content-Type':'application/json','Authorization':'Bearer '+key},method='POST')
    opener=urllib.request.build_opener(NoRedirect())
    try: response=opener.open(req,timeout=timeout)
    except urllib.error.HTTPError as exc: response=exc
    with response:
        interrupted=None
        try: raw=response.read()
        except IncompleteRead as exc: raw=exc.partial;interrupted='IncompleteRead'
        result={'status':response.status,'body_base64':base64.b64encode(raw).decode('ascii'),
            'headers':{k.lower():v for k,v in response.headers.items() if k.lower() in ('x-request-id','retry-after')}}
        if interrupted: result['transport_error']=interrupted
        return result

def decode(raw):
    text=base64.b64decode(raw['body_base64'],validate=True).decode('utf-8')
    body=json.loads(text,parse_float=Decimal,parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite JSON')),object_pairs_hook=_unique_pairs)
    return text,body

def _unique_pairs(items):
    d={}
    for k,v in items:
        if k in d: raise ValueError('duplicate provider field')
        d[k]=v
    return d

def json_text(value): return json.dumps(value,default=str,ensure_ascii=False,allow_nan=False)

def parse(raw, model, labels):
    result={'status':'invalid','selected_label':None,'failure_reason':'malformed_response','final':None,'reasoning':None,
        'probability':empty_probability(),'returned_model':None,'returned_route':None,'provider_request_id':raw.get('headers',{}).get('x-request-id'),
        'effective_controls':None,'finish_reason':None,'usage':{},'observed_usd':None}
    decisions=model['adapter']=='openai_decisions_gateway'
    if decisions: result['provider_confidence']=measurement()
    if raw.get('transport_error')=='IncompleteRead':
        result.update(status='transport_error',failure_reason='IncompleteRead');return result
    try:
        text,body=decode(raw)
        result['returned_model']=body.get('model');result['returned_route']=body.get('provider')
        result['provider_request_id']=body.get('id',result['provider_request_id'])
        result['usage']=body.get('usage',{})
        if result['usage'] is None: result['usage']={}
        if not isinstance(result['usage'],dict):
            result.update(status='drift',failure_reason='provider_usage_schema_mismatch',usage={});return result
        result['observed_usd']=result['usage'].get('cost')
        result['effective_controls']=body.get('effective_controls') # May be unavailable; evidence is separately reviewed.
        if not 200 <= raw['status'] < 300 or body.get('error'):
            result.update(status='http_error',failure_reason='provider_http_or_error_response');return result
        expected=model['version'] or model['model_id']
        if body.get('model')!=expected:
            result.update(status='drift',failure_reason='returned_model_version_mismatch');return result
        if model['adapter']!='jev' and body.get('provider') != (model['route'] or 'fictional-provider'):
            result.update(status='drift',failure_reason='returned_route_unavailable_or_mismatch');return result
        details=result['usage'].get('completion_tokens_details')
        if decisions and details is None: details=result['usage'].get('output_tokens_details')
        if details is not None and not isinstance(details,dict):
            result.update(status='drift',failure_reason='provider_usage_details_schema_mismatch');return result
        if model['adapter'] in {'jev','jev_gateway','openai_decisions_gateway'}:
            if model['adapter']=='jev': result['returned_route']='typesafe-direct'
            if model['adapter'] in {'jev_gateway','openai_decisions_gateway'} and any(body.get(k) for k in ('choices','messages','tool_calls','function_call','reasoning')):
                result.update(status='drift',failure_reason='unexpected_generated_or_tool_output');return result
            if not isinstance(body.get('answers'),dict) or set(body['answers'])!={'answer'}: return result
            answer=body['answers']['answer'];result['final']=json_text(answer)
            if decisions and isinstance(answer,dict) and answer.get('type')=='refusal':
                result.update(status='refused',failure_reason='provider_refusal');return result
            if not isinstance(answer,dict) or answer.get('type')!='choice': return result
            label=answer.get('choice')
            if label not in labels: return result
            vector=answer.get('probabilities')
            if isinstance(vector,dict):
                vector={k:(str(v) if not isinstance(v,bool) and isinstance(v,(Decimal,int)) else v) for k,v in vector.items()} if decisions else {k:str(v) for k,v in vector.items()}
            concentration=answer.get('confidence')
            if decisions:
                scalar=concentration
                try:
                    if isinstance(scalar,bool) or not isinstance(scalar,(str,int,Decimal)) or not Decimal(str(scalar)).is_finite(): raise ValueError()
                except (ValueError,TypeError,ArithmeticError):
                    result['provider_confidence']=measurement(reason='provider_unavailable' if scalar is None else 'nonfinite_or_invalid')
                else: result['provider_confidence']=measurement(str(scalar))
                concentration=None
            if concentration is not None: concentration=str(concentration)
            result['probability']=native_probability(vector,label,labels,concentration,model.get('probability_semantics','unverified'),
                sum_tolerance=model.get('sum_tolerance'),decisions=True) if decisions else native_probability(vector,label,labels,concentration,model.get('probability_semantics','unverified'))
            result.update(status='valid',selected_label=label,failure_reason=None);return result
        choices=body.get('choices')
        if not isinstance(choices,list) or len(choices)!=1: return result
        choice=choices[0];message=choice['message'];finish=choice.get('finish_reason');result['finish_reason']=finish
        result['final']=message.get('content'); result['reasoning']=message.get('reasoning',message.get('reasoning_details'))
        if finish in ('tool_calls','function_call') or message.get('tool_calls') or message.get('function_call'):
            result.update(status='drift',failure_reason='unexpected_tool_call');return result
        if message.get('refusal'):
            result.update(status='refused',failure_reason='provider_refusal');return result
        if finish in ('length','max_tokens'):
            result.update(status='truncated',failure_reason='generated_token_limit');return result
        if finish=='content_filter':
            result.update(status='refused',failure_reason='content_filter');return result
        if finish!='stop': return result
        final=strict_json(result['final'])
        if not isinstance(final,dict) or set(final)!={'answer'} or final['answer'] not in labels: return result
        result.update(status='valid',selected_label=final['answer'],failure_reason=None)
    except (ValueError,KeyError,TypeError,AttributeError,UnicodeError): pass
    return result

def usage_record(usage, model):
    def count(value):
        return measurement(value) if type(value) is int and value>=0 else measurement(reason='provider_count_unavailable_or_invalid')
    input_count=usage.get('prompt_tokens',usage.get('input_tokens'))
    output=usage.get('completion_tokens',usage.get('output_tokens'))
    details=usage.get('completion_tokens_details',{}) or {}
    if model.get('adapter')=='openai_decisions_gateway' and not details: details=usage.get('output_tokens_details',{}) or {}
    if not isinstance(details,dict): details={}
    reasoning=details.get('reasoning_tokens',usage.get('reasoning_tokens'))
    final=usage.get('final_tokens')
    accounting=model.get('reasoning_in_output')
    flag=accounting if type(accounting) is bool else None
    # No subtraction unless endpoint accounting is explicitly verified.
    if final is None and flag is True and type(output) is int and type(reasoning) is int and 0<=reasoning<=output: final=output-reasoning
    return {'input_tokens':count(input_count),'output_tokens':count(output),'reasoning_tokens':count(reasoning),
        'final_tokens':count(final),'billed_tokens':count(usage.get('billed_tokens')),'reasoning_in_output':flag,
        'accounting_definition':model.get('output_accounting') or 'unverified; provider totals retained, no inferred billing'}

def billing_record(parsed, usage, model):
    observed=parsed.get('observed_usd');observed_reason='provider_charge_unavailable'
    if observed is not None:
        try:
            if isinstance(observed,bool) or not isinstance(observed,(str,int,float,Decimal)) or not Decimal(str(observed)).is_finite() or Decimal(str(observed))<0: raise ValueError()
        except (ValueError,TypeError,ArithmeticError): observed=None;observed_reason='provider_charge_invalid'
    estimate=None; pricing=model.get('pricing',{})
    i=usage['input_tokens']['value'];o=usage['output_tokens']['value'];r=usage['reasoning_tokens']['value']
    if i is not None and o is not None and usage['reasoning_in_output'] is not None and pricing.get('source') and pricing.get('as_of'):
        billed=o if usage['reasoning_in_output'] else (o+r if r is not None else None)
        if billed is not None:
            estimate=(Decimal(i)*Decimal(str(pricing['input_per_million']))+Decimal(billed)*Decimal(str(pricing['output_per_million'])))/1000000
    return {'observed_usd':measurement(str(observed) if observed is not None else None,observed_reason),
        'estimated_usd':measurement(str(estimate) if estimate is not None else None,'rates_or_accounting_unverified'),
        'currency':'USD','rate_provenance':pricing.get('source'),'rate_date':pricing.get('as_of')}
