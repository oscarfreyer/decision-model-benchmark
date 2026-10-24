"""Pure, namespace-aware reported-ID crosswalk. Never rewrites source artifacts."""
CURRENT_NAMESPACE = 'model-ids-20261011'
ORIGINAL_NAMESPACE = 'original-campaign'
LUNA_NAMESPACE = 'luna-extension-campaign'
DRAFT_NAMESPACE = 'decision-extension-draft-20261011'

_GENERATIVE = (
 ('C03','GPT-6.1 Sol','openai/gpt-6.1-sol','low'),
 ('C05','GPT-6 Luna','openai/gpt-6-luna','low'),
 ('C06','GPT-6 Luna','openai/gpt-6-luna','high'),
 ('C09','Claude Sonnet 5.5','anthropic/claude-sonnet-5.5','low'),
 ('C11','Gemini 3.8 Flash','google/gemini-3.8-flash','low'),
 ('C13','DeepSeek V4.1 Flash','deepseek/deepseek-v4.1-flash','low'),
 ('C14','DeepSeek V4.1 Flash','deepseek/deepseek-v4.1-flash','high'),
 ('C15','Qwen3.8 Max','qwen/qwen3.8-max-0902','low'),
 ('C17','Qwen3.8-27B','qwen/qwen3.8-27b','low'),
 ('C18','Qwen3.8-27B','qwen/qwen3.8-27b','xhigh'),
 ('C19','Mistral Medium 3.5','mistralai/mistral-medium-3-5','none'),
 ('C22','GLM-5','z-ai/glm-5','native'),
 ('C23','Kimi K2.5','moonshotai/kimi-k2.5','native'),
 ('C24','Llama 4 Maverick','meta-llama/llama-4-maverick','standard'),
 ('C25','Kimi K2.5','moonshotai/kimi-k2.5','instant'),
)
_DECISION = (
 ('JEV 1.13','typesafe/jev-1.13'),
 ('GPT-6 Luna Decisions','openai/gpt-6-luna-decisions'),
 ('Kev 4B','jaredpalmer/kev-4b'), ('Solar Decide','upstage/solar-decide'),
 ('Mercury Decide','inception/mercury-decide'), ('Clef','cloudflare/clef'),
 ('Clef Flash','cloudflare/clef-flash'), ('Liquid d1','liquid/d1'),
 ('Decider V1.1 27B','perplexity/pplx-decider-v1.1-27b'),
 ('Solar Decide Flash','upstage/solar-decide-flash'), ('Clef Omni','cloudflare/clef-omni'),
 ('Drex v1.5','nace-ai/drex-v1.5'), ('Microsoft-Decision-1','microsoft/microsoft-decision-1'),
)
CONFIGURATIONS = {f'G{i:02d}': dict(config_id=f'G{i:02d}', display_name=name, model_id=model, effort=effort)
                  for i, (_, name, model, effort) in enumerate(_GENERATIVE, 1)}
CONFIGURATIONS.update({f'D{i:02d}': dict(config_id=f'D{i:02d}', display_name=name, model_id=model, effort='native')
                       for i, (name, model) in enumerate(_DECISION, 1)})
MAPPINGS = {
 CURRENT_NAMESPACE: {cid: cid for cid in CONFIGURATIONS},
 ORIGINAL_NAMESPACE: {**{r[0]: f'G{i:02d}' for i, r in enumerate(_GENERATIVE, 1)}, 'J01': 'D01'},
 LUNA_NAMESPACE: {'D01': 'D02'},
 DRAFT_NAMESPACE: {f'D{i:02d}': f'D{i+1:02d}' for i in range(2, 13)},
}
EXTENSION_IDS = tuple(f'D{i:02d}' for i in range(3, 14))


def canonical_configuration(source_namespace, source_config_id, *, model_id=None, effort=None):
    try:
        record = dict(CONFIGURATIONS[MAPPINGS[source_namespace][source_config_id]])
    except KeyError:
        raise ValueError('unknown/ambiguous configuration namespace or ID') from None
    if model_id is not None and model_id != record['model_id']:
        raise ValueError('configuration model mismatch')
    if source_namespace == ORIGINAL_NAMESPACE and source_config_id == 'C25' and effort == 'none':
        effort = 'instant'  # Historical disabled-thinking control; preserve source value outside this view.
    if effort is not None and effort != record['effort']:
        raise ValueError('configuration native-setting mismatch')
    return record


def configuration_mapping(source_namespace, source_config_id, *, model_id=None, effort=None):
    record = canonical_configuration(source_namespace, source_config_id, model_id=model_id, effort=effort)
    return {**record, 'source_namespace': source_namespace, 'source_config_id': source_config_id,
            'current_config_id': record['config_id'], 'config_namespace': CURRENT_NAMESPACE}
