---
type: protocol
project_id: prj-202610-jev-medqa
status: draft
created: 2026-10-04
updated: 2026-10-12
version: "2.8"
---
# Study Protocol

**Version 2.8 · 12 October 2026**

## Objectives and study design

This study compares native decision models and generative language-model configurations on medical multiple-choice answer selection. The primary objective is to compare first-valid top-1 accuracy. Key descriptive comparisons relate accuracy to final-attempt cost and latency. Secondary analyses examine three native-effort contrasts and the confidence and repeatability of decision-model responses.

The study uses a closed-book, text-only design with 13 decision models and 15 generative configurations. Each generative configuration has one primary run. Each decision model has one primary run and two repeatability runs. Every run covers all 1,273 questions in the official English US four-option MedQA evaluation split. The unit of observation is a question/configuration/run outcome; each outcome can contain one initial attempt and up to ten failure-recovery attempts.

This protocol specifies model selection, inputs, request conditions, collection and provenance. The accompanying Statistical Analysis Plan, version 1.8 specifies scoring, statistical inference, resource analysis, confidence, repeatability and sensitivities.

## Dataset and input preparation

- Use the official [MedQA](https://github.com/jind11/MedQA) English US four-option data, comprising 10,178 training, 1,272 development and 1,273 evaluation questions. Keep the splits separate. Software and interface development use synthetic fixtures or development questions; development calls do not enter evaluation denominators.
- Identify the evaluation source by SHA-256 `c3b905ccfa66152dc25afbcb2c10e86c0bdb208824f0658fcdb2c040f60a2beb`. The dataset reproducibility record documents acquisition, split audits, source-row mapping and input checksums.
- Preserve every evaluation question's stem, four option labels A–D, option text and option order. Assign stable question IDs linked to the source file and row. Validate four labelled options and agreement between the supplied reference label and reference-answer text.
- Prepare solution-free inference inputs containing questions and options. Store reference keys and explanations separately; inference and retry selection do not load them. Join reference keys only in offline scoring.
- Retain all 1,273 questions and their original reference keys for primary analysis. The SAP defines the fixed 108-question missing-visual sensitivity, which leaves 1,165 questions. Do not repair keys, supply missing images or alter evaluation inputs in response to model performance.

## Model configurations

G01–G15 identify generative configurations and D01–D13 identify decision models. Each native-effort setting is a separate generative configuration. The three matched effort pairs are GPT-6 Luna Low/High, DeepSeek V4.1 Flash Low/High and Qwen3.8-27B Low/Xhigh. Other configurations are unpaired.

The roster covers selected hosted generative services and services exposing native bounded-choice outputs. Selection does not use the added models' evaluation accuracy. Local inference, free variants, automatic model selection and unselected configurations are excluded. Decision-model eligibility requires a native selected-label interface and support for four label-indexed option probabilities; generation of an option letter alone does not meet this definition.

| ID | Model | Native setting | OpenRouter request ID |
| --- | --- | --- | --- |
| G01 | GPT-6.1 Sol | Low | `openai/gpt-6.1-sol` |
| G02 | GPT-6 Luna | Low | `openai/gpt-6-luna` |
| G03 | GPT-6 Luna | High | `openai/gpt-6-luna` |
| G04 | Claude Sonnet 5.5 | Low | `anthropic/claude-sonnet-5.5` |
| G05 | Gemini 3.8 Flash | Low | `google/gemini-3.8-flash` |
| G06 | DeepSeek V4.1 Flash | Low | `deepseek/deepseek-v4.1-flash` |
| G07 | DeepSeek V4.1 Flash | High | `deepseek/deepseek-v4.1-flash` |
| G08 | Qwen3.8 Max | Low | `qwen/qwen3.8-max-0902` |
| G09 | Qwen3.8-27B | Low | `qwen/qwen3.8-27b` |
| G10 | Qwen3.8-27B | Xhigh | `qwen/qwen3.8-27b` |
| G11 | Mistral Medium 3.5 | None | `mistralai/mistral-medium-3-5` |
| G12 | GLM-5 | Native | `z-ai/glm-5` |
| G13 | Kimi K2.5 | Native | `moonshotai/kimi-k2.5` |
| G14 | Llama 4 Maverick | Standard | `meta-llama/llama-4-maverick` |
| G15 | Kimi K2.5 | Instant (thinking disabled) | `moonshotai/kimi-k2.5` |
| D01 | JEV 1.13 | Native | `typesafe/jev-1.13` |
| D02 | GPT-6 Luna Decisions | Native | `openai/gpt-6-luna-decisions` |
| D03 | Kev 4B | Native | `jaredpalmer/kev-4b` |
| D04 | Solar Decide | Native | `upstage/solar-decide` |
| D05 | Mercury Decide | Native | `inception/mercury-decide` |
| D06 | Clef | Native | `cloudflare/clef` |
| D07 | Clef Flash | Native | `cloudflare/clef-flash` |
| D08 | Liquid d1 | Native | `liquid/d1` |
| D09 | Decider V1.1 27B | Native | `perplexity/pplx-decider-v1.1-27b` |
| D10 | Solar Decide Flash | Native | `upstage/solar-decide-flash` |
| D11 | Clef Omni | Native | `cloudflare/clef-omni` |
| D12 | Drex v1.5 | Native | `nace-ai/drex-v1.5` |
| D13 | Microsoft-Decision-1 | Native | `microsoft/microsoft-decision-1` |

All configurations use OpenRouter with a fixed serving provider and disabled fallbacks. Each configuration record specifies the provider slug, endpoint, expected returned model/provider identity, native settings and available version metadata before evaluation. Requested gateway model IDs and returned identities are recorded separately from advertised backend snapshots. When backend version or serving geography cannot be verified, record it as unavailable. Model, provider, prompt, native settings and parsing rules remain fixed within each model's runs and retries.

## Requests and response handling

Each request contains one unchanged question stem and its four options, a bounded-choice instruction and empty conversation history. Retrieval, browsing, tools, native search, images, reference answers and explanations are excluded. Interface-specific schemas express the same answer-selection task. Complete request templates and effective controls are versioned with the collection source.

### Generative configurations

- Request a final JSON object with exactly one field, `answer`, whose value is one of the supplied labels A–D. Parse only the final response channel. Do not extract an answer from reasoning text, repair malformed output, request self-reported confidence or introduce an abstention option.
- Set `max_completion_tokens=16384`, with no simultaneous `max_tokens` override. Preserve each configuration's native effort/mode. Kimi Instant disables reasoning; Mistral Medium uses its None setting. Native and Standard configurations retain their documented controls.
- Within an effort pair, keep the model, route, prompt, parser, requested output allowance, supported sampling controls, timeout and scheduling policy identical apart from effort. Record supported seeds and sampling values, or documented defaults when explicit controls are unavailable. Requested effort labels and output allowances are reported separately from observed token usage and backend enforcement.

### Decision models

- Request a native selected option through the model's supported Decisions or System One interface. Preserve the supplied A–D labels and their order. Do not impose generative effort, sampling or text-decoding controls on native decision requests.
- Record the native selected label, the four label-indexed option values and every separate provider scalar. Keep JEV concentration, Luna API confidence and other provider scores in distinct fields. Do not replace the selected label with the largest vector value, infer absent probabilities or substitute a scalar for an option probability.
- A valid selected label remains eligible for accuracy when its probability vector is missing or ineligible. Confidence and probability-repeatability eligibility follow the SAP: four matching labels, finite values in [0,1], positive sum and inclusive absolute sum deviation at most 0.02. Preserve native values and precision; normalization is limited to distribution-comparison measures.

### Interface and input verification

Use development inputs to verify transport, fixed-provider routing, returned identities, supported controls, final-answer parsing, native field meaning, refusal/error handling, usage, billing and timing. Check the full request against the selected route's effective input limit, including the instruction and schema envelope. Do not truncate or shorten evaluation questions to fit a route. A deployment check uses one prepared development question per configuration with the evaluation request settings and saves the complete response.

An incompatible route, unsupported required control, identity drift or inability to preserve the complete input prevents collection for the affected configuration until resolved. Do not silently substitute a model or provider, weaken parsing, or remove a configuration from the analysis family. Document any changed configuration and its timing before subsequent evaluation. Evaluation correctness is not used to tune prompts, controls, routing or eligibility thresholds.

## Collection campaigns and run roles

Collection comprises the Original, Luna extension and Decision model extension campaigns. These names identify source provenance; all configurations follow the same analysis rules.

| Collection campaign | Configurations | Primary outcomes, E1 | Repeat outcomes, E2 | Total outcomes |
| --- | --- | ---: | ---: | ---: |
| Original | G01–G15 and D01 | 20,368 | 2,546 | 22,914 |
| Luna extension | D02 | 1,273 | 2,546 | 3,819 |
| Decision model extension | D03–D13 | 14,003 | 28,006 | 42,009 |
| **Total** | **28 configurations** | **35,644** | **33,098** | **68,742** |

Counts are question/configuration/run outcomes. Retry calls increase the number of attempts without increasing these denominators. E1 supplies run-1 accuracy, confidence and resource comparisons. E2 supplies runs 2–3 for each decision model. Repeat runs do not replace the primary answer, form an ensemble or add calibration observations.


The Original campaign began on 6 October 2026 and completed initial collection on 7 October. JEV repeat runs preceded the separately collected Luna runs. Supplementary failure recovery for the Original and Luna campaigns occurred on 7–8 October. The 11 additional decision models were selected on 11 October after access to the earlier outcomes. Report actual service dates and returned identities for every run; the campaigns do not represent simultaneous service conditions.

Within each campaign, all primary initial attempts reach terminal outcomes before secondary initial attempts begin. Primary failure recovery completes or exhausts its limits before secondary failure recovery. Each repeat uses fresh requests with the same model, prompt, route and native controls. Decision-model repeats introduce no seed overrides.

## Scheduling and collection parameters

Use the settings associated with the actual source campaign and continuation segment. Historical segments retain their recorded runtime; the current settings govern the Decision model extension. The same provider's model configurations share concurrency and start-spacing limits.

### Recorded settings for completed collection

| Campaign and segment | Global workers | Provider concurrency | Minimum start spacing | Response to three consecutive HTTP 429 responses |
| --- | ---: | --- | --- | --- |
| Original: `main`, `main-v2` | 20 | 20 per provider, including SiliconFlow and InferenceNet | InferenceNet 2 s; Mistral 2 s; no additional spacing for other providers | Stop dispatch globally and drain active calls |
| Original: `main-v3`, `main-v4` | 20 | 20 per provider; InferenceNet 1 | InferenceNet 10 s; Mistral 2 s; no additional spacing for other providers | Pause the affected provider; others continue |
| Original: `main-v5` | 20 | 20 per provider; InferenceNet 10 | InferenceNet 5 s; Mistral 2 s; no additional spacing for other providers | Pause the affected provider; others continue |
| Luna extension: `luna-main-20261007` | 1 | OpenAI 1 | OpenAI 1 s | Pause OpenAI |
| Original and Luna failure recovery: `failure-retries-20261007` | 10 shared | 10 per provider | OpenAI 5 s; InferenceNet 5 s; Mistral 2 s; no additional spacing for other providers | Pause the affected provider; others continue |

All completed segments used a 600-second request timeout and disabled SDK/transport retries and provider fallbacks. Initial collection used one attempt per question/run, round-robin scheduling and question-order seed 20261006. Failure recovery added up to ten attempts per failed identity, stopped at the first valid answer and followed the original seeded item order with earlier retry rounds first. Retry-After paced subsequent starts. Rate-limit stops and pauses required reviewed continuation; operator stops drained active calls, and unresolved starts or identity/schema drift prevented further dispatch. Initial Original-campaign generative requests used a 16,384-token completion allowance; native decision calls had no generative token allowance. Exact source packages and segment dates are recorded in the run provenance.

### Current settings for the Decision model extension

| Setting               | Value                                                                                                                                                                      |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Global concurrency    | Ten shared workers across models, providers and native interfaces                                                                                                          |
| Provider concurrency  | Ten concurrent calls per provider, shared across that provider's models                                                                                                    |
| Request timeout       | 600 seconds                                                                                                                                                                |
| Minimum start spacing | Five seconds between request starts at each provider, shared across that provider's models                                                                                                                                                               |
| Scheduling            | Round-robin model scheduling; question-order seed 20261006; retry targets follow original seeded item order and earlier retry rounds first                                 |
| Rate-limit pause      | Pause the affected provider after three consecutive HTTP 429 responses; unaffected providers continue                                                                      |
| Retry-After           | Delay subsequent provider starts according to the header and configured minimum spacing                                                                                    |
| Stop and resume       | Drain active calls on an operator stop; a provider pause remains latched until reviewed continuation; identity/schema drift or indeterminate starts block further dispatch |
| Attempt limit         | One initial attempt plus at most ten persisted retry starts per question/configuration/run; at most one active call for that identity                                      |
| Transport behavior    | One call per persisted attempt; SDK/transport retries and provider fallbacks disabled                                                                                      |

## Failure recovery, stopping and completeness

- Make one initial attempt for each question/configuration/run identity. A failed or invalid response permits up to ten additional persisted starts, stopping at the first parser-valid answer. A valid wrong answer is final because reference correctness is unavailable to the collector.
- Eligible failures include refusals, truncations, malformed answers, HTTP errors and resolved transport/timeout failures. Preserve the request payload, route and parser across retries. Missing billing or probability fields do not trigger another call when the selected answer is valid.
- Each persisted retry start consumes one of the ten allowed retries, including across restarts. Permit at most one active call per original identity. Record the original-query link and retry index, and reject duplicate starts or a second recovery campaign claiming the same identity.
- Resume skips completed attempts. An unresolved start is not automatically resent; reconcile it before further dispatch. Operator stops drain active calls and save their outcomes. Provider pauses remain latched until reviewed continuation, while unaffected providers can continue eligible work. Identity or schema drift stops affected collection for review.
- Score an exhausted failure as incorrect under the SAP. An unstarted or indeterminate identity remains missing and prevents a complete-study analysis. Verify the full 68,742-identity inventory, item/key alignment, primary/secondary roles, unique source packages and complete parent/retry lineage before combining campaigns.

## Outcomes and analysis overview

Primary accuracy uses one selected final outcome per question: the valid initial response, first valid retry or exhausted-failure outcome. Every primary configuration has denominator 1,273. The SAP specifies Wilson intervals, paired effects, resampling and sensitivity analyses.

The primary S1 family contains 273 contrasts: 195 decision-versus-generative comparisons and 78 unordered decision-model comparisons, with one Holm correction across the complete family. The three exploratory native-effort contrasts form a separate S2 Holm family. Report the full families independently of accuracy ranking or statistical significance.

### Cost and latency

Use the same final attempt for cost and latency: the valid initial response, first valid retry or last terminal attempt for an exhausted failure. Exclude earlier unsuccessful charges and durations from primary resource summaries and frontiers. Attribute each selected attempt to its actual provider, campaign, segment and execution conditions. Mean cost uses 1,273 primary question outcomes; the latency frontier coordinate is median final-attempt request-wall duration. Queueing, backoff, pauses and collection windows are reported separately.

Apply the SAP's common primary-run imputation rules to every model. Missing final-attempt cost or duration uses the unrounded mean of observed values from the corresponding selected-final-attempt donor pool within that configuration's primary run. Include observed zeros and exhausted final failures; exclude earlier attempts, other models, repeat runs and imputed donors. With no donors, leave the estimate unavailable. Missing resource values in repeat runs remain unavailable. Keep raw missingness and assignments separate and report observed-only summaries. Do not impute answers, option probabilities or provider confidence fields.

### Confidence and repeatability

Analyze confidence using each decision model's eligible run-1 probabilities. Analyze answer and probability repeatability across runs 1–3 for all D01–D13, using the same measures and eligibility rules, with model-specific denominators. Incorrect but valid answers remain included.

Answer agreement uses valid pairs or triples. Probability comparisons use their own eligible pairs or triples and report excluded counts. Additional provider-specific scalar fields are retained as raw evidence and excluded from the analysis. Accuracy and correctness/failure transitions retain all 1,273 questions, including exhausted failures. E2 adds no calibration observations. The SAP defines confidence measures, distribution comparisons, all denominators and the 12 October 2026 post hoc removal of provider-scalar analyses.

## Data integrity and reproducibility

- Save raw requests and responses, final/reasoning channels, provider request IDs, requested and returned model/provider identities, effective controls, finish/failure categories, native probabilities/scalars, token usage, billing, timestamps, request durations and missing-value reasons.
- Link each attempt to its benchmark, question, configuration, campaign, run, primary/secondary role and retry index. Preserve source IDs alongside the reported G/D IDs using the configuration-ID crosswalk. Validate the source namespace before joining records.
- Bind each collection package to its actual source code, request configuration, prepared input checksums and execution environment. Keep reference keys separate from inference inputs. A continuation records its parent and changed execution settings without rewriting earlier attempts.
- Bind the final numerical results to the protocol/SAP versions, complete analysis source/settings/environment and input checksums. Bind exported tables and figures to those saved numerical results. Report amendments with their dates and whether they followed access to evaluation outcomes.
- Provide the preparation, collection and analysis code, dependency specifications, reproduction instructions and permitted question-level data with the publication. Exclude credentials from all releases. Preserve source-linked prior versions and collection artifacts so reported results can be reconstructed.

## Registration, data use and reporting

The study is unregistered. Core accuracy, Wilson, McNemar, Holm and bootstrap methods were documented before Original-campaign collection. Failure recovery, the first-valid primary analysis, the larger decision-model roster and comparison family, reporting additions, missing-visual sensitivity and resource-imputation rules followed access to earlier outcomes. This protocol and the SAP describe the resulting study, with collection and amendment chronology in the run provenance.

The study analyzes existing benchmark material without recruiting participants or collecting new patient data. Dataset attribution and the repository's MIT copyright/permission notice accompany distributed benchmark material. Disclose funding and relationships with model providers alongside the manuscript.

Report the selected services, gateway IDs, native settings, serving providers, collection dates, failures, missingness and analysis exclusions. Generative configurations have one realized primary response per question, so their run-to-run variability is not estimated. Differences in serving dates, provider load and scheduling conditions accompany resource comparisons. The public evaluation split may have appeared in model training; the dataset audit does not establish the absence of training contamination.