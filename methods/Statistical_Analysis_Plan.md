---
type: protocol
project_id: prj-202610-jev-medqa
status: draft
version: "1.8"
created: 2026-10-08
updated: 2026-10-12
---
# Statistical Analysis Plan

**Version 1.8 · 12 October 2026**

## Study design and collection campaigns

The study compares 13 decision models and 15 generative configurations on all 1,273 official English US four-option MedQA evaluation questions in a closed-book, text-only setting. Each generative configuration has one primary run. Each decision model has three runs: one primary run and two repeatability runs. A run consists of one initial attempt per question and up to ten additional attempts for an invalid or failed response, stopping at the first valid answer.

Collection comprises three campaigns. Campaign names identify provenance; all configurations follow the same analysis rules.

| Collection campaign | Configurations | Primary outcomes, E1 | Repeat outcomes, E2 | Total outcomes |
| --- | --- | ---: | ---: | ---: |
| Original | G01–G15 and D01 | 20,368 | 2,546 | 22,914 |
| Luna extension | D02 | 1,273 | 2,546 | 3,819 |
| Decision model extension | D03–D13 | 14,003 | 28,006 | 42,009 |
| **Total** | **28 configurations** | **35,644** | **33,098** | **68,742** |

Counts refer to question/configuration/run outcomes. Retries increase the number of calls without increasing these denominators. Development calls are excluded. Run 1 supplies primary accuracy, confidence and resource comparisons. Runs 1–3 supply within-model repeatability; repeats do not replace the primary response or add calibration observations.

## Collection and failure recovery

Each request contains the unchanged question stem and four labelled options, with no reference key, explanation, retrieval, tools or conversation history. Generative configurations return a strict final JSON option label. Decision models return a native selected label and label-indexed probabilities. Model, provider, prompt, native settings and parsing rules remain fixed across a model's three runs and retries.

The tables below distinguish recorded settings for completed collection from the current settings for the Decision model extension. Resource summaries use the actual campaign and segment settings attached to each selected final attempt.

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

| Setting | Value |
| --- | --- |
| Global concurrency | Ten shared workers across models, providers and native interfaces |
| Provider concurrency | Ten concurrent calls per provider, shared across that provider's models |
| Request timeout | 600 seconds |
| Minimum start spacing | Five seconds between request starts at each provider, shared across that provider's models |
| Scheduling | Round-robin model scheduling; question-order seed 20261006; retry targets follow original seeded item order and earlier retry rounds first |
| Rate-limit pause | Pause the affected provider after three consecutive HTTP 429 responses; unaffected providers continue |
| Retry-After | Delay subsequent provider starts according to the header and configured minimum spacing |
| Stop and resume | Drain active calls on an operator stop; a provider pause remains latched until reviewed continuation; identity/schema drift or indeterminate starts block further dispatch |
| Attempt limit | One initial attempt plus at most ten persisted retry starts per question/configuration/run; at most one active call for that identity |
| Transport behavior | One call per persisted attempt; SDK/transport retries and provider fallbacks disabled |

Only failed or invalid attempts are eligible for another attempt. Refusals, truncations, malformed outputs and resolved HTTP/transport failures follow the same ten-retry limit. Selection stops at the first parser-valid response without consulting correctness. A valid wrong answer is final. Missing billing or probability fields do not trigger another call when the selected answer is valid. Resuming a campaign does not reset attempt counts.

Within each collection campaign, all primary initial attempts reach a terminal outcome before secondary initial attempts begin. Primary failure recovery completes or exhausts its retry limits before secondary failure recovery. Initial and retry calls retain separate attempt records linked to the same question/configuration/run outcome.

## Outcomes and analysis hierarchy

- **Primary outcome:** first-valid top-1 accuracy on all 1,273 questions, with Wilson 95% intervals and the complete 273-contrast decision-model McNemar/Holm family.
- **Key descriptive comparisons:** final-attempt cost and final-attempt request duration in relation to primary accuracy, with missingness, imputation and point-estimate nondominance reported explicitly.
- **Secondary analyses:** native-effort contrasts, decision-model confidence and repeatability.
- **Sensitivity analyses:** original first-attempt accuracy, paired complete-valid comparisons and the fixed visual-nomination exclusion. Report all performed analyses, including negative findings, with their specified denominators and comparison families.

## Analysis population and scoring

- 1,273 official English US four-option MedQA evaluation questions and their original reference keys.
- Select the valid initial response or earliest parser-valid retry without using reference keys. Compare the selected label with the reference label for top-1 correctness.
- Score exhausted failures as incorrect, including refusals, truncations, invalid outputs and exhausted HTTP/transport failures. Unstarted or indeterminate identities remain missing and prevent complete-study inference.
- Verify 68,742 unique question/configuration/run identities, complete item/key alignment and absence of duplicate source packages. Distinguish original and selected-output failure categories, valid-output rates, retry counts and exhausted failures.
- **Primary estimand:** correctness delivered per benchmark question by the evaluated configuration and service under the recorded request, parsing and retry conditions.

## Configuration key

G01–G15 identify generative configurations; D01–D13 identify decision models. Native effort settings define separate generative configurations. Tables and figures identify models by name and setting, with IDs linking their records.

| ID | Model | Native setting |
| --- | --- | --- |
| G01 | GPT-6.1 Sol | Low |
| G02 | GPT-6 Luna | Low |
| G03 | GPT-6 Luna | High |
| G04 | Claude Sonnet 5.5 | Low |
| G05 | Gemini 3.8 Flash | Low |
| G06 | DeepSeek V4.1 Flash | Low |
| G07 | DeepSeek V4.1 Flash | High |
| G08 | Qwen3.8 Max | Low |
| G09 | Qwen3.8-27B | Low |
| G10 | Qwen3.8-27B | Xhigh |
| G11 | Mistral Medium 3.5 | None |
| G12 | GLM-5 | Native |
| G13 | Kimi K2.5 | Native |
| G14 | Llama 4 Maverick | Standard |
| G15 | Kimi K2.5 | Instant (thinking disabled) |
| D01 | JEV 1.13 | Native |
| D02 | GPT-6 Luna Decisions | Native |
| D03 | Kev 4B | Native |
| D04 | Solar Decide | Native |
| D05 | Mercury Decide | Native |
| D06 | Clef | Native |
| D07 | Clef Flash | Native |
| D08 | Liquid d1 | Native |
| D09 | Decider V1.1 27B | Native |
| D10 | Solar Decide Flash | Native |
| D11 | Clef Omni | Native |
| D12 | Drex v1.5 | Native |
| D13 | Microsoft-Decision-1 | Native |

## Accuracy and paired inference

- Report correct/n, accuracy and Wilson 95% intervals for each configuration. Primary n=1,273.
- For each contrast, define b as first configuration correct/second incorrect and c as the reverse. Report all four paired contingency cells, n and `100 × (b−c)/n` percentage points. Positive differences favor the first named configuration.
- Use two-sided exact conditional McNemar for equal marginal accuracy: conditional on b+c, test b against Binomial(b+c, 0.5). With no discordance, report P=1.
- **S1:** each of 13 decision models minus each of 15 generative configurations (195 contrasts), plus all unordered decision-model pairs (78 contrasts). Total: 273 distinct run-1 contrasts. For decision-model pairs, subtract the later model from the earlier model in the fixed order D01, D02, …, D13. Enumerate every pair once, independently of accuracy or ranking.
- **Exploratory S2:** Luna High−Low (G03−G02), DeepSeek High−Low (G07−G06), Qwen 27B Xhigh−Low (G10−G09). Total: three accuracy contrasts; resource differences remain descriptive.
- Apply Holm separately to all 273 S1 P values and all three S2 P values, including nonsignificant comparisons, with α=0.05 in each family. Report raw/adjusted P values, effect direction and intervals. Statistical rejection follows adjusted P≤0.05.
- S1 addresses decision-model comparisons; S2 addresses exploratory reasoning-effort comparisons. Report family-specific error control under [Holm (1979)](https://doi.org/10.2307/4615733).
- Compute S1 over the complete 273-contrast family. Incomplete roster coverage prevents a complete-family result. Generative-versus-generative comparisons enter only the three specified S2 contrasts.
- Treat all displayed 95% intervals as pointwise. Nonsignificance does not establish equivalence.

## Intervals and resampling

- Use 10,000 question-level percentile resamples with seed **20261008**, NumPy PCG64 and linear quantiles at 0.025/0.975. Record the generator and dependency versions.
- Sample question indices with replacement; share each draw across configurations and contrasts. Keep matched runs together. Preserve frozen question ordering and record the sampled-index digest during execution.
- For pair-specific eligibility masks, apply the mask within each shared full-set draw and divide by that draw's eligible count. If any draw has no eligible pairs, report the interval unavailable and the undefined-draw count.
- For a common filtered question set, resample the retained ordered set with the same generator, seed and resample count.
- Flag constant bootstrap distributions and zero-width intervals. Retain the declared method.
- Add supplementary two-sided 95% **Newcombe method 10 paired score intervals** using all four paired cells and the paired correlation, alongside the bootstrap intervals. Use the same denominator and orientation. Verify extreme margins and zero-discordance behavior against the paired-method reference.
- Keep question-level resampling and disclose the two identified shared-context pairs. Wilson, McNemar and ordinary question-bootstrap inference assumes independent or approximately independent question units. Interpret intervals as question-sampling uncertainty conditional on these realized responses; generative rerun variability is not estimated.

## Sensitivities

- **Original first attempt:** report original-key accuracy/Wilson intervals on all 1,273 questions, with original failures incorrect. Report initial-attempt failure categories and show changes from first-valid primary accuracy descriptively.
- **Paired complete-valid:** retain questions with selected valid responses from both configurations. Report paired eligible n, excluded IDs, difference and bootstrap interval. It estimates accuracy conditional on both producing usable answers.
- **Potentially missing visual material:** retain all 1,273 questions in primary analysis. In a secondary descriptive sensitivity, exclude all 108 questions in the fixed nomination set, leaving 1,165. Apply identical exclusions to every configuration and each separate repeat. Report filtered correct/n, Wilson intervals and paired differences/bootstrap intervals. Preserve original keys and first-valid/failure scoring.
- Sensitivities are descriptive robustness checks. They introduce no additional hypothesis-test family.

## Cost, latency and frontiers

- Resource comparisons use each configuration's primary run. Cost and latency use the same final attempt for each question: the valid initial response, first valid retry, or last terminal attempt for an exhausted failure. Validity is determined by the parser, independently of correctness. Earlier unsuccessful-attempt charges and durations are excluded from primary resource summaries and frontiers.
- Report observed final-attempt billing, missing counts, and the imputed analysis basis defined below. Mean cost divides the analysis total by 1,273. A known-charge subtotal is labeled partial when a complete observed or imputed basis is unavailable. Missing charges are never treated as zero or replaced with an earlier attempt's charge.
- Keep token-rate estimates separate from billed costs and document the rates/accounting assumptions. Cost per correct answer is undefined with zero correct answers and unavailable on an incomplete cost basis.
- Measure latency as the request-wall duration of the selected final attempt. Report its sum, median, IQR and p95 across the 1,273 primary question outcomes, with observed, missing and imputed counts. Each question contributes at most one duration, including exhausted failures. Queueing, backoff, provider pauses and collection windows are reported separately; they are not added to final-attempt latency.
- Report resource and failure summaries by configuration, collection campaign, segment, provider, service dates, concurrency and pacing. Attribute each final-attempt resource value to the segment in which that attempt was made, including the failure-recovery campaign when applicable. Report initial and final failure categories and truncation rates for both arms of each effort contrast.
- Plot all 28 primary configurations together. The cost coordinate is mean final-attempt cost in USD per question. The latency coordinate is median final-attempt request-wall duration in seconds across the 1,273 primary question outcomes, including exhausted failures and applying the missing-value rules below. Use these same selected attempts and coordinates for accuracy–cost, accuracy–latency and three-dimensional frontiers.
- Define dominance as accuracy at least as high and resource use at most as high, with one strict improvement; exact ties remain jointly nondominated. Frontier membership is descriptive.
- Primary frontiers use the common observed-plus-imputed resource basis below. Also report observed-only frontiers, excluding configurations with incomplete coordinates. Identify all exclusions, imputed coordinates and any separate token-estimated frontiers.

## Missing resource values and imputation

The same rules apply to the primary run of every generative configuration and decision model, regardless of collection campaign.

- **Cost:** replace each missing final-attempt charge with the unrounded arithmetic mean of observed final-attempt charges from the same configuration's primary run. Include observed zero charges and observed charges on failed final attempts. Exclude earlier unsuccessful-attempt charges, other configurations, repeat runs and imputed values from the donor pool.
- **Duration:** replace each missing selected-final-attempt duration with the unrounded arithmetic mean of observed selected-final-attempt durations from the same configuration's primary run. Include observed zero durations and final failed attempts for exhausted questions. Exclude earlier unsuccessful attempts, other configurations, repeat runs and imputed values from the donor pool. Do not impute durations for calls excluded from primary latency. Review or reconciliation timestamps do not substitute for measured call durations.
- With no observed donors in the required pool, leave the estimate unavailable. Report the donor count, observed subtotal, exact donor mean, missing and imputed counts, each assigned value and its question/run/attempt identity. Store assignments separately from raw missing values.
- Use observed values plus assignments for primary cost totals/means, cost per correct answer, duration summaries and frontiers. Label imputed results and report observed-only summaries alongside them. Single mean imputation adds no uncertainty for missing values and can reduce observed duration dispersion; these summaries are descriptive.
- Repeat runs supply confidence/answer repeatability and separate observed resource summaries; missing secondary resource values remain unavailable. Do not impute answer labels, correctness, option probabilities or provider confidence fields.

## Confidence

- Analyze all 13 decision models separately using each selected first-valid run-1 native selected-option probability. Use native selected-option probabilities for confidence analyses.
- Require four matching option labels, finite values in [0,1], positive sum and inclusive absolute sum deviation ≤0.02. Apply thresholds without binary floating-point boundary errors. Preserve raw vectors, sums/deviations, eligible n/1,273 and exclusion reasons.
- Apply the same numerical eligibility rule to D01–D13. Verify field meaning, label mapping and precision using development evidence before evaluating confidence. Missing or unreviewed fields produce unavailable confidence results with reasons and denominators; otherwise valid selected answers remain in accuracy. Do not reconstruct absent probabilities or infer calibration from a provider description.

- Keep native selected-option values for confidence analyses. Normalize vectors only for distribution comparisons. Invalid vectors do not remove otherwise valid answers from accuracy scoring.
- **Measures:** correctness AUROC; ten fixed equal-width reliability bins, with 1 included in the last bin; error count/rate among eligible scores ≥0.90; risk–coverage including whole score ties and eligible coverage.
- AUROC is unavailable with one correctness class. Reliability interpretation requires reviewed probability semantics for that model.
- **Additional reporting:** score-support counts/range/endpoints; model-specific stratified 10,000-resample percentile AUROC intervals preserving observed correct/incorrect class counts; bin-wise Wilson accuracy intervals; binary Brier `mean((p_selected − correct)^2)` on the same eligible responses after field review, using squared-error probability scoring. Use seed 20261008 and record generator, class counts and degeneracy.
- Constant scores retain their observed behavior: AUROC=0.5 if both classes occur; risk–coverage has one attainable nonzero coverage point. Report empty bins and undefined/degenerate intervals.
- Confidence results are secondary/descriptive. E2 adds no calibration observations. Preserve native outputs; fit no recalibration model or test-set threshold.

## Decision-model repeatability

- For each of D01–D13 separately, report all three accuracies/Wilson intervals, selected-valid rates and service dates.
- Report agreement for each of the three run pairs over valid selected labels, with valid-pair n; report three-way agreement over valid triples, with triple n.
- Report correctness/failure transitions over all original question/run identities, including stable wrong answers.
- For eligible vector pairs, normalize each vector and calculate total variation `0.5 × sum(|p_r−p_s|)`; report mean/median, eligible n and exclusions.
- Report absolute changes in the earlier run's native selected-option probability, with its eligible denominator and exclusions.
- Also report absolute changes between each run’s native probability for its own selected answer, with exact unchanged counts/rates, mean/median/p95 and whole-tie cumulative distributions. Retain the earlier-selected-option measure separately so a label change does not obscure the compared quantity.
- Report exact three-run agreement separately for native selected-answer probabilities and native four-option vectors, with eligible/excluded counts. Interpret exact equality at each model's returned precision.
- Valid-label agreement excludes failures. Confidence-change curves use their own eligible denominators.
- Compare repeatability profiles descriptively; generative configurations have no repeatability data.
- Calculate the same eligible measures for all 13 models. Use compact common summaries and model-specific supplementary panels where a 13-model display would obscure denominators or trajectories; retain complete model-level outputs. Add no confidence or repeatability hypothesis-test family.


## Fixed visual-nomination sensitivity

- Exclude the fixed 108-question nomination set listed below: 107 screened candidates plus ambiguous item 000795. The screen identified explicit references to absent photographs, imaging, microscopy, tracings, graphs, diagrams, arrows and pedigrees. Item 000795 states “Cardiac examination is shown” without identifying the medium.
- Describe this secondary analysis as **post hoc exclusion of questions flagged for potentially missing visual material that was not supplied to the models**. The screen includes visual references and an ambiguous examination reference; image necessity is not adjudicated for the preprint.
- Validate all 108 unique IDs against the frozen 1,273-question set and bind the exact excluded IDs, nomination rule, question checksum and implementing source to the statistical release. Use a common retained n=1,165, with the frozen relative question order, across primary configurations and separate repeats.
- Disclose outcome-informed nomination and post hoc timing. The filtered analysis changes question composition and difficulty; it does not estimate the causal effect of withholding images.
- Broad visual-item exclusion has precedent in [Kung et al. (2023)](https://doi.org/10.1371/journal.pdig.0000198). [Nori et al. (2023)](https://arxiv.org/html/2303.13375v2) separately report questions referencing unavailable visual media.
- Detailed answerability review, adjudication and inter-rater reliability assessment are deferred to the full paper.

### Visual-nomination IDs

All suffixes below use the question-ID prefix `medqa-us-evaluation-c3b905ccfa66152d-`.

000010, 000012, 000019, 000032, 000046, 000075, 000078, 000096, 000112, 000117, 000124, 000127, 000140, 000141, 000147, 000161, 000206, 000221, 000233, 000234, 000252, 000257, 000265, 000271, 000275, 000285, 000312, 000331, 000332, 000337, 000341, 000366, 000374, 000398, 000407, 000412, 000449, 000455, 000468, 000470, 000471, 000476, 000485, 000486, 000519, 000522, 000538, 000547, 000558, 000573, 000576, 000582, 000588, 000604, 000617, 000623, 000634, 000641, 000645, 000654, 000658, 000669, 000677, 000683, 000684, 000686, 000710, 000712, 000717, 000726, 000728, 000731, 000740, 000762, 000786, 000794, 000795, 000822, 000831, 000836, 000862, 000882, 000911, 000928, 000936, 000946, 000973, 000978, 001003, 001021, 001033, 001061, 001065, 001080, 001096, 001111, 001144, 001164, 001178, 001187, 001198, 001202, 001203, 001212, 001238, 001240, 001249, 001260

## Reproducibility and reporting

- The official evaluation source file has SHA-256 `c3b905ccfa66152dc25afbcb2c10e86c0bdb208824f0658fcdb2c040f60a2beb`. Record input/key checksums, collection campaign and run/attempt identities, requested/returned model and provider, execution settings, analysis source, statistical settings, seed and dependency versions with the reported results.
- Link the reported G/D configuration IDs to source IDs through the configuration-ID crosswalk. Record original and retry calls, selected outcomes, probability eligibility, missing resources and imputation assignments separately. Report denominators, exclusions, failures and imputed values alongside the affected estimates.
- Release the analysis notebook, implementation/dependencies, reproduction instructions and question-level analysis tables. Bind exported tables and figures to the saved numerical results and their checksum.
- The study is unregistered. Core accuracy, Wilson, McNemar, Holm and bootstrap methods were documented before original collection on 6 October 2026. The 13-model decision roster, 273-contrast family, additional reporting, visual sensitivity and resource-imputation rules were specified after access to outcomes from the Original and Luna extension campaigns. Collection dates and analysis versions are recorded in the run provenance.



