# Structured study data

The CC BY 4.0 derivative contains 68,742 question/configuration/run outcomes and the scientific fields from 69,318 calls. It is extracted from the verified study observations; raw API payloads and benchmark inputs remain private or upstream. The data files are bound by [data_manifest.json](../provenance/data_manifest.json).

| File | Purpose |
| --- | --- |
| `study.json` | Ordered item IDs/content hashes, frozen effective model settings, 54 runs in replicate/roster order, method settings and original numerical-source digest. |
| `outcomes.jsonl.gz` | One row per question/configuration/run, with final and original outcomes and ordered call IDs. |
| `calls.jsonl.gz` | One row per initial/retry call, containing the explicit scientific field projection. |
| `reference_results.json.gz` | Saved numerical estimates and plotting arrays for presentation and full-summary comparison. |

The G15 configuration retains the historical effective effort label `none`, corresponding to the Kimi Instant variant with thinking disabled. Runs follow replicate order and the frozen G01-G15, D01-D13 roster. These metadata reproduce the accepted comparison labels and frontier-list order; the underlying outcomes and numerical results are unchanged.

# Outcome fields

`identity` uses reporting IDs G01-G15 and D01-D13. `source_identity` and `source_namespace` preserve the original collection identity. `status` and `selected_label` describe the first valid response, or final exhausted failure. `correct` is the selected response's correctness under the original upstream key; failures are incorrect. `original_status`, `original_selected_label` and `original_correct` describe the initial attempt and support the first-attempt sensitivity. No separate reference-key file or explicit reference-key field is included. The released selected labels and correctness can reveal reference answers; the MedQA answer keys are publicly available upstream.

`selection` retains original/selected/cost attempt IDs, selected retry index and recovery outcome. `call_ids` lists every preserved initial/retry call in retry-index order. Selection is independent of correctness. An exhausted outcome has eleven calls; a valid response stops recovery.

# Call fields

`attempt_id` identifies the retained observation; `retry_index` is zero for the initial call. `source_package` identifies its collection segment. The nested `record` contains the mapped identity, terminal status, selected label, failure classification, native token counters with missing-value reasons, recorded billing, request duration and queue wait, timestamps, scientific route/settings and scheduler provider.

`probability.native_vector` retains the reported A-D values, together with sum, deviation, validity/reason and native selected-answer probability. Values are not rounded or repaired in this derivative. The scientific kernels apply the same reviewed probability eligibility and normalization rules. Additional provider-specific scalar analyses are excluded under SAP v1.8.

Charges are USD. Durations and queue waits are seconds. Zero and unavailable measurements remain distinct; missing values retain their recorded reasons. Primary resource imputation is performed by the unchanged scientific kernel, rather than filled into public raw measurements. Repeated-run resources retain observed accounting.

# Provenance and omissions

[Field mapping](../provenance/data_field_mapping.json) binds the source result and source attempt files and lists retained/omitted fields. The public loader verifies derivative file hashes and study consistency. It cannot independently revalidate the private request/response journals; the frozen source digests identify that retained evidence.

The saved reference omits source-package manifests, private environment/input records and collection-provenance containers. Scientific estimates and plotting arrays remain unchanged. The full-summary comparison excludes only those named provenance fields. Question/option content hashes and the exact upstream test-file hash allow local reconstruction without bundling upstream question text, option text or a separate reference-key file.
