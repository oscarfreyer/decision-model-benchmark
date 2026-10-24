# Preparation and hosted-model collection

The included collection code records the study's original routes, prompts, native settings and recovery rules. Its configurations describe historical endpoints; API availability and terms may change. Saved-result reproduction uses no API calls.

`python -m preparation.prepare_medqa --help` describes the required upstream train/development/test inputs, source provenance and explicit output name. Preparation retains the four A-D options, checks source/key consistency and writes solution-free question records plus separate keys. The evaluated test-file hash and reconstruction route are in the root README.

`python -m benchmarking.run --help` describes the main collection, Luna Decisions, failure-recovery and smoke entry points. Collection commands require an explicit `--live` or `--mock`; neither is inferred. Real calls require the applicable OpenRouter/TypeSafe credentials and funded provider access. The `plan` command is a no-call count preview. `python -m benchmarking.extension --help` describes the native decision-model extension. Use a new study and output directory for any new collection.

The [study protocol](../methods/Study_Protocol.md) records the performed campaign settings and model roster. It supplies a public scientific protocol file where the CLI requests one. Preparation keys are consumed only by offline scoring. Original packages are immutable; reviewed continuations and supplementary failures use the collector's explicit commands.

The public structured data omit raw request/response bodies and are consumed through `analysis.public_release`. They do not replace the collector's raw package format for resuming or validating a new collection.
