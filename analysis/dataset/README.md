# Dataset readers

The public reader is `analysis.public_release`. It consumes `data/study.json`, `outcomes.jsonl.gz` and `calls.jsonl.gz` after checking the public manifest. It rebuilds analysis-shaped records for the unchanged SAP kernels and reconstructs keys only from an explicitly supplied upstream MedQA file.

The historical flat/extension readers are retained as original collection interfaces. They require the original raw, manifest-bound packages and are not the public reproduction route. The explicit `structured-results-v1` dispatch in `expanded.py` applies only to this separately versioned derivative; it does not weaken the original raw-package checks.
