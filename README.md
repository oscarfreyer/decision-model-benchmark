# Decision model benchmark

Code and structured results for **Benchmarking decision models against generative large language models on medical multiple-choice questions**. The study evaluated 15 generative configurations and 13 decision models on the same 1,273 four-option MedQA questions. Generative configurations had one run each; decision models had three, giving 54 runs and 68,742 outcomes.

Release candidate 0.1.0-rc3 uses the SAP v1.8 numerical release and presentation 1.51. The three main figures use portrait A4 layouts. Five Extended Data figures retain the reviewed layouts. [Verified exports](results/verified_exports/) contain the exact publication graphics and disclosure-filtered numerical table projections. Source-to-public mappings are in [provenance](provenance/).

# Install

Use Python 3.14.7. From this repository's root:

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
```

The lock records the environment used for the verified frozen presentation on macOS 27.0.1 arm64. Installation and execution results are recorded separately against this candidate archive's SHA-256; the captured environment describes macOS arm64. No installation-time or minimum-memory measurement is claimed.

Arial is required for the exact publication rendering. The font is installed separately and its reviewed hash is recorded in [presentation provenance](provenance/presentation.json). No font binary or local model weights are included.

# View and reproduce the presentation

Keep [analysis/Analysis.ipynb](analysis/Analysis.ipynb) as the cleared source. Create a writable execution copy so saved outputs do not change its release binding:

```sh
mkdir -p analysis/output/public-review-v01
cp analysis/Analysis.ipynb analysis/output/public-review-v01/Review_Analysis.ipynb
```

Open the repository folder in VS Code, open that execution copy, select `.venv/bin/python`, restart the kernel and Run All. Save the completed copy. JupyterLab is also supported by the pinned environment:

```sh
python -m jupyterlab analysis/output/public-review-v01/Review_Analysis.ipynb
```

This is a portable derivative of the same frozen notebook. Nine scientific display and figure cells, including all figure settings, are unchanged. Startup loads the public structured reference through relative paths; it does not read the private raw logs or require an API key. The notebook displays the scientific tables, eight static figures and interactive resource figure.

Exports are enabled to a fresh `analysis/output/public-render-v01/`. They include PDF/SVG/300-dpi PNG graphics, copies of the verified public CSV projections and a new export manifest. The three main PNGs are checked against the frozen preview hashes. Existing output directories are refused; choose a new name when intentionally producing another run.

The canonical notebook is cleared. Installation, execution and reproduction checks are recorded separately against the exact candidate archive SHA-256.

# Recalculate the numerical analysis

The structured data retain selected answers, native probability vectors, token/resource measurements, first-attempt outcomes and the scientific fields from every preserved initial/retry call. The optional recalculation route uses the original scientific kernels. It needs locally reconstructed MedQA reference keys, rather than inferred keys from model answers.

First obtain the original US four-option `phrases_no_exclude_test.jsonl` from the [upstream MedQA repository](https://github.com/jind11/MedQA), under `data_clean/questions/US/4_options/`. The evaluated file has SHA-256:

```text
c3b905ccfa66152dc25afbcb2c10e86c0bdb208824f0658fcdb2c040f60a2beb
```

Then, from this repository's root:

```sh
python -m analysis.public_release verify
python -m analysis.public_release recompute --medqa-test /path/to/phrases_no_exclude_test.jsonl --output analysis/output/recomputed-v01
```

The loader checks the public data hashes, exact 28-configuration/54-run coverage, continuous retry histories and first-valid selection. It checks all 1,273 upstream question/option hashes, reconstructs reference keys locally and verifies selected and first-attempt correctness. Recalculation uses the accepted seed 20261008, 10,000 bootstrap resamples, the 273-comparison S1 family, three-comparison S2 family and fixed 108-item visual sensitivity. A different result is preserved for review and reported as an error. A successful command writes `verification.json` beside the recalculated summary.

Scientific methods are in the [study protocol](methods/Study_Protocol.md) and [SAP](methods/Statistical_Analysis_Plan.md). The [data dictionary](data/README.md) describes the structured fields, omissions and provenance boundary.

# Collection and adaptation

`preparation/` converts compatible four-option MedQA inputs into solution-free questions and separate keys. `benchmarking/` contains the original hosted-model collection code and fixed study configurations, including retry recovery and the decision-model extension. The [collection guide](documentation/Collection.md) describes the entry points. Reproducing the saved study does not require new paid calls.

For a new compatible dataset, retain four labelled options A-D, unique item IDs and a separate reference-key file. Use the preparation schema and an explicit new study/output location. The fixed MedQA analysis enforces this study's roster, pairing, run counts and visual exclusions; adapting its statistical design requires corresponding, reviewed method changes.

Fictional regression fixtures are included. [Verification instructions](analysis/README.md) distinguish data integrity, numerical checks and presentation reproduction. Suite results are recorded separately against the exact candidate archive. Historical source kept under `benchmarking/legacy/` supports retained regression bindings; it is not the entry point for new collection.

# Code, data and citation

Original code is licensed under [MIT](LICENSE). Structured study data and result graphics are licensed under [CC BY 4.0](data/LICENSE). Dependencies and upstream MedQA retain their own terms; see [third-party notices](THIRD_PARTY_NOTICES.md). Raw request/response bodies, reasoning text, credentials, benchmark question text and option text are excluded. No separate reference-key file or explicit reference-key field is included. The released selected labels and correctness can reveal reference answers; the MedQA answer keys are publicly available upstream. Numerical recalculation uses the verified upstream keys described above.

Use [CITATION.cff](CITATION.cff) to cite this release. Repository/archive DOI links can be added after publication. The release manifest identifies this local candidate's exact files; it does not claim that a remote repository or archive has been published. The separately supplied SHA-256 checksum file identifies the final manifest and ZIP. `.DS_Store` and other files outside the manifest inventory are excluded from the archive.
