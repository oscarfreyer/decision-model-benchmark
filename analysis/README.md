# Analysis and verification

Create and open the writable execution copy described in the root README using the installed `.venv/bin/python` environment. Keep the canonical `Analysis.ipynb` cleared. The notebook presents the saved numerical reference and uses the frozen presentation source unchanged. Public data loading and disclosure-filtered export are implemented in `public_release.py`; scientific kernels remain in `methods/`.

The following commands perform offline verification. Retain their exit statuses and outputs with the exact candidate archive SHA-256:

```sh
python -m analysis.public_release verify
python -m unittest analysis.tests.test_public_release analysis.tests.test_inputs_and_statistics analysis.tests.test_expanded_sap analysis.tests.test_final_sap -v
python -B verify.py
```

The retained tests use fictional inputs. Public-copy notebook/workflow behavior needs its own execution check because private frozen-run approval tests do not apply to the portable entry notebook. These workflow-only classes are omitted from the public copy of `test_final_sap.py`; the original file remains bound in the source mapping. No numerical assertion is removed from its retained statistical classes.

For the complete optional numerical reproduction, acquire the exact upstream MedQA test file and use the README's `analysis.public_release recompute` command. It writes only to a fresh output directory and requires exact scientific-summary equality with the public reference. A mismatch is retained for inspection.

`results/verified_exports/` contains the exact frozen graphics and 15 CSV projections suitable for this public package. The original 18-table export's questions and two answer/key tables are excluded; the [export mapping](../provenance/export_mapping.json) records each disposition and source hash. Public outcome/call records add the first-attempt information needed for recalculation. The original private numerical and presentation freezes are unchanged.
