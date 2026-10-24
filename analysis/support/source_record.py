"""Explicit current-layout source/dependency closure for the later analysis freeze.

Historical freezes retain their original inventories. Restore their exact source
layout from the preserved archive when reproducing an older accepted release.
"""
from common import ROOT, file_hash

SOURCES = (
    "common.py", "requirements.txt", "requirements.lock",
    "benchmarking/__init__.py", "benchmarking/common.py", "benchmarking/retry_common.py",
    "analysis/__init__.py", "analysis/Analysis.ipynb", "analysis/notebook-requirements.txt",
    "analysis/consolidate.py", "analysis/pipeline.py", "analysis/recovery.py",
    "analysis/dataset/__init__.py", "analysis/dataset/consolidate.py",
    "analysis/dataset/flat.py", "analysis/dataset/inputs.py", "analysis/dataset/visual_review.py",
    "analysis/dataset/review_packet.py",
    "analysis/methods/__init__.py", "analysis/methods/core.py",
    "benchmarking/configuration_ids.py", "benchmarking/extension_contract.py",
    "analysis/dataset/expanded.py", "analysis/dataset/decision_field_review.json",
    "analysis/methods/expanded.py",
    "analysis/methods/decisions.py", "analysis/methods/pipeline.py",
    "analysis/methods/statistics.py", "analysis/support/__init__.py",
    "analysis/support/notebook.py", "analysis/support/figures.py",
    "analysis/support/lean.py", "analysis/support/recovery.py",
    "analysis/support/readiness.py", "analysis/support/reproduction.py",
    "analysis/support/source_record.py",
)


SOURCES = SOURCES + ("analysis/public_release.py",)

def analysis_code_record(root=ROOT):
    """Bind implementation bytes as well as the stable entry modules."""
    from pathlib import Path
    root = Path(root)
    return {name: file_hash(root / name) for name in sorted(SOURCES)}


# The notebook's numerical path. Keep the historical full SOURCES inventory above.
# Plotting code and its runtime versions have a separate presentation record.
STATISTICAL_SOURCES = (
    "common.py", "requirements.txt", "requirements.lock",
    "benchmarking/__init__.py", "benchmarking/common.py", "benchmarking/retry_common.py",
    "analysis/__init__.py", "analysis/Analysis.ipynb", "analysis/notebook-requirements.txt",
    "analysis/dataset/__init__.py", "analysis/dataset/flat.py",
    "analysis/dataset/inputs.py", "analysis/dataset/visual_review.py",
    "analysis/methods/__init__.py", "analysis/methods/core.py",
    "benchmarking/configuration_ids.py", "benchmarking/extension_contract.py",
    "analysis/dataset/expanded.py", "analysis/dataset/decision_field_review.json",
    "analysis/methods/expanded.py",
    "analysis/methods/decisions.py", "analysis/methods/pipeline.py",
    "analysis/methods/statistics.py", "analysis/support/__init__.py",
    "analysis/support/notebook.py", "analysis/support/source_record.py",
)
PRESENTATION_SOURCES = (
    "benchmarking/configuration_ids.py",
    "common.py", "analysis/__init__.py", "analysis/Analysis.ipynb",
    "analysis/support/__init__.py", "analysis/support/notebook.py",
    "analysis/support/source_record.py", "analysis/support/figures.py",
)


STATISTICAL_SOURCES = STATISTICAL_SOURCES + ("analysis/public_release.py",)
PRESENTATION_SOURCES = PRESENTATION_SOURCES + ("analysis/public_release.py", "analysis/dataset/expanded.py")

def statistical_code_record(root=ROOT):
    """Bind the notebook and complete numerical implementation, excluding plotting."""
    from pathlib import Path
    root = Path(root)
    return {name: file_hash(root / name) for name in sorted(STATISTICAL_SOURCES)}


def presentation_code_record(root=ROOT):
    """Record plotting code and the shared notebook/result/output helpers it uses."""
    from pathlib import Path
    root = Path(root)
    return {name: file_hash(root / name) for name in PRESENTATION_SOURCES}
