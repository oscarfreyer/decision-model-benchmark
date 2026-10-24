"""Verified cross-study D01 joins and current complete-roster statistics.

The virtual manifest is an offline view. Each source retains its original study,
configuration, execution provenance and independently validated attempt ledger.
"""
from pathlib import Path

from common import BUDGET_ROSTERS, KIMI_INSTANT_DECISION, attempt_identity, fingerprint, strict_json

JOIN_WORKFLOW = "lean-analysis-joined-v1"
DECISIONS_WORKFLOW = "lean-decisions-v1"
CORE_ROSTER = {*BUDGET_ROSTERS[KIMI_INSTANT_DECISION], "D01"}
PRIMARY_FAMILY = "decision_vs_generative_31"


def reject_repeated_sources(packages):
    """Reject repeated packages and an ancestor supplied alongside its descendant."""
    selected = [Path(p).resolve() for p in packages]
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate saved package path")
    for selected_path in selected:
        current, seen = selected_path, set()
        while True:
            if current in seen:
                raise ValueError("Continuation parent cycle")
            seen.add(current)
            data = strict_json((current / "manifest.json").read_text())
            link = data.get("continuation")
            if not link:
                break
            parent = Path(link["parent"])
            if parent.is_absolute():
                raise ValueError("Continuation parent must be relative")
            current = (current / parent).resolve()
            if current in selected:
                raise ValueError("Continuation ancestor supplied twice")


def join_studies(loaded, packages):
    """Join independently verified original and D01 packages without rebinding."""
    workflows = {m.get("execution_workflow") for m, _, _ in loaded}
    if workflows != {"lean-v1", DECISIONS_WORKFLOW}:
        raise ValueError("Core analysis requires original lean-v1 and D01 lean-decisions-v1 packages")
    manifests = [m for m, _, _ in loaded]
    original = next(m for m in manifests if m["execution_workflow"] == "lean-v1")
    for source in manifests:
        if any(source.get(k) != original.get(k) for k in ("benchmark", "split", "items", "provenance")):
            raise ValueError("Core packages disagree on benchmark/split/items/order/provenance")
        for binding in ("questions", "keys"):
            if source["bindings"].get(binding) != original["bindings"].get(binding):
                raise ValueError("Core packages disagree on question/key hash")
        if source.get("key_manifest", {}).get("sha256") != original.get("key_manifest", {}).get("sha256"):
            raise ValueError("Core packages disagree on isolated original keys")
    for workflow in workflows:
        group = [m for m in manifests if m["execution_workflow"] == workflow]
        for other in group[1:]:
            if any(other.get(field) != group[0].get(field) for field in
                   ("study_id", "bindings", "configurations", "key_manifest", "schedule", "roster_decision", "probability_tolerance")):
                raise ValueError("Packages within one source study have incompatible execution/configuration bindings")
    configs = {}
    for source in manifests:
        for config in source["configurations"]:
            cid = config["config_id"]
            if cid in configs and configs[cid] != config:
                raise ValueError("Conflicting source configuration identity")
            configs[cid] = config
    if set(configs) != CORE_ROSTER:
        raise ValueError("Core analysis requires exactly J01, D01 and the 15 retained generative configurations")
    runs = [r for source in manifests for r in source["runs"]]
    if (len({r["run_id"] for r in runs}) != len(runs)
            or len({(r["config_id"], r["replicate"]) for r in runs}) != len(runs)):
        raise ValueError("Duplicate source run or replicate identity")
    config_order = [c["config_id"] for c in original["configurations"]] + ["D01"]
    runs.sort(key=lambda r: (r["experiment_id"], config_order.index(r["config_id"]), r["replicate"]))
    rows = [row for _, records, _ in loaded for row in records]
    if len({attempt_identity(row) for row in rows}) != len(rows):
        raise ValueError("Duplicate source attempt identity")
    manifest = {
        "schema_version": 1, "execution_workflow": JOIN_WORKFLOW,
        "study_id": original["study_id"], "provenance": original["provenance"],
        "benchmark": original["benchmark"], "split": original["split"],
        "items": original["items"], "key_manifest": original["key_manifest"],
        "bindings": {"questions": original["bindings"]["questions"], "keys": original["bindings"]["keys"]},
        "configurations": [configs[cid] for cid in config_order], "runs": runs,
        "roster_decision": KIMI_INSTANT_DECISION,
        "source_manifests": manifests,
        "source_packages": [{"package_id": fingerprint(m), "manifest_sha256": fingerprint(m),
                             "study_id": m["study_id"], "bindings": m["bindings"],
                             "execution_workflow": m["execution_workflow"],
                             "provenance": m["provenance"], "schedule": m.get("schedule"),
                             "execution_segments": coverage.get("segments", [])}
                            for p, (m, _, coverage) in zip(packages, loaded)],
    }
    coverage = {
        "expected": sum(c["expected"] for _, _, c in loaded), "terminal": len(rows),
        "unstarted": [i for _, _, c in loaded for i in c["unstarted"]],
        "indeterminate": [i for _, _, c in loaded for i in c["indeterminate"]],
        "complete": all(c["complete"] for _, _, c in loaded), "last_event_sha256": None,
        "source_chain_heads": [c["last_event_sha256"] for _, _, c in loaded],
    }
    return manifest, rows, coverage


def validate_join(manifest):
    """Recompute virtual plan and validate every source using its own contract."""
    from benchmarking.common import validate_manifest
    sources = manifest.get("source_manifests", [])
    packages = manifest.get("source_packages", [])
    if len(sources) != len(packages) or len(sources) < 2:
        raise ValueError("Core manifest requires separate source provenance")
    for source, package in zip(sources, packages):
        validate_manifest(source)
        if fingerprint(source) != package["manifest_sha256"]:
            raise ValueError("Core source manifest binding mismatch")
    empty = {"expected": 0, "terminal": 0, "unstarted": [], "indeterminate": [],
             "complete": True, "last_event_sha256": None}
    rebuilt, _, _ = join_studies([(m, [], {**empty, "segments": p["execution_segments"]})
                                for m, p in zip(sources, packages)], [p["package_id"] for p in packages])
    if rebuilt != manifest:
        raise ValueError("Core virtual plan differs from original source manifests")
    return manifest


def finish_source(record, manifest):
    run_id = record["identity"]["run_id"]
    sources = [source for source in manifest["source_manifests"]
               if any(run["run_id"] == run_id for run in source["runs"])]
    if len(sources) != 1:
        raise ValueError("Core attempt requires one original source manifest")
    return sources[0]


def core_paired(matrix, configs, pairs, *, seed, valid_matrix=None, inference=True):
    """One explicitly reviewed S1=31 family and unchanged S2=3 family."""
    from analysis.methods.pipeline import paired_contrasts
    if set(matrix) != CORE_ROSTER or set(configs) != CORE_ROSTER or set(pairs) != {"P03", "P07", "P09"}:
        raise ValueError("Core contrasts require all 17 aligned primary configurations and three effort pairs")
    generative = sorted(CORE_ROSTER - {"J01", "D01"})
    contrasts = [("S1", f"{decision}-{cid}", decision, cid)
                 for decision in ("J01", "D01") for cid in generative]
    contrasts += [("S1", "J01-D01", "J01", "D01")]
    contrasts += [("S2", pair, levels["upper"], levels["lower"]) for pair, levels in sorted(pairs.items())]
    return paired_contrasts(matrix, configs, contrasts, {"S1": 31, "S2": 3}, seed=seed,
                            valid_matrix=valid_matrix, inference=inference)
