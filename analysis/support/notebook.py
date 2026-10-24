"""Notebook inputs, response tables and the immutable numerical-result boundary."""
from pathlib import Path

from analysis.support.source_record import statistical_code_record
from common import (attempt_identity,
                    file_hash, load_questions, strict_json, utc_now, write_new)


def report_probability(probability):
    """Project raw probability evidence onto the active report's vector fields."""
    return {k: v for k, v in probability.items() if k in (
        "native_vector", "sum", "deviation", "valid", "reason",
        "selected_probability", "semantics")}


def report_stability(repeat):
    """Copy retained measures from current or historical repeatability results."""
    from copy import deepcopy
    repeat = deepcopy(repeat)
    repeat.pop("provider_scalars", None)
    for pair in repeat.get("pairs", []):
        for key in list(pair):
            if key.endswith("_change_separate"):
                del pair[key]
        if "confidence_change_distribution" in pair:
            pair["confidence_change_distribution"] = {
                k: v for k, v in pair["confidence_change_distribution"].items()
                if k in ("earlier_selected_probability", "total_variation")}
    if "confidence_three_way" in repeat:
        repeat["confidence_three_way"] = {
            k: v for k, v in repeat["confidence_three_way"].items()
            if k in ("selected_probability", "native_vector")}
    return repeat


def report_figure_data(data):
    """Discard retired scalar payloads without recalculating saved coordinates."""
    from copy import deepcopy
    return {k: deepcopy(v) for k, v in data.items()
            if not k.endswith("_repeatability_provider_confidence")}


def active_report_view(result):
    """Project report content only; preserve saved release/input provenance.

    This never writes or rebinds a historical bundle. The original file's digest
    continues to identify the source of the retained numerical values.
    """
    from copy import deepcopy
    result = deepcopy(result)
    summary = result["summary"]
    if "stability" in summary:
        summary["stability"] = report_stability(summary["stability"])
    for model in summary.get("decision_models", {}).values():
        model["stability"] = report_stability(model["stability"])
    for row in result["review_tables"].get("selected_attempts", []):
        row.pop("provider_scalars", None)
        row.pop("provider_confidence", None)
        row["probability"] = report_probability(row["probability"])
    result["figure_data"] = report_figure_data(result["figure_data"])
    return result


def repeatability_tables(summary, model_names):
    """Common notebook rows with explicit probability-only measure selection."""
    run_rows, pair_rows, triple_rows = [], [], []
    for cid, model in summary["decision_models"].items():
        repeat = report_stability(model["stability"])
        for run, row in repeat.get("runs", {}).items():
            run_rows.append({"Model": model_names[cid], "Run": run, **row})
        for pair in repeat.get("pairs", []):
            distributions = pair.get("confidence_change_distribution", {})
            for metric, values in (
                ("earlier_selected_probability", distributions.get("earlier_selected_probability")),
                ("total_variation", distributions.get("total_variation")),
                ("own_selected_probability", pair.get("selected_probability_change")),
            ):
                if values is None:
                    continue
                pair_rows.append({"Model": model_names[cid],
                    "Runs": f"{pair['earlier_run']}–{pair['later_run']}", "Measure": metric,
                    "Valid label pairs": pair["valid_pair_n"], "Label agreement": pair["answer_agreement"],
                    "Failure/correctness transitions": pair["transitions"],
                    **{k: v for k, v in values.items() if k not in ("ecdf", "field")},
                    "exclusions": values.get("exclusions", pair["vector_exclusions"])})
        triple = repeat.get("three_way", {})
        triple_rows.append({"Model": model_names[cid], "Measure": "valid selected labels",
            "Eligible n": triple.get("valid_triple_n"), "Unchanged n": triple.get("agreements"),
            "Rate": triple.get("rate"),
            "Excluded n": repeat["planned_n"]-triple["valid_triple_n"] if "valid_triple_n" in triple else None})
        for metric, values in repeat.get("confidence_three_way", {}).items():
            triple_rows.append({"Model": model_names[cid], "Measure": metric,
                "Eligible n": values["eligible_n"], "Unchanged n": values["unchanged_n"],
                "Rate": values["rate"], "Excluded n": values["excluded_n"]})
    return {"Per-run accuracy": run_rows, "Pairwise changes": pair_rows,
            "Exact agreement across all three runs": triple_rows}


def _approved_release_record(path, expected_sha256):
    """Authenticate a dated human release record against its independent digest.

    The independently retained release digest is supplied at runtime, outside
    notebook source, so the notebook and its release can be bound without a cycle.
    """
    import hashlib
    import re
    from datetime import datetime
    if path is None or not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Supply the approved analysis-release JSON and independent SHA-256")
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Analysis release differs from its approved SHA-256")
    record = strict_json(payload.decode("utf-8"))
    if (not isinstance(record, dict) or record.get("schema_version") != 1 or record.get("type") != "flat_notebook_analysis_release"
            or record.get("actor") != "human" or record.get("approved") is not True
            or not record.get("reviewer") or not record.get("approved_at")):
        raise ValueError("Dated human approval of the analysis release is required")
    datetime.fromisoformat(record["approved_at"])
    return record


def verify_notebook_release(path, expected_sha256, bindings):
    """Verify independently approved release bytes and exact runtime bindings."""
    record = _approved_release_record(path, expected_sha256)
    if record.get("bindings") != bindings:
        raise ValueError("Source, settings, environment, SAP or inputs differ from the approved analysis release")
    return {"sha256": expected_sha256, "reviewer": record["reviewer"], "approved_at": record["approved_at"]}


def input_inventory(packages, questions_path, keys_path, *, input_root=None):
    """Bind explicit input bytes, including every continuation parent."""
    inventory = {}
    root = Path(input_root).resolve() if input_root is not None else None

    def label(path):
        if root is not None:
            if not path.is_relative_to(root):
                raise ValueError("Input escapes the local input folder")
            return path.relative_to(root).as_posix()
        # Historical explicit-path callers retain hash identity without recording
        # an operator's absolute workspace path in derived provenance.
        return "sha256:" + file_hash(path)

    def visit(package, ancestors):
        package = Path(package).resolve()
        if package in ancestors:
            raise ValueError("Continuation parent cycle")
        for name in ("manifest.json", "events.jsonl", "provenance.json"):
            path = package / name
            if name != "provenance.json" or path.exists():
                inventory[label(path)] = file_hash(path)
        manifest = strict_json((package / "manifest.json").read_text())
        if manifest.get("execution_workflow") not in {"lean-v1", "lean-decisions-v1"}:
            raise ValueError("Notebook requires lean-v1 or lean-decisions-v1 packages; use the historical reader for older runs")
        link = manifest.get("continuation")
        if link:
            parent = Path(link["parent"])
            if parent.is_absolute():
                raise ValueError("Continuation parent must be relative")
            visit(package / parent, ancestors | {package})

    for package in packages:
        visit(package, set())
    for path in (questions_path, keys_path):
        path = Path(path).resolve()
        inventory[label(path)] = file_hash(path)
    return inventory


def load_review_inputs(packages, questions_path, keys_path, *, input_root=None):
    """Read verified packages once; reject inputs that change during loading."""
    from analysis.methods.pipeline import load_study
    packages = [Path(p) for p in packages]
    if not packages:
        raise ValueError("Select at least one explicit saved package")
    before = input_inventory(packages, questions_path, keys_path, input_root=input_root)
    manifest, terminal, coverage, keys = load_study(packages, questions_path, keys_path)
    questions = load_questions(questions_path)
    after = input_inventory(packages, questions_path, keys_path, input_root=input_root)
    if before != after:
        raise ValueError("Inputs changed during reading. Wait for collection to stop or use a stable saved copy")
    return manifest, terminal, coverage, keys, questions, before


def response_tables(manifest, terminal, coverage, keys, questions):
    """Return primary answers, statuses, question text and separate extra-run answers.

    Rows retain preparation order. Failed final answers show their terminal status;
    absent starts and unresolved starts have distinct markers. E2 never substitutes
    for a missing primary answer. These are review records, not scoring repairs.
    """
    from analysis.methods.pipeline import expected_identities

    def question_identity(question):
        # Current and historical readers have distinct Question classes.
        return question.benchmark, question.split, question.item_id

    if [q.item_id for q in questions] != [q["item_id"] for q in manifest["items"]]:
        raise ValueError("Response tables must retain preparation order")
    if set(keys) != {question_identity(q) for q in questions}:
        raise ValueError("Response tables require exact key coverage")
    records = {}
    expected = set(expected_identities(manifest))
    for record in terminal:
        ident = attempt_identity(record)
        if ident in records or ident not in expected:
            raise ValueError("Duplicate or unexpected terminal identity")
        records[ident] = record
    unstarted = {tuple(i) for i in coverage["unstarted"]}
    indeterminate = {tuple(i) for i in coverage["indeterminate"]}
    if (set(records) | unstarted | indeterminate != expected
            or set(records) & (unstarted | indeterminate) or unstarted & indeterminate):
        raise ValueError("Response-table coverage disagrees with terminal records")
    primary = {r["config_id"]: r for r in manifest["runs"] if r["role"] == "primary"}
    if len(primary) != sum(r["role"] == "primary" for r in manifest["runs"]):
        raise ValueError("One primary run per configuration required")
    secondary = [r for r in manifest["runs"] if r["role"] == "secondary"]

    def cell(q, run):
        if run is None:
            return "[not_planned]", "not_planned"
        ident = (*question_identity(q), run["config_id"], run["run_id"])
        record = records.get(ident)
        if record is None:
            status = "indeterminate" if ident in indeterminate else "unstarted"
            from common import fingerprint
            status = coverage.get("missing_outcome_reasons", {}).get(fingerprint(list(ident)), status)
            return f"[{status}]", status
        status = record["status"]
        if status == "valid":
            label = record["selected_label"]
            if label not in q.labels:
                raise ValueError("Selected answer absent from supplied options")
            return label, status
        return f"[{status}]", status

    answers, statuses, details, extra = [], [], [], []
    for q in questions:
        key = keys[question_identity(q)]["reference_answer"]
        options = {o.label: o.text for o in q.options}
        row = {"item_id": q.item_id, "correct_answer": key}
        status_row = {"item_id": q.item_id}
        for config in manifest["configurations"]:
            cid = config["config_id"]
            row[cid], status_row[cid] = cell(q, primary.get(cid))
        answers.append(row); statuses.append(status_row)
        details.append({"item_id": q.item_id, "benchmark": q.benchmark, "split": q.split,
                        "question": q.stem, **{f"option_{o.label}": o.text for o in q.options},
                        "correct_answer_text": options[key]})
        extra_row = {"item_id": q.item_id, "correct_answer": key}
        for run in secondary:
            extra_row[run["run_id"]] = cell(q, run)[0]
        extra.append(extra_row)
    secondary_by_configuration = {
        cid: [{"item_id": row["item_id"], "correct_answer": row["correct_answer"],
               **{run["run_id"]: row[run["run_id"]] for run in secondary if run["config_id"] == cid}}
              for row in extra]
        for cid in dict.fromkeys(run["config_id"] for run in secondary)
    }
    selected_attempts = [{"identity": r["identity"], "source_identity": r.get("source_identity", r["identity"]),
        "source_namespace": r.get("source_namespace"), "status": r["status"],
        "selected_label": r["selected_label"], "selection": r.get("analysis_selection"),
        "billing": r["billing"], "timing": r["timing"], "probability": report_probability(r["probability"])}
        for r in terminal]
    return {"answers": answers, "statuses": statuses, "questions": details, "selected_attempts": selected_attempts,
            "secondary_answers": extra if secondary else [],
            "secondary_by_configuration": secondary_by_configuration,
            "models": [{"config_id": c["config_id"], "model": c["base_model"],
                        "effort": c["effort"], "primary_run_id": primary.get(c["config_id"], {}).get("run_id")}
                       for c in manifest["configurations"]]}


def notebook_code_record(root):
    """Bind the later notebook release without changing the execution inventory."""
    root = Path(root)
    return statistical_code_record(root)


def read_execution_notebook(root, path):
    """Read one cleared saved copy and compare its exact bytes to canonical source."""
    import hashlib
    root = Path(root).resolve()
    if path is None or not str(path).strip():
        raise ValueError("Supply JEV_EXECUTION_NOTEBOOK for the saved execution copy")
    path = Path(path).expanduser()
    path = (path if path.is_absolute() else root/path).resolve()
    output_root = (root/"analysis/output").resolve()
    if not path.is_relative_to(output_root) or path.suffix != ".ipynb" or not path.is_file():
        raise ValueError("Execution notebook must be a saved .ipynb copy beneath analysis/output")
    payload = path.read_bytes()
    if payload != (root/"analysis/Analysis.ipynb").read_bytes():
        raise ValueError("Execution notebook differs from the canonical cleared source")
    notebook = strict_json(payload.decode("utf-8"))
    if (not isinstance(notebook, dict) or notebook.get("nbformat") != 4
            or not isinstance(notebook.get("cells"), list)):
        raise ValueError("Execution notebook must use notebook format 4")
    if any(c.get("execution_count") is not None or c.get("outputs", [])
           for c in notebook["cells"] if c.get("cell_type") == "code"):
        raise ValueError("Execution notebook must have cleared counts and outputs")
    return payload, {"path": path.relative_to(root).as_posix(),
                     "sha256": hashlib.sha256(payload).hexdigest()}


def execution_notebook_record(root, path):
    return read_execution_notebook(root, path)[1]


def require_compute_execution(root, path):
    """Final computation uses the saved snapshot supplied by the fresh-kernel runner."""
    import os
    record = execution_notebook_record(root, path)
    if os.environ.get("JEV_SAVED_NOTEBOOK_SHA256") != record["sha256"]:
        raise ValueError("Run definitive compute with python -B -m analysis.support.notebook --compute-notebook COPY")
    return record


def review_item_id(item_ids, requested=None):
    """Select a review question without editing notebook source or numerical results."""
    item_ids = list(item_ids)
    if not item_ids:
        raise ValueError("No question rows are available for review")
    if requested is None or requested == "":
        return item_ids[0]
    if requested not in item_ids:
        raise ValueError("JEV_REVIEW_ITEM_ID must match a question ID in the response table")
    return requested


def run_compute_notebook(path, *, root=None):
    """Execute verified saved bytes in a new kernel; preserve the cleared copy."""
    import os
    import nbformat
    from nbclient import NotebookClient
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    payload, execution = read_execution_notebook(root, path)
    release = _approved_release_record(os.environ.get("JEV_ANALYSIS_RELEASE_JSON"),
                                       os.environ.get("JEV_ANALYSIS_RELEASE_SHA256"))
    bindings = release.get("bindings")
    if (not isinstance(bindings, dict) or bindings.get("analysis_code") != notebook_code_record(root)
            or bindings.get("executed_notebook_sha256") != execution["sha256"]):
        raise ValueError("Saved notebook or statistical source differs from the approved release")
    executed_path = (root/execution["path"]).with_name("Executed_Analysis.ipynb")
    if executed_path.exists() or executed_path.is_symlink():
        raise FileExistsError("Executed notebook destination already exists; use a fresh run directory")
    fresh_output_directory(os.environ.get("JEV_RESULT_DIR"), root)
    # This exact in-memory snapshot is passed to the executor. Kernel variables
    # identify its saved bytes; interactive review cannot start final computation.
    notebook = nbformat.reads(payload.decode("utf-8"), as_version=4)
    environment = {**os.environ, "JEV_ANALYSIS_MODE": "compute",
                   "JEV_EXECUTION_NOTEBOOK": str(root/execution["path"]),
                   "JEV_SAVED_NOTEBOOK_SHA256": execution["sha256"]}
    client = NotebookClient(notebook, timeout=None, kernel_name="python3",
                            allow_errors=False, force_raise_errors=True)
    try:
        client.execute(cwd=str(root), env=environment)
        if (read_execution_notebook(root, path)[0] != payload
                or notebook_code_record(root) != bindings["analysis_code"]):
            raise ValueError("Saved notebook or statistical source changed during execution")
    finally:
        # A separate file retains outputs or the failure location. The cleared
        # execution copy and all previous outputs remain unchanged.
        write_new(executed_path, notebook)
    print(f"Saved executed notebook: {executed_path}")
    return executed_path


RESULT_TYPE = "flat_notebook_analysis_results"


def notebook_mode(value):
    """Separate editable exploration from the final saved computation."""
    if value not in ("explore", "compute", "render"):
        raise ValueError("Choose explore, compute or render")
    return value


def model_labels(models):
    """Resolve hyphenated labels from saved model identities and native effort settings."""
    titles = {
        "openai/gpt-6.1-sol": "GPT-6.1 Sol", "openai/gpt-6-luna": "GPT-6 Luna",
        "anthropic/claude-sonnet-5.5": "Claude Sonnet 5.5",
        "google/gemini-3.8-flash": "Gemini 3.8 Flash",
        "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
        "qwen/qwen3.8-max-0902": "Qwen3.8 Max", "qwen/qwen3.8-27b": "Qwen3.8-27B",
        "mistralai/mistral-medium-3-5": "Mistral Medium 3.5",
        "z-ai/glm-5": "GLM-5", "moonshotai/kimi-k2.5": "Kimi K2.5",
        "meta-llama/llama-4-maverick": "Llama 4 Maverick",
        "typesafe/jev-1.13": "JEV 1.13", "openai/gpt-6-luna-decisions": "GPT-6 Luna Decisions",
    }
    from benchmarking.configuration_ids import canonical_configuration, CURRENT_NAMESPACE, CONFIGURATIONS
    decision_models = {c['model_id'] for cid, c in CONFIGURATIONS.items() if cid.startswith('D')}
    labels = {}
    for row in models:
        model, effort = row["model"], row["effort"]
        if row['config_id'].startswith(('G', 'D')) and model not in titles:
            title = canonical_configuration(CURRENT_NAMESPACE, row['config_id'], model_id=model)['display_name']
        else:
            title = titles.get(model, model)
        if model == "moonshotai/kimi-k2.5" and effort == "none":
            title += "-Instant"
        elif model not in decision_models:
            title += "-" + {"xhigh": "Xhigh", "native": "Native", "none": "None"}.get(effort, effort.title())
        title = "-".join(title.split())
        if row["config_id"] in labels or title in labels.values():
            raise ValueError("Model display labels must identify configurations uniquely")
        labels[row["config_id"]] = title
    return labels


def resource_integrity(manifest, terminal, *, imputation_policy=None, sonnet_policy=None, duration_policy=None):
    """Flag missing charges and durations before expensive numerical analysis.

    No values are imputed here. The approved numerical kernel owns assignments.
    Unknown missingness remains an explicit issue, including in repeat runs.
    """
    from collections import defaultdict
    from math import isfinite
    from analysis.methods.pipeline import (COST_IMPUTATION_ITEMS, COST_IMPUTATION_RUN, COST_IMPUTATION_POLICY,
        SONNET_COST_POLICY, SONNET_COST_RUN, SONNET_COST_ITEM, KIMI_DURATION_POLICY,
        KIMI_DURATION_RUN, KIMI_DURATION_ITEM)
    if manifest.get('execution_workflow') == 'analysis-reporting-ids-v1':
        return expanded_resource_integrity(manifest, terminal)
    groups = defaultdict(list)
    for row in terminal:
        groups[row["identity"]["run_id"]].append(row)
    report = []
    for run in manifest["runs"]:
        rows = groups[run["run_id"]]
        missing, missing_duration, missing_duration_items = [], 0, []
        for row in rows:
            amount = row["billing"]["observed_usd"]["value"]
            if amount is None:
                missing.append(row["identity"]["item_id"])
            elif not isfinite(float(amount)) or float(amount) < 0:
                raise ValueError("Invalid observed charge in " + run["run_id"])
            calls = [a["record"] for a in row.get("analysis_attempts", [])] or [row]
            missing_duration_items += [call.get("identity", row["identity"])["item_id"] for call in calls
                                       if call["timing"]["duration_seconds"]["value"] is None]
            missing_duration += sum(call["timing"]["duration_seconds"]["value"] is None for call in calls)
        approved = (imputation_policy == COST_IMPUTATION_POLICY and run["run_id"] == COST_IMPUTATION_RUN
                    and len(missing) == 2 and set(missing) == COST_IMPUTATION_ITEMS)
        approved |= (sonnet_policy == SONNET_COST_POLICY and run["run_id"] == SONNET_COST_RUN
                     and missing == [SONNET_COST_ITEM])
        duration_approved = (duration_policy == KIMI_DURATION_POLICY and run["run_id"] == KIMI_DURATION_RUN
                             and missing_duration_items == [KIMI_DURATION_ITEM])
        repeat_unresolved = run["role"] == "secondary" and run["config_id"] == "D01" and bool(missing)
        report.append({"configuration": run["config_id"], "run": run["replicate"],
                       "role": run["role"], "questions": len(rows),
                       "missing final-attempt charges": len(missing),
                       "missing call durations": missing_duration,
                       "duration handling": "Approved mean imputation" if duration_approved else
                           "Unresolved; duration comparison unavailable" if missing_duration else "Observed durations complete",
                       "cost handling": "Unresolved; secondary cost not analyzed" if repeat_unresolved else
                           "Approved mean imputation" if approved else
                           "Unresolved; cost comparison unavailable" if missing else "Observed billing complete",
                       "missing-charge questions": missing})
    return report


def fresh_output_directory(value, root):
    """Validate a new result/presentation destination without creating it."""
    if value is None or not str(value).strip():
        raise ValueError("Supply a new output directory beneath analysis/output")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(root)/path
    path = path.resolve()
    base = (Path(root)/"analysis/output").resolve()
    if path == base or not path.is_relative_to(base) or path.exists():
        raise ValueError("Output directory must be new and beneath analysis/output")
    return path


def figure_data(summary):
    """Freeze the data for every figure during computation; no styling or drawing."""
    data = {}
    rows = summary["primary"]
    accuracy = [{"configuration": cid, "accuracy": r["accuracy"], "wilson95": r["wilson95"]}
                for cid, r in rows.items() if r["accuracy"] is not None]
    if accuracy:
        data["accuracy"] = accuracy
    families = summary.get("paired", {}).get("families", {})
    for name, contrasts in families.items():
        if contrasts:
            data[name.lower()+"_paired_effects"] = contrasts
    cost_basis = "analysis_usd" if "analysis_usd" in summary["frontiers"] else "observed_usd"
    for name, basis in (("accuracy_cost", cost_basis),
                        ("accuracy_estimated_cost", "estimated_usd"),
                        ("accuracy_latency", None)):
        frontier = (summary["frontiers"][basis]["accuracy_cost"] if basis
                    else summary["frontiers"]["accuracy_latency"])
        points = []
        for cid, row in rows.items():
            if cid in frontier["excluded"]:
                continue
            x = (row["operations"][basis]["mean_per_attempt"] if basis
                 else row["operations"].get("analysis_latency", row["operations"]["latency"])["median"])
            points.append({"configuration": cid, "x": x, "accuracy": row["accuracy"],
                           "wilson95": row["wilson95"], "nondominated": cid in frontier["nondominated"]})
        available = {p["configuration"] for p in points}
        connections = [{"lower": p["negative_configuration"], "upper": p["positive_configuration"],
                        "label": f"{p['negative_configuration']} {p['negative_effort']} → {p['positive_configuration']} {p['positive_effort']}"}
                       for p in families.get("S2", [])
                       if {p["negative_configuration"], p["positive_configuration"]} <= available]
        if points:
            data[name] = {"points": points, "effort_connections": connections}
            if basis == "analysis_usd":
                data[name]["cost_basis"] = "observed billing plus approved within-run mean imputation"
    models = summary.get("decision_models")
    if models is None:
        models = {"J01": {"confidence": summary.get("confidence", {}), "stability": summary["stability"]}}
    for cid, diagnostics in models.items():
        prefix = "jev" if cid == "J01" else cid.lower()
        conf, repeat = diagnostics["confidence"], diagnostics["stability"]
        if conf.get("risk_coverage"):
            data[prefix+"_risk_coverage"] = conf["risk_coverage"]
        if conf.get("reliability") is not None:
            data[prefix+"_reliability"] = conf["reliability"]
        if "runs" in repeat:
            scores = [{"replicate": r, "accuracy": v["accuracy"], "wilson95": v.get("wilson95")}
                      for r, v in repeat["runs"].items() if v["accuracy"] is not None]
            if scores:
                data[prefix+"_stability_accuracy"] = scores
            agreements = [{k: p[k] for k in ("earlier_run", "later_run", "valid_pair_n", "answer_agreement")}
                          for p in repeat["pairs"] if p["answer_agreement"] is not None]
            if agreements:
                data[prefix+"_stability_agreement"] = agreements
    flows = [{"configuration": cid, **diagnostics["stability"]["answer_flows"]}
             for cid, diagnostics in models.items() if "answer_flows" in diagnostics["stability"]]
    if flows:
        data["repeatability_sankey"] = flows
    for metric, name in (("selected_probability_change", "repeatability_confidence"),
                         ("total_variation", "repeatability_vector")):
        rows = []
        for cid, diagnostics in models.items():
            for pair in diagnostics["stability"].get("pairs", []):
                distribution = (pair.get(metric) if metric == "selected_probability_change" else
                                pair.get("confidence_change_distribution", {}).get(metric))
                if distribution and distribution["eligible_n"]:
                    rows.append({"configuration": cid, "earlier_run": pair["earlier_run"],
                                 "later_run": pair["later_run"], **distribution})
        if rows:
            data[name] = rows
    return data


def save_results(folder, summary, review_tables, bindings, approval):
    """Save one complete numerical bundle and its checksum in a new directory."""
    import json
    import re
    if (not isinstance(bindings, dict) or not bindings or not isinstance(approval, dict)
            or not isinstance(approval.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", approval["sha256"])):
        raise ValueError("Numerical results require statistical-release bindings and an approved release digest")
    if folder is None:
        raise ValueError("Supply JEV_RESULT_DIR for a new numerical-result directory")
    if not summary.get("study_complete") or not summary.get("primary_complete"):
        raise ValueError("Cannot save a definitive result from incomplete coverage")
    result = {"schema_version": 1, "type": RESULT_TYPE, "created_at": utc_now(),
              "statistical_release": {"bindings": bindings, "approval": approval},
              "summary": summary, "review_tables": review_tables, "figure_data": figure_data(summary)}
    result = active_report_view(result)
    # Check serialization before creating the destination. Missing values remain null.
    json.dumps(result, allow_nan=False)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    path = folder/"results.json"
    write_new(path, result)
    digest = file_hash(path)
    write_new(folder/"result_manifest.json", {"schema_version": 1, "results_file": path.name,
                                             "results_sha256": digest, "statistical_release_sha256": approval["sha256"]})
    return result, path, digest


def load_results(path, expected_sha256):
    """Authenticate original bytes, then project the active report without recalculation."""
    import hashlib
    import re
    if path is None or not str(path).strip() or not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Render requires an explicitly selected RESULT_JSON and independently retained RESULT_SHA256; set both in the settings cell or via JEV_RESULT_JSON/JEV_RESULT_SHA256")
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Saved results differ from their expected SHA-256")
    result = strict_json(payload.decode("utf-8"))
    if (not isinstance(result, dict) or result.get("schema_version") != 1 or result.get("type") != RESULT_TYPE
            or not isinstance(result.get("summary"), dict) or not isinstance(result.get("review_tables"), dict)
            or not isinstance(result.get("figure_data"), dict) or not isinstance(result.get("statistical_release"), dict)):
        raise ValueError("Unsupported or incomplete numerical-result bundle")
    if not result["summary"].get("study_complete") or not result["summary"].get("primary_complete"):
        raise ValueError("Saved definitive results require complete study coverage")
    release = result["statistical_release"]
    if (not isinstance(release.get("bindings"), dict) or not release["bindings"]
            or not isinstance(release.get("approval"), dict)
            or not isinstance(release["approval"].get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", release["approval"]["sha256"])):
        raise ValueError("Saved results require statistical-release bindings and an approved release digest")
    return active_report_view(result)


def render_result_info(result, path, digest, *, sap_version, methods):
    """Identify the authenticated saved release and differences from working settings."""
    bindings = result["statistical_release"]["bindings"]
    saved_version = bindings.get("sap", {}).get("version")
    saved_methods = bindings.get("settings", {}).get("methods")
    differences = (sorted(key for key in set(saved_methods)|set(methods)
                          if saved_methods.get(key) != methods.get(key))
                   if isinstance(saved_methods, dict) else None)
    return {"file": str(path), "created_at": result.get("created_at"),
            "study_id": result["summary"].get("study_id"), "sha256": digest,
            "saved_sap_version": saved_version, "working_sap_version": sap_version,
            "sap_matches": saved_version == sap_version if saved_version is not None else None,
            "different_method_settings": differences}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the approved saved analysis notebook in a fresh kernel")
    parser.add_argument("--compute-notebook", required=True, help="Cleared copy beneath analysis/output")
    run_compute_notebook(parser.parse_args().compute_notebook)


def expanded_resource_integrity(manifest, terminal):
    """Pre-computation report uses exactly the selected resource rows."""
    from collections import defaultdict
    from fractions import Fraction
    groups = defaultdict(list)
    for row in terminal:
        groups[row['identity']['run_id']].append(row)
    report = []
    for run in manifest['runs']:
        rows = groups[run['run_id']]
        result = {'configuration': run['config_id'], 'run': run['replicate'], 'role': run['role'],
                  'questions': len(rows), 'missing outcomes': len(manifest['items'])-len(rows)}
        for label, section, field in [('cost', 'billing', 'observed_usd'), ('duration', 'timing', 'duration_seconds')]:
            values = [r[section][field]['value'] for r in rows]
            donors = [Fraction(str(v)) for v in values if v is not None]
            if any(v < 0 for v in donors):
                raise ValueError('Invalid resource donor')
            missing = len(values)-len(donors)
            result['missing final-attempt charges' if label == 'cost' else 'missing call durations'] = missing
            result[label+' donor n'] = len(donors)
            result[label+' handling'] = ('Observed only; missing values unavailable' if run['role'] == 'secondary'
                else 'No observed donors; unavailable' if not donors
                else 'Within-configuration primary mean imputation' if missing else 'Observed complete')
        report.append(result)
    return report
