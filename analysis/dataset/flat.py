"""Six-file evidence format and read-only reconstruction of saved study packages.

Embedded bytes are data. Only the installed pure package readers execute.
Selection runs before reference keys are loaded; no statistical analysis runs here.
"""
from __future__ import annotations

import base64
from collections import defaultdict
import hashlib
from pathlib import Path
import tempfile

from benchmarking.common import (attempt_identity, fingerprint, load_keys,
    load_questions, question_hash, read_package, strict_json)
from benchmarking.retry_common import read_retry_package, source_context
from analysis.methods.core import join_studies
from analysis.dataset.inputs import folder_inventory, local_path, package_closure, sha256_file

FILES = ("questions.jsonl", "keys.jsonl", "all_attempts.jsonl",
         "selected_responses.jsonl", "provenance.json", "input_manifest.json")
TYPE = "flat_analysis_input"
VERSION = 1
EXPECTED_COUNTS = {"original_attempts": 26733, "retry_attempts": 533,
    "all_attempts": 27266, "retry_targets": 223, "recovered_targets": 204,
    "exhausted_targets": 19, "selected_outcomes": 26733, "selected_valid": 26714,
    "primary_queries": 21641, "secondary_queries": 5092}


def encode_bytes(data):
    """Keep exact whitespace, line endings and arbitrary binary content."""
    return base64.b64encode(data).decode("ascii")


def decode_bytes(value):
    if not isinstance(value, str):
        raise ValueError("Embedded bytes must be base64 text")
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Invalid embedded base64 bytes") from exc


def attempt_id(record, retry_index):
    return fingerprint([*attempt_identity(record), retry_index])


def select_attempts(rows):
    """Select by validity/index only, without keys or correctness fields.

    Retry index zero is the original. Valid originals cannot have retries. A
    failed original requires contiguous retries through first valid or index 10.
    """
    grouped = defaultdict(list)
    seen = set()
    for row in rows:
        record = row["record"]
        index = row["retry_index"]
        if type(index) is not int or not 0 <= index <= 10:
            raise ValueError("Invalid retry index")
        ident = attempt_identity(record)
        aid = attempt_id(record, index)
        if aid != row["attempt_id"] or aid in seen:
            raise ValueError("Duplicate or inconsistent attempt identity")
        if row["target_id"] != fingerprint(list(ident)):
            raise ValueError("Original target identity mismatch")
        seen.add(aid)
        grouped[ident].append(row)
    selected = []
    for ident, trials in sorted(grouped.items()):
        trials.sort(key=lambda row: row["retry_index"])
        if [row["retry_index"] for row in trials] != list(range(len(trials))):
            raise ValueError("Missing original or noncontiguous retries")
        original = trials[0]
        if any(row["run"] != original["run"] for row in trials):
            raise ValueError("Retries changed primary/repeat role")
        valid = [row for row in trials if row["record"]["status"] == "valid"]
        if valid:
            final = valid[0]
            if final is not trials[-1]:
                raise ValueError("Retry after first valid response")
            outcome = "original_valid" if final["retry_index"] == 0 else "recovered"
        else:
            if trials[-1]["retry_index"] != 10:
                raise ValueError("Incomplete recovery: ten retries required for exhaustion")
            final = trials[-1]
            outcome = "exhausted"
        selected.append({"identity": original["record"]["identity"],
            "target_id": original["target_id"], "run": original["run"],
            "outcome": outcome, "original_attempt_id": original["attempt_id"],
            "selected_attempt_id": final["attempt_id"],
            "cost_attempt_id": final["attempt_id"],
            "duration_attempt_ids": [row["attempt_id"] for row in trials],
            "selected_retry_index": final["retry_index"],
            "status": final["record"]["status"],
            "selected_label": final["record"]["selected_label"]})
    return selected


def collection_counts(rows, selected):
    originals = [r for r in rows if r["retry_index"] == 0]
    return {"original_attempts": len(originals), "retry_attempts": len(rows)-len(originals),
        "all_attempts": len(rows),
        "retry_targets": sum(r["record"]["status"] != "valid" for r in originals),
        "recovered_targets": sum(s["outcome"] == "recovered" for s in selected),
        "exhausted_targets": sum(s["outcome"] == "exhausted" for s in selected),
        "selected_outcomes": len(selected),
        "selected_valid": sum(s["status"] == "valid" for s in selected),
        "primary_queries": sum(s["run"]["role"] == "primary" for s in selected),
        "secondary_queries": sum(s["run"]["role"] == "secondary" for s in selected)}


def read_collection(payload, retry_relative):
    """Use original execution readers; retain continuation/source containment."""
    retry_path = local_path(payload, retry_relative)
    retry_header = strict_json((retry_path / "manifest.json").read_text())
    paths = []
    for source in retry_header["sources"]:
        if Path(source["path"]).is_absolute():
            raise ValueError("Retry source must be relative")
        resolved = (retry_path / source["path"]).resolve()
        if not resolved.is_relative_to(Path(payload).resolve()):
            raise ValueError("Retry source escapes snapshot")
        paths.append(resolved.relative_to(Path(payload).resolve()).as_posix())
    closure = package_closure(payload, paths)
    retry_manifest, retries, recovery = read_retry_package(retry_path)
    if (not recovery["complete"] or recovery["indeterminate"]
            or recovery["paused_providers"] or recovery["drift"]):
        raise ValueError("Complete recovery without indeterminate starts, pauses or drift required")
    loaded = [read_package(local_path(payload, p)) for p in paths]
    manifest, originals, coverage = join_studies(loaded, paths)
    if not coverage["complete"] or coverage["unstarted"] or coverage["indeterminate"]:
        raise ValueError("Complete original collection required")
    context, _, segments = source_context(retry_path, retry_manifest)
    if {fingerprint(list(attempt_identity(r))): r for r in originals} != context:
        raise ValueError("Original source closure differs from retry evidence")
    return manifest, coverage, retry_manifest, recovery, segments, closure + [retry_relative]


class RawEvidence:
    """Require every embedded file to match the independently bound raw inventory."""
    def __init__(self, inventory_bytes, expected_digest):
        if hashlib.sha256(inventory_bytes).hexdigest() != expected_digest:
            raise ValueError("Raw inventory differs from supplied SHA-256")
        self.inventory = strict_json(inventory_bytes.decode("utf-8"))
        if (self.inventory.get("schema_version") != 1
                or self.inventory.get("type") != "raw_collection_inventory"
                or self.inventory.get("payload_root") != "payload"):
            raise ValueError("Unsupported raw inventory")
        rows = self.inventory["files"]
        self.files = {row["path"]: row for row in rows}
        if len(rows) != len(self.files):
            raise ValueError("Duplicate raw inventory filename")

    def check(self, name, data):
        # Reject path traversal even when a malicious inventory supplies the name.
        local_path(Path(tempfile.gettempdir()) / "jev-path-check", name)
        entry = self.files.get(name)
        if (entry is None or len(data) != entry["bytes"]
                or hashlib.sha256(data).hexdigest() != entry["sha256"]):
            raise ValueError("Embedded source size/hash mismatch: " + name)
        return data

    def restore(self, root, name, data):
        data = self.check(name, data)
        path = local_path(root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError("Conflicting duplicate embedded source")
        else:
            with path.open("xb") as stream:
                stream.write(data)


def iter_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                raise ValueError("Flat JSONL cannot contain blank records")
            yield strict_json(line)


def restore_packages(root, evidence, provenance, attempts_path):
    """Reconstruct only study evidence in a disposable directory, never execute it."""
    journals = defaultdict(dict)
    for package, files in provenance["package_files"].items():
        for name, value in files.items():
            evidence.restore(root, package + "/" + name, decode_bytes(value))
    for package, journal in provenance["journals"].items():
        for event in journal["other_events"]:
            if event["line"] in journals[package]:
                raise ValueError("Duplicate journal line")
            data = decode_bytes(event["bytes_base64"])
            if strict_json(data.decode())["event_sha256"] != event["event_sha256"]:
                raise ValueError("Saved event digest differs from metadata")
            journals[package][event["line"]] = data
    rows = []
    for row in iter_jsonl(attempts_path):
        package = row["source_package"]
        if package not in provenance["journals"]:
            raise ValueError("Attempt source package absent from provenance")
        events = row["events"]
        for kind in ("start", "finish"):
            event = events[kind]
            line = event["line"]
            data = decode_bytes(event["bytes_base64"])
            if strict_json(data.decode())["event_sha256"] != event["event_sha256"]:
                raise ValueError("Saved event digest differs from metadata")
            if type(line) is not int or line < 0 or line in journals[package]:
                raise ValueError("Duplicate or invalid journal line")
            journals[package][line] = data
        start = strict_json(decode_bytes(events["start"]["bytes_base64"]).decode())
        finish = strict_json(decode_bytes(events["finish"]["bytes_base64"]).decode())
        if (start["event"] != "start" or finish["event"] != "finish"
                or finish["record"] != row["record"]
                or start["request"] != row["record"]["artifacts"]["request"]):
            raise ValueError("Embedded record differs from saved start/finish")
        if row["retry_index"]:
            if any(event.get("retry_index") != row["retry_index"]
                   or event.get("target_id") != row["target_id"] for event in (start, finish)):
                raise ValueError("Retry metadata differs from journal")
        elif "retry_index" in finish:
            raise ValueError("Original attempt mislabeled as retry")
        if set(row["artifacts"]) != set(row["record"]["artifacts"]):
            raise ValueError("Missing or unexpected embedded artifact")
        for kind, ref in row["record"]["artifacts"].items():
            value = row["artifacts"][kind]
            if ref is None:
                if value is not None:
                    raise ValueError("Absent artifact has embedded content")
            else:
                data = decode_bytes(value)
                if hashlib.sha256(data).hexdigest() != ref["sha256"]:
                    raise ValueError("Record artifact hash mismatch")
                evidence.restore(root, package + "/" + ref["path"], data)
        rows.append({k: v for k, v in row.items() if k not in {"artifacts", "events"}})
    for package, journal in provenance["journals"].items():
        events = journals[package]
        if sorted(events) != list(range(journal["line_count"])):
            raise ValueError("Missing journal lines")
        evidence.restore(root, package + "/events.jsonl", b"".join(events[i] for i in sorted(events)))
    for name, value in provenance["preparation_evidence"].items():
        evidence.check(name, decode_bytes(value))
    return rows


def verified_flat_records(root):
    """Verify source proofs and recompute selection, before loading any keys."""
    root = Path(root)
    provenance = strict_json((root / "provenance.json").read_text())
    if provenance.get("type") != TYPE or provenance.get("schema_version") != VERSION:
        raise ValueError("Unsupported flat provenance")
    evidence = RawEvidence(decode_bytes(provenance["raw_inventory_base64"]),
                           provenance["raw_inventory_sha256"])
    selected = list(iter_jsonl(root / "selected_responses.jsonl"))
    with tempfile.TemporaryDirectory(prefix="jev-flat-reader-") as folder:
        # Source readers resolve paths, including macOS /var -> /private/var.
        restored = Path(folder).resolve()
        rows = restore_packages(restored, evidence, provenance, root / "all_attempts.jsonl")
        expected_files = sorted((entry for name, entry in evidence.files.items()
            if any(name.startswith(package + "/") for package in provenance["package_files"])),
            key=lambda entry: entry["path"])
        if sorted(folder_inventory(restored), key=lambda entry: entry["path"]) != expected_files:
            raise ValueError("Embedded study evidence is missing, unexpected or changed")
        manifest, coverage, retry, recovery, segments, packages = read_collection(
            restored, provenance["retry_package"])
        if set(packages) != set(provenance["package_files"]):
            raise ValueError("Embedded package set differs from validated closure")
        if manifest != provenance["manifest"] or coverage != provenance["coverage"]:
            raise ValueError("Flat study plan/coverage differs from saved packages")
        campaign = retry["campaign_id"]
        for row in rows:
            ident = attempt_identity(row["record"])
            run = next(r for r in manifest["runs"] if r["run_id"] == ident[-1])
            tid = fingerprint(list(ident))
            expected_package = (provenance["retry_package"] if row["retry_index"]
                else segments[tid].relative_to(restored).as_posix())
            if (row["run"] != run or row["source_package"] != expected_package
                    or row["campaign_id"] != (campaign if row["retry_index"] else None)):
                raise ValueError("Attempt run/package/campaign metadata mismatch")
        computed = select_attempts(rows)
        if selected != computed:
            raise ValueError("Selected responses differ from first-valid/final-attempt rules")
        counts = collection_counts(rows, computed)
        if counts != provenance["counts"]:
            raise ValueError("Flat count reconciliation failed")
        if manifest["provenance"] == "definitive" and counts != EXPECTED_COUNTS:
            raise ValueError("Accepted complete collection counts differ")
        raw_counts = evidence.inventory["coverage"]
        if any(counts[key] != value for key, value in raw_counts.items() if key in counts):
            raise ValueError("Flat counts differ from frozen raw inventory")
        if counts["retry_attempts"] != recovery["terminal"]:
            raise ValueError("Flat retry count differs from verified campaign")
    questions = load_questions(root / "questions.jsonl")
    if (sha256_file(root / "questions.jsonl") != manifest["bindings"]["questions"]
            or sha256_file(root / "keys.jsonl") != manifest["bindings"]["keys"]
            or [(q.item_id, question_hash(q)) for q in questions]
            != [(q["item_id"], q["content_sha256"]) for q in manifest["items"]]):
        raise ValueError("Prepared question/key bytes or item order changed")
    # Keys are read only after selection is verified against saved evidence.
    keys = load_keys(root / "keys.jsonl", questions)
    by_id = {row["attempt_id"]: row for row in rows}
    conditions = {}
    for package, files in provenance["package_files"].items():
        saved = strict_json(decode_bytes(files["manifest.json"]).decode())
        conditions[package] = {"execution_workflow": saved["execution_workflow"],
                               "schedule": saved.get("schedule"),
                               "retry_policy": saved.get("policy")}
    terminal = []
    for outcome in computed:
        final = by_id[outcome["selected_attempt_id"]]["record"]
        original = by_id[outcome["original_attempt_id"]]["record"]
        terminal.append({**final, "analysis_selection": outcome,
            "original_attempt": original,
            "analysis_attempts": [{"record": by_id[aid]["record"],
                                   "retry_index": by_id[aid]["retry_index"],
                                   "source_package": by_id[aid]["source_package"]}
                                  for aid in outcome["duration_attempt_ids"]],
            "all_attempt_timings": [by_id[aid]["record"]["timing"]
                                    for aid in outcome["duration_attempt_ids"]]})
    return manifest, terminal, {**coverage, "flat_counts": counts,
                                "collection_conditions": conditions}, keys, questions
