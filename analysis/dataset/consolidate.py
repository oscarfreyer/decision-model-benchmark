"""Create a new six-file flat dataset from an independently bound raw snapshot."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

from benchmarking.common import attempt_identity, fingerprint, strict_json
from analysis.dataset.inputs import folder_inventory, local_path, sha256_file, check_complete_study
from analysis.dataset.flat import (EXPECTED_COUNTS, FILES, TYPE, VERSION, RawEvidence,
    attempt_id, collection_counts, encode_bytes, read_collection, select_attempts,
    verified_flat_records)

DEFAULT_RETRY = "benchmarking/output/failure-retries-20261007"
DEFAULT_QUESTIONS = "preparation/output/questions/medqa-full-evaluation.jsonl"
DEFAULT_KEYS = "preparation/output/answers/medqa-full-evaluation-keys.jsonl"
SOURCE_FILES = ("analysis/consolidate.py", "analysis/dataset/consolidate.py",
    "analysis/dataset/flat.py", "analysis/dataset/inputs.py", "analysis/methods/core.py",
    "analysis/__init__.py", "analysis/dataset/__init__.py", "analysis/methods/__init__.py",
    "common.py", "benchmarking/__init__.py", "benchmarking/common.py", "benchmarking/retry_common.py")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2)
        stream.write("\n")


def verify_raw_snapshot(snapshot, digest):
    """Verify all inventoried bytes, including development and preparation evidence."""
    inventory = local_path(snapshot, "raw_inventory.json").read_bytes()
    evidence = RawEvidence(inventory, digest)
    payload = local_path(snapshot, "payload")
    names = sorted(path.relative_to(payload).as_posix()
                   for path in payload.rglob("*") if path.is_file())
    if names != sorted(evidence.files):
        raise ValueError("Raw snapshot has missing or unexpected files")
    # Path ordering is component-wise; frozen inventories sort POSIX strings.
    actual = sorted(folder_inventory(payload), key=lambda row: row["path"])
    if actual != sorted(evidence.inventory["files"], key=lambda row: row["path"]):
        raise ValueError("Raw snapshot has missing, changed or unexpected files")
    tool = evidence.inventory["assembly_tool"]
    if sha256_file(local_path(snapshot, tool["path"])) != tool["sha256"]:
        raise ValueError("Raw snapshot assembly utility changed")
    return evidence, inventory, payload


def catalog_attempts(payload, packages, manifest, retry, evidence, stream):
    """Stream lossless artifacts and source journal lines; keep selection metadata."""
    rows = []
    journals = {}
    runs = {r["run_id"]: r for r in manifest["runs"]}
    targets = {t["target_id"]: t["identity"] for t in retry["targets"]}
    for package in packages:
        path = local_path(payload, package + "/events.jsonl")
        data = evidence.check(package + "/events.jsonl", path.read_bytes())
        starts = {}
        other = []
        lines = data.splitlines(keepends=True)
        for number, line in enumerate(lines):
            event = strict_json(line.decode("utf-8"))
            ref = {"line": number, "event_sha256": event["event_sha256"],
                   "bytes_base64": encode_bytes(line)}
            kind = event["event"]
            index = event.get("retry_index", 0)
            if kind not in {"start", "finish"}:
                other.append(ref)
                continue
            ident = (attempt_identity(targets[event["target_id"]]) if index
                     else attempt_identity(event))
            key = (ident, index)
            if kind == "start":
                if key in starts:
                    raise ValueError("Duplicate source start")
                starts[key] = ref
                continue
            if key not in starts:
                raise ValueError("Finish has no original start")
            record = event["record"]
            row = {"attempt_id": attempt_id(record, index), "retry_index": index,
                "target_id": fingerprint(list(ident)),
                "campaign_id": retry["campaign_id"] if index else None,
                "source_package": package, "run": runs[ident[-1]], "record": record}
            artifacts = {}
            for name, artifact in record["artifacts"].items():
                artifacts[name] = (None if artifact is None else encode_bytes(evidence.check(
                    package + "/" + artifact["path"],
                    local_path(payload, package + "/" + artifact["path"]).read_bytes())))
            full = {**row, "events": {"start": starts.pop(key), "finish": ref},
                    "artifacts": artifacts}
            stream.write(json.dumps(full, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            rows.append(row)
        if starts:
            raise ValueError("Indeterminate source starts")
        journals[package] = {"line_count": len(lines), "other_events": other}
    return rows, journals


def publish(candidate, destination):
    """Reserve a new directory exclusively; publish the completion inventory last."""
    destination.mkdir(exist_ok=False)
    created = []
    try:
        for name in FILES:
            target = destination / name
            with target.open("xb") as output:
                created.append(target)
                with (candidate / name).open("rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if sha256_file(target) != sha256_file(candidate / name):
                raise ValueError("Published file differs from verified candidate")
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        # Unexpected operator files are never removed.
        try:
            destination.rmdir()
        except OSError:
            pass
        raise


def consolidate(snapshot, digest, destination, *, retry_package=DEFAULT_RETRY,
                questions=DEFAULT_QUESTIONS, keys=DEFAULT_KEYS, allow_fictional=False):
    snapshot = Path(snapshot).resolve()
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("Use a fresh destination; existing output is never replaced")
    if (destination.resolve().is_relative_to(snapshot)
            or snapshot.is_relative_to(destination.resolve())):
        raise ValueError("Destination must be separate from the raw snapshot")
    if not destination.parent.is_dir():
        raise ValueError("Destination parent must already exist")
    print("Verifying frozen raw inventory and every payload", flush=True)
    evidence, inventory, payload = verify_raw_snapshot(snapshot, digest)
    manifest, coverage, retry, _, _, packages = read_collection(payload, retry_package)
    if manifest["provenance"] != "definitive" and not allow_fictional:
        raise ValueError("Real consolidation requires definitive source packages")
    if manifest["provenance"] == "definitive":
        check_complete_study(manifest, [None] * coverage["terminal"], coverage)
    before_code = {name: sha256_file(Path(__file__).resolve().parents[2] / name)
                   for name in SOURCE_FILES}
    with tempfile.TemporaryDirectory(prefix="jev-flat-build-", dir=destination.parent) as folder:
        candidate = Path(folder)
        print("Writing complete attempt evidence", flush=True)
        with (candidate / "all_attempts.jsonl").open("x", encoding="utf-8") as stream:
            rows, journals = catalog_attempts(payload, packages, manifest, retry, evidence, stream)
        selected = select_attempts(rows)
        counts = collection_counts(rows, selected)
        if manifest["provenance"] == "definitive" and counts != EXPECTED_COUNTS:
            raise ValueError("Accepted collection counts differ")
        with (candidate / "selected_responses.jsonl").open("x", encoding="utf-8") as stream:
            for row in selected:
                stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n")
        # Copy exact key bytes only after selection, without parsing them.
        for source, name, binding in ((questions, "questions.jsonl", "questions"),
                                      (keys, "keys.jsonl", "keys")):
            data = evidence.check(source, local_path(payload, source).read_bytes())
            if sha256_file(local_path(payload, source)) != manifest["bindings"][binding]:
                raise ValueError("Prepared input binding mismatch")
            (candidate / name).write_bytes(data)
        package_files = {}
        row_artifacts = {row["source_package"] + "/" + ref["path"]
                         for row in rows for ref in row["record"]["artifacts"].values()
                         if ref is not None}
        for package in packages:
            names = [name for name in evidence.files if name.startswith(package + "/")
                     and name != package + "/events.jsonl" and name not in row_artifacts]
            package_files[package] = {name[len(package)+1:]: encode_bytes(evidence.check(
                name, local_path(payload, name).read_bytes())) for name in names}
        preparation = {name: encode_bytes(evidence.check(name, local_path(payload, name).read_bytes()))
            for name in evidence.files if name.startswith("preparation/output/")
            and name.endswith(".json")}
        provenance = {"schema_version": VERSION, "type": TYPE,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "raw_inventory_sha256": digest, "raw_inventory_base64": encode_bytes(inventory),
            "retry_package": retry_package, "manifest": manifest, "coverage": coverage,
            "counts": counts, "package_files": package_files, "journals": journals,
            "preparation_evidence": preparation,
            "consolidation": {"source_sha256": before_code,
                "python": platform.python_version(), "platform": platform.platform(),
                "settings": {"retry_package": retry_package, "questions": questions, "keys": keys},
                "selection": "valid_original_else_first_valid_retry_else_exhausted_last_terminal",
                "cost_basis": "final_attempt_per_original_query",
                "duration_basis": "all_attempts",
                "restoration": "relative package paths beneath disposable local root",
                "excluded_flat_payloads": "development smokes, study registries, source ZIP/source splits; retained in raw snapshot"}}
        write_json(candidate / "provenance.json", provenance)
        print("Re-reading flat evidence and independently reconciling selection", flush=True)
        verified_flat_records(candidate)
        # Reject source/code changes during assembly; all inputs are read-only.
        if (sha256_file(snapshot / "raw_inventory.json") != digest
                or sorted(folder_inventory(payload), key=lambda row: row["path"])
                != sorted(evidence.inventory["files"], key=lambda row: row["path"])):
            raise ValueError("Raw snapshot changed during consolidation")
        if before_code != {name: sha256_file(Path(__file__).resolve().parents[2] / name) for name in SOURCE_FILES}:
            raise ValueError("Consolidation sources changed during assembly")
        write_json(candidate / "input_manifest.json", {"schema_version": 2,
            "type": TYPE, "files": folder_inventory(candidate), "counts": counts,
            "raw_inventory_sha256": digest})
        expected = sha256_file(candidate / "input_manifest.json")
        publish(candidate, destination)
        if folder_inventory(destination) != folder_inventory(candidate) or sha256_file(destination / "input_manifest.json") != expected:
            raise ValueError("Published dataset differs from verified candidate")
    return {"destination": str(destination), "input_manifest_sha256": expected,
            "raw_inventory_sha256": digest, "counts": counts, "files": list(FILES)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--raw-inventory-sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(consolidate(args.snapshot, args.raw_inventory_sha256, args.destination), indent=2))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print("Consolidation blocked: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
