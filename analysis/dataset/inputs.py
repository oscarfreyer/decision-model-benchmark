"""Portable local input inventory. No network, credentials or private path map.

Create the inventory at the authoritative dataset handoff, after collection has
stopped. Its independently retained SHA-256 authenticates the inventory; each
entry binds the exact relative filename, byte size and file SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

MANIFEST = "input_manifest.json"
FORBIDDEN = {".git", ".agents", ".codex", ".ssh", ".aws", ".venv", ".venv-analysis",
             "__pycache__", ".ipynb_checkpoints", ".netrc", ".npmrc", ".pypirc",
             "credentials.json", "credentials.toml", "secrets.json", "secrets.toml",
             "id_rsa", "id_ed25519", "external-locations.local.json"}


def sha256_file(path):
    """Stream large raw files without loading their entire contents into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def local_path(root, relative):
    """Resolve a portable reference inside the input root, with no symlinks."""
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("Input references must be nonempty POSIX relative paths")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or any(p in ("", ".", "..") for p in relative.split("/")) or ":" in relative:
        raise ValueError("Input reference escapes the local input folder")
    root = Path(root)
    current = root
    if current.is_symlink():
        raise ValueError("Input root must not be a symlink")
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Input symlinks are not supported")
    if not current.resolve().is_relative_to(root.resolve()):
        raise ValueError("Input reference escapes the local input folder")
    return current


def folder_inventory(root):
    """Inventory every payload, rejecting hidden environments and symlinks."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("A local input directory is required")
    entries = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        local_path(root, relative)
        if any(p.lower() in FORBIDDEN or p.lower().startswith(".env") for p in path.relative_to(root).parts):
            raise ValueError("Input contains an excluded credential, environment or private registry path")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("Input payloads must be ordinary files")
        if relative == MANIFEST:
            continue
        entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return entries


def package_closure(root, packages):
    """Check continuation containment before the benchmark reader opens parents."""
    visited = set()

    def visit(relative, ancestors):
        path = local_path(root, relative)
        if relative in ancestors:
            raise ValueError("Continuation parent cycle")
        if relative in visited:
            return
        visited.add(relative)
        data = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if data.get("execution_workflow") not in {"lean-v1", "lean-decisions-v1"}:
            raise ValueError("Portable notebook inputs require current saved benchmark packages")
        if data.get("continuation"):
            parent = data["continuation"]["parent"]
            if not isinstance(parent, str) or Path(parent).is_absolute() or "\\" in parent:
                raise ValueError("Continuation parent must be relative")
            resolved = (path / parent).resolve()
            if not resolved.is_relative_to(Path(root).resolve()):
                raise ValueError("Continuation parent escapes input folder")
            visit(resolved.relative_to(Path(root).resolve()).as_posix(), ancestors | {relative})

    if not isinstance(packages, list) or not packages or len(set(packages)) != len(packages):
        raise ValueError("Select distinct explicit package paths")
    for relative in packages:
        visit(relative, set())
    return sorted(visited)


def create_inventory(root, packages, questions, keys):
    """Bind an already assembled stable folder; never copy or edit raw files."""
    root = Path(root)
    target = root / MANIFEST
    if target.exists():
        raise ValueError("An input inventory already exists; use a new dataset directory")
    for relative in (questions, keys):
        if not local_path(root, relative).is_file():
            raise ValueError("Prepared questions and original keys are required")
    before = folder_inventory(root)
    closure = package_closure(root, packages)
    # Import the same pure consumer used by the notebook. No collection code runs.
    from analysis.methods.pipeline import load_study
    manifest, terminal, coverage, _ = load_study(
        [local_path(root, p) for p in packages], local_path(root, questions), local_path(root, keys))
    check_complete_study(manifest, terminal, coverage)
    if before != folder_inventory(root):
        raise ValueError("Inputs changed during inventory creation")
    payload = {"schema_version": 1, "type": "analysis_input_inventory",
               "packages": packages, "package_closure": closure,
               "questions": questions, "keys": keys, "files": before}
    with target.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2)
        stream.write("\n")
    return sha256_file(target)


def verify_inventory(root, expected_sha256):
    """Require the source handoff digest; a local self-check is insufficient."""
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("Set INPUT_MANIFEST_SHA256 from the authoritative dataset handoff")
    root = Path(root)
    path = local_path(root, MANIFEST)
    if sha256_file(path) != expected_sha256:
        raise ValueError("Input inventory differs from the authoritative handoff SHA-256")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") == 2 and data.get("type") == "flat_analysis_input":
        from analysis.dataset.flat import FILES
        if set(data) != {"schema_version", "type", "files", "counts", "raw_inventory_sha256"}:
            raise ValueError("Unsupported flat input inventory")
        if {p.name for p in root.iterdir()} != set(FILES) or any(not p.is_file() for p in root.iterdir()):
            raise ValueError("Flat input requires exactly six files and no subfolders")
        if data["files"] != folder_inventory(root):
            raise ValueError("Input files are missing, unexpected or changed relative to the source inventory")
        provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
        if (provenance.get("raw_inventory_sha256") != data["raw_inventory_sha256"]
                or provenance.get("counts") != data["counts"]):
            raise ValueError("Flat inventory/provenance bindings disagree")
        return data
    if set(data) != {"schema_version", "type", "packages", "package_closure", "questions", "keys", "files"} or data["schema_version"] != 1 or data["type"] != "analysis_input_inventory":
        raise ValueError("Unsupported input inventory")
    actual = folder_inventory(root)
    if data["files"] != actual:
        raise ValueError("Input files are missing, unexpected or changed relative to the source inventory")
    if package_closure(root, data["packages"]) != data["package_closure"]:
        raise ValueError("Continuation closure differs from the source inventory")
    for relative in (data["questions"], data["keys"]):
        if relative not in {e["path"] for e in actual}:
            raise ValueError("Prepared questions/keys absent from source inventory")
    return data


def check_complete_study(manifest, terminal, coverage):
    """Require the full question/configuration/replicate design, not just files."""
    from analysis.methods.core import CORE_ROSTER
    if len(manifest["items"]) != 1273 or {c["config_id"] for c in manifest["configurations"]} != CORE_ROSTER:
        raise ValueError("Complete study requires 1,273 questions and all 17 configurations, including D01")
    expected = {(cid, "E1", 1, "primary") for cid in CORE_ROSTER}
    expected |= {(cid, "E2", r, "secondary") for cid in ("J01", "D01") for r in (2, 3)}
    runs = [(r["config_id"], r["experiment_id"], r["replicate"], r["role"]) for r in manifest["runs"]]
    if set(runs) != expected or len(runs) != len(expected):
        raise ValueError("Complete study requires one primary run per configuration and repeats 2–3 for both decision models")
    if (not coverage["complete"] or coverage["unstarted"] or coverage["indeterminate"]
            or coverage["expected"] != 26733 or coverage["terminal"] != 26733 or len(terminal) != 26733):
        raise ValueError("Complete study requires 26,733 terminal attempts with no missing or unresolved identities")


def load_input_folder(root, expected_sha256):
    """Validate bytes, saved execution evidence and complete study coverage."""
    from analysis.support.notebook import load_review_inputs
    data = verify_inventory(root, expected_sha256)
    if data["schema_version"] == 2:
        from analysis.dataset.flat import verified_flat_records
        manifest, terminal, coverage, keys, questions = verified_flat_records(root)
        check_complete_study(manifest, terminal, coverage)
        if data != verify_inventory(root, expected_sha256):
            raise ValueError("Inputs changed during loading")
        hashes = {entry["path"]: entry["sha256"] for entry in data["files"]}
        hashes[MANIFEST] = expected_sha256
        return manifest, terminal, coverage, keys, questions, hashes
    loaded = load_review_inputs([local_path(root, p) for p in data["packages"]],
                                local_path(root, data["questions"]), local_path(root, data["keys"]),
                                input_root=root)
    check_complete_study(*loaded[:3])
    if data != verify_inventory(root, expected_sha256):
        raise ValueError("Inputs changed during loading")
    manifest, terminal, coverage, keys, questions, _ = loaded
    hashes = {entry["path"]: entry["sha256"] for entry in data["files"]}
    hashes[MANIFEST] = expected_sha256
    return manifest, terminal, coverage, keys, questions, hashes


def main():
    parser = argparse.ArgumentParser(description="Bind a complete, already assembled stable analysis input folder")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--packages", nargs="+", required=True, help="Relative latest package paths; retain continuation parents")
    parser.add_argument("--questions", required=True, help="Relative prepared questions JSONL")
    parser.add_argument("--keys", required=True, help="Relative original keys JSONL")
    args = parser.parse_args()
    print(create_inventory(args.root, args.packages, args.questions, args.keys))


if __name__ == "__main__":
    main()
