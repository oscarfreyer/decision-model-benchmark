"""Verify preserved execution provenance without requiring its machine or registry.

Only locator resolution differs from inference: every supplied byte is still
checked against the original accepted inventory. This module cannot authorize
new calls, resume a run, edit manifests, or synthesize a review.
"""
from pathlib import Path
import re

from analysis.support.source_record import analysis_code_record
from common import (EXECUTION_FREEZE_ROLES, EXECUTION_REVIEW_TOPICS, EXECUTION_SOURCES,
                    execution_freeze_enabled, FREEZE_ROLES, REVIEW_TOPICS, ROOT, PRODUCTION_SOURCES,
                    config_binding, environment_record, file_hash, fingerprint,
                    load_questions, runtime_blockers, strict_json,
                    validate_manifest, validate_manifest_config)


class ArtifactResolver:
    """Optional checksum → local path map, rooted at the map's own directory."""
    def __init__(self, map_path=None):
        self.mapping = {}
        self.base = Path.cwd()
        self.map_sha256 = None
        if map_path is not None:
            map_path = Path(map_path)
            data = strict_json(map_path.read_text())
            if set(data) != {"schema_version", "artifacts"} or data["schema_version"] != 1 or not isinstance(data["artifacts"], dict):
                raise ValueError("Artifact map requires schema_version 1 and artifacts {sha256: local_path}")
            for digest, path in data["artifacts"].items():
                if not re.fullmatch(r"[0-9a-f]{64}", digest) or not isinstance(path, str) or not path.strip():
                    raise ValueError("Invalid artifact map checksum/path")
            self.mapping = data["artifacts"]
            self.base = map_path.resolve().parent
            self.map_sha256 = file_hash(map_path)

    def resolve(self, original_path, checksum, *, parent=None):
        if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
            raise ValueError("Explicit SHA-256 required for preserved evidence")
        if checksum in self.mapping:
            path = self.base / self.mapping[checksum]
        else:
            path = Path(parent or self.base) / original_path
        if file_hash(path) != checksum:
            raise ValueError("Preserved evidence checksum mismatch: " + str(original_path))
        return path


def reproduction_environment(original):
    """Same Python language version and pinned scientific dependency closure.

    Platform, Python patch and unrelated distributions are recorded, not used as
    evidence that an accepted original environment has changed. Source hashes
    independently bind the exact dependency declarations used for reproduction.
    """
    current = environment_record()
    packages = {}
    lock = ROOT / "requirements.lock"
    if not lock.is_file():
        raise ValueError("Pinned requirements.lock required for definitive reproduction")
    for line in lock.read_text().splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line.strip())
        if match:
            packages[match[1].lower().replace("_", "-")] = match[2]
        elif line.strip() and not line.lstrip().startswith('#'):
            raise ValueError("Unsupported dependency lock entry")
    if not {'numpy','scipy','matplotlib'}.issubset(packages):
        raise ValueError("Scientific dependency lock incomplete")
    normalize = lambda env: {k.lower().replace("_", "-"): v for k, v in env["packages"].items()}
    old, new = normalize(original), normalize(current)
    if original["python"].split(".")[:2] != current["python"].split(".")[:2]:
        raise ValueError("Reproduction requires the frozen Python major.minor version")
    for name in sorted(packages):
        if old.get(name) != packages[name] or new.get(name) != packages[name]:
            raise ValueError("Reproduction dependency differs from frozen version: " + name)
    return {"schema_version": 1, "original_environment_sha256": fingerprint(original),
            "reproduction_environment": current, "checked_packages": sorted(packages),
            "compatibility": "same Python major.minor and pinned dependency versions; OS, paths, Python patch and unrelated packages may differ"}


def verify_execution_provenance(manifest, config, *, freeze_path, freeze_sha256,
                                review_path, review_sha256, resolver):
    """Validate original approval and artifacts; schema 2 also binds consumer bytes."""
    validate_manifest(manifest)
    if manifest["provenance"] != "definitive":
        raise ValueError("Preserved execution validation requires definitive provenance")
    freeze_path = resolver.resolve(freeze_path, freeze_sha256)
    review_path = resolver.resolve(review_path, review_sha256)
    candidate = strict_json(freeze_path.read_text())
    review = strict_json(review_path.read_text())
    separate = candidate.get("schema_version") == 3 and candidate.get("freeze_scope") == "execution"
    roles = EXECUTION_FREEZE_ROLES if separate else FREEZE_ROLES
    topics = EXECUTION_REVIEW_TOPICS if separate else REVIEW_TOPICS
    if (not separate and candidate.get("schema_version") != 2
        or separate != execution_freeze_enabled(config)
        or set(candidate["artifacts"]) != roles):
        raise ValueError("Preprint freeze inventory incompatible")
    if candidate.get("missing_artifacts") or candidate.get("blockers"):
        raise ValueError("Preserved freeze candidate has unresolved blockers")
    if review.get("actor") != "human" or not review.get("reviewer") or not review.get("approved_at") or review.get("manifest_sha256") != freeze_sha256:
        raise ValueError("Separate dated human freeze review required")
    if any(review.get("approvals", {}).get(topic, review.get("approvals", {}).get("runtime_budget") if topic == "runtime" else None) is not True for topic in topics):
        raise ValueError("Required preprint approvals pending")
    if candidate["config_sha256"] != config_binding(config) or candidate["prompt_sha256"] != fingerprint(config["prompt"]):
        raise ValueError("Original config/prompt freeze mismatch")
    if candidate["git"].get("dirty") is not False or not re.fullmatch(r"[0-9a-f]{40,64}", candidate["git"].get("commit", "")):
        raise ValueError("Original freeze requires clean accepted versioned code")
    source = analysis_code_record()
    required_sources = EXECUTION_SOURCES if separate else PRODUCTION_SOURCES
    if (not set(required_sources).issubset(candidate["code"])
        or separate and set(candidate["code"]) != set(EXECUTION_SOURCES)
        or any(not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
               for digest in candidate["code"].values())):
        raise ValueError("Original freeze production source inventory incomplete")
    if not separate and any(candidate["code"].get(name) != digest for name, digest in source.items()):
        raise ValueError("Analysis production source differs from accepted freeze")
    paths = {role: resolver.resolve(ref["path"], ref["sha256"], parent=freeze_path.parent)
             for role, ref in candidate["artifacts"].items()}
    load = lambda role: strict_json(paths[role].read_text())
    original_protocol = Path(candidate["artifacts"]["protocol_record"]["path"])
    if original_protocol.name != "Preprint_Protocol.md" or original_protocol.parent.name != "protocols":
        raise ValueError("Freeze must bind actual protocols/Preprint_Protocol.md")
    if load("environment_lock") != candidate["environment"]:
        raise ValueError("Original environment lock mismatch")
    environment = ({"schema_version": 1, "original_environment_sha256": fingerprint(candidate["environment"])}
                   if separate else reproduction_environment(candidate["environment"]))
    errors = runtime_blockers(config, evidence_resolver=resolver.resolve,
                              expected_protocol_sha256=candidate["artifacts"]["protocol_record"]["sha256"])
    if errors:
        raise ValueError("Preserved runtime/route evidence invalid: " + "; ".join(errors))
    questions = load_questions(paths["questions_medqa"])
    if len(questions) != 1273 or any(q.benchmark != "medqa" or q.split != "evaluation" for q in questions):
        raise ValueError("Definitive requires complete MedQA evaluation package")
    audit = load("audit_medqa")
    original_qpath = candidate["artifacts"]["questions_medqa"]["path"]
    if (audit.get("ready") is not True
        or audit.get("keys_sha256", audit.get("keys_hash")) != candidate["artifacts"]["keys_medqa"]["sha256"]
        or audit.get("questions_sha256", audit.get("files", {}).get(original_qpath)) != candidate["artifacts"]["questions_medqa"]["sha256"]):
        raise ValueError("Original audit/question/key freeze mismatch")
    for role in ("route_control_evidence", "data_permissions"):
        artifact = load(role)
        if artifact.get("actor") != "human" or artifact.get("approved") is not True or artifact.get("config_sha256") != config_binding(config):
            raise ValueError("Accepted config-bound evidence required: " + role)
    validate_manifest_config(manifest, config, questions)
    expected = {"config": candidate["config_sha256"], "prompt": candidate["prompt_sha256"],
                "code": fingerprint(candidate["code"]), "environment": fingerprint(candidate["environment"]),
                **{name: candidate["artifacts"][role]["sha256"] for name, role in
                   (("questions", "questions_medqa"), ("keys", "keys_medqa"), ("protocol_record", "protocol_record"))}}
    if any(manifest["bindings"].get(k) != digest for k, digest in expected.items()):
        raise ValueError("Attempt package does not match accepted freeze bindings")
    environment.update(freeze_sha256=freeze_sha256, freeze_review_sha256=review_sha256,
                       artifact_map_sha256=resolver.map_sha256, analysis_code=source)
    return candidate, paths, environment
