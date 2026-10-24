"""Offline checks of immutable accepted execution provenance and SAP evidence."""
from analysis.support.source_record import analysis_code_record
from common import fingerprint, load_config, strict_json
from analysis.support.reproduction import ArtifactResolver, reproduction_environment, verify_execution_provenance


def verify_definitive_analysis(manifest, *, seed, config=None, freeze_path=None,
                               freeze_sha256=None, review_path=None, review_sha256=None,
                               resolver=None, reproduction_record=None,
                               analysis_review_path=None, analysis_review_sha256=None):
    from analysis.methods.pipeline import NORMALIZATION_POLICIES, review_file
    if config is None or freeze_path is None or review_path is None:
        raise ValueError("Definitive freeze reproduction requires explicit config, freeze manifest/review and their SHA-256 values")
    config = load_config(config) if not isinstance(config, dict) else config
    resolver = resolver or ArtifactResolver()
    candidate, paths, environment = verify_execution_provenance(
        manifest, config, freeze_path=freeze_path, freeze_sha256=freeze_sha256,
        review_path=review_path, review_sha256=review_sha256, resolver=resolver)
    separate = candidate["schema_version"] == 3
    if separate:
        if analysis_review_path is None or analysis_review_sha256 is None:
            raise ValueError("Separate dated human analysis release and explicit SHA-256 required")
        release_path = resolver.resolve(analysis_review_path, analysis_review_sha256)
        release = strict_json(release_path.read_text())
        if (release.get("schema_version") != 3 or release.get("type") != "analysis_release"
            or release.get("actor") != "human" or release.get("approved") is not True
            or not release.get("reviewer") or not release.get("approved_at")):
            raise ValueError("Separate dated human analysis release required")
        if release.get("execution_freeze_sha256") != freeze_sha256:
            raise ValueError("Analysis release execution freeze binding mismatch")
        if release.get("analysis_code") != analysis_code_record():
            raise ValueError("Analysis production source differs from accepted analysis release")
        analysis_config = release.get("analysis")
        if not isinstance(analysis_config, dict) or not analysis_config:
            raise ValueError("Analysis release requires reviewed analysis configuration")
        if not isinstance(release.get("environment"), dict):
            raise ValueError("Analysis release requires reviewed environment")
        later_environment = reproduction_environment(release["environment"])
        environment.update({"analysis_environment_sha256": later_environment.pop("original_environment_sha256"),
                            **later_environment,
                            "analysis_review_sha256": analysis_review_sha256,
                            "analysis_configuration_sha256": fingerprint(analysis_config)})
        artifact = release.get("method_review")
        if not isinstance(artifact, dict) or set(artifact) != {"path", "sha256"}:
            raise ValueError("Checksum-bound analysis method review required")
        method_path = resolver.resolve(artifact["path"], artifact["sha256"], parent=release_path.parent)
    else:
        if analysis_review_path is not None or analysis_review_sha256 is not None:
            raise ValueError("Historical joint freezes require their original method/code bindings")
        artifact = candidate["artifacts"]["method_review"]
        method_path = paths["method_review"]
        analysis_config = config["analysis"]
    methods = review_file(method_path, artifact["sha256"], manifest,
                          kind="analysis_methods", resolver=resolver)
    if methods.get("actor") != "human" or methods.get("approved") is not True:
        raise ValueError("Human acceptance of frozen SAP required")
    if methods.get("analysis_sha256") != fingerprint(analysis_config):
        raise ValueError("Frozen analysis configuration mismatch")
    if type(methods.get("bootstrap_seed")) is not int or methods["bootstrap_seed"] < 0 or methods["bootstrap_seed"] != seed:
        raise ValueError("CLI bootstrap seed must match accepted frozen method")
    if methods.get("normalization") not in NORMALIZATION_POLICIES or methods["normalization"] != analysis_config.get("normalization"):
        raise ValueError("Accepted normalization policy pending/mismatched")
    if type(methods.get("probability_semantics_verified")) is not bool:
        raise ValueError("Explicit native probability semantics review required")
    stem_review = methods.get("shared_stem_review")
    if not stem_review or set(stem_review) != {"path", "sha256"}:
        raise ValueError("Checksum-bound shared-stem dependence review required")
    resolver.resolve(stem_review["path"], stem_review["sha256"], parent=method_path.parent)
    if reproduction_record is not None:
        reproduction_record.update(environment)
    return methods
