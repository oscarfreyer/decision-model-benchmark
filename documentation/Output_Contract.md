# Data contracts

# Public structured results

The [data dictionary](../data/README.md) and `provenance/data_manifest.json` define the public derivative. Every selected identity is `(benchmark, split, item_id, config_id, run_id)`. Exactly 1,273 items × 54 runs yield 68,742 outcomes. Selected/original labels and statuses remain separate. Native probability vectors, missing measurements and observed resources retain their source values.

The loader requires exact roster/run coverage, one selected outcome per identity, contiguous retry indices starting at zero, at most ten retries, first-valid stopping and eleven calls for exhausted failures. It checks every referenced call and rejects missing, duplicate or conflicting identities.

# Collection packages

The original preparation and collection interfaces remain in `preparation/` and `benchmarking/`. Questions contain benchmark, split, item ID, stem and four labelled options; keys are separate. Collection manifests bind source, configuration, question order, environment and inputs. Journals preserve starts and terminal records, with content-addressed artifacts and a hash chain. A start without a finish remains indeterminate; a missing start remains unstarted.

The historical v3, lean collection and schema-6 extension readers retain their own validation. Raw collection evidence is not part of this public derivative. New collection must use its own input/output locations, fixed routes/settings, explicit mode and credentials.

# Numerical and presentation provenance

The public data manifest identifies derivative bytes. The source numerical and presentation hashes in `provenance/presentation.json` identify the retained accepted evidence. Public-file changes receive their own source mappings and release manifest. A portable derivative does not acquire the original private file's checksum.

The presentation uses saved estimates and arrays. Rendering never refits the statistics. PDF/SVG/PNG exports record the current public sources and are checked against the frozen main-figure PNG hashes.
