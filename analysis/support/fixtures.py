"""Consumer-owned fictional fixtures; no model/config/private-data imports."""
from pathlib import Path

from common import (PREPRINT_ROSTER, ROSTER_DECISION, KIMI_INSTANT_DECISION, SCHEMA_VERSION, Option, Question, artifact_ref, empty_probability, file_hash, fingerprint,
                    identity, measurement, native_probability, question_hash, question_record, write_new)


def write_fixture(root, n=8, experiment='E1+E2', *, revised_native_pairs=False, preprint_roster=False, concurrent_finishes=False, kimi_instant=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    package = root/"attempts"
    package.mkdir()
    questions = [Question("fictional-medqa", f"q{i:03d}", "development", f"Fictional symbol question {i}",
                          tuple(Option(k, f"Symbol {k}") for k in ("D", "A", "B", "C"))) for i in range(n)]
    qp, kp = root/"questions.jsonl", root/"isolated_keys.jsonl"
    write_new(qp, [question_record(q) for q in questions], jsonl=True)
    keys = [{**dict(zip(("benchmark", "split", "item_id"), identity(q))), "reference_answer": "ABCD"[i%4]} for i, q in enumerate(questions)]
    write_new(kp, keys, jsonl=True)
    configs = []
    for i in range(1, 25):
        pair = (i+1)//2
        configs.append({"config_id": f"C{i:02d}", "model_id": f"fictional/base{pair if i <= 20 else i}",
                        "base_model": f"fictional/base{pair if i <= 20 else i}",
                        "effort": ("low" if i%2 else "high") if i <= 20 else "standard",
                        "pair_id": f"P{pair:02d}" if i <= 20 else None, "adapter": "fictional"})
    configs.append({"config_id": "J01", "model_id": "fictional/jev", "base_model": "fictional/jev",
                    "effort": "native", "pair_id": None, "adapter": "fictional"})
    if revised_native_pairs or preprint_roster:
        # This opt-in fixture represents the amended native labels. The default
        # preserves historical v3 Low/High packages for compatibility tests.
        revised = {"C16": "xhigh", "C18": "xhigh", "C19": "none"}
        for config in configs:
            if config["config_id"] in revised:
                config["effort"] = revised[config["config_id"]]
    if preprint_roster:
        configs = [c for c in configs if c["config_id"] in PREPRINT_ROSTER]
        for config in configs:
            if config["config_id"] in {"C22", "C23"}: config["effort"] = "native"
            if config["pair_id"] not in {"P03", "P07", "P09"}:
                config["pair_id"] = None
    if kimi_instant:
        if not preprint_roster: raise ValueError('Kimi replacement requires the budget roster fixture')
        for config in configs:
            if config['config_id'] == 'C21':
                config.update(config_id='C25', model_id='moonshotai/kimi-k2.5', base_model='moonshotai/kimi-k2.5', effort='none')
    runs = [{"run_id": c["config_id"]+"-r1", "experiment_id": "E1", "config_id": c["config_id"], "replicate": 1, "role": "primary"} for c in configs]
    runs += [{"run_id": f"J01-r{r}", "experiment_id": "E2", "config_id": "J01", "replicate": r, "role": "secondary"} for r in (2, 3)]
    runs=[r for r in runs if r['experiment_id'] in experiment.split('+')]
    bindings = {k: fingerprint("fictional-"+k) for k in ("code", "environment", "config", "prompt", "protocol_record")}
    bindings.update(questions=file_hash(qp), keys=file_hash(kp))
    manifest = {"schema_version": SCHEMA_VERSION, "study_id": "consumer-fictional-study", "provenance": "mock",
                "benchmark": "fictional-medqa", "split": "development", "bindings": bindings,
                "key_manifest": {"sha256": file_hash(kp), "preparation_manifest_sha256": fingerprint("fictional-preparation")},
                "items": [{"item_id": q.item_id, "content_sha256": question_hash(q), "labels": list(q.labels)} for q in questions],
                "configurations": configs, "runs": runs}
    if preprint_roster:
        manifest["roster_decision"] = KIMI_INSTANT_DECISION if kimi_instant else ROSTER_DECISION
    write_new(package/"manifest.json", manifest)
    events, records, previous = [], [], None
    def append(event):
        nonlocal previous
        event.update(schema_version=SCHEMA_VERSION, bindings=bindings, manifest_sha256=fingerprint(manifest),previous_sha256=previous)
        previous = fingerprint(event)
        event["event_sha256"] = previous
        events.append(event)
    append({'event':'plan'})
    for run in runs:
        config_id, replicate = run["config_id"], run["replicate"]
        for i, q in enumerate(questions):
            ident = {"benchmark": q.benchmark, "split": q.split, "item_id": q.item_id,
                     "config_id": config_id, "run_id": run["run_id"]}
            prefix = f"{run['run_id']}-{q.item_id}"
            request = package/(prefix+".request.json")
            write_new(request, {"prompt": question_record(q), "configuration": config_id})
            request_ref = artifact_ref(request, package)
            append({"event": "start", "identity": ident, "request": request_ref,
                    "started_at": "2026-10-04T10:00:00+00:00", "reserved_usd": "0.1"})
            target = keys[i]["reference_answer"]
            good = ((i+replicate)%3 != 0) if config_id == "J01" else ((i+int(config_id[1:]))%4 != 0)
            selected = target if good else "ABCD"[("ABCD".index(target)+1)%4]
            status = "valid"
            if (config_id, i) == ("C05", 2):
                status, selected = "transport_error", None
            if (config_id, i) == ("C03", 1):
                status, selected = "refused", None
            raw, final = package/(prefix+".raw.json"), package/(prefix+".final.txt")
            write_new(raw, {"fictional": True, "status": status, "answer": selected})
            final.write_text('{"answer": "'+selected+'"}' if selected else "Fictional terminal failure")
            p = empty_probability("LLM confidence not requested")
            if config_id == "J01":
                score = "0.70" if good else "0.94"
                remainder = "0.10" if good else "0.02"
                vector = {k: score if k == selected else remainder for k in "ABCD"}
                p = native_probability(vector, selected, concentration="0.42", semantics="fictional native selected-option probability")
            index = int(config_id[1:]) if config_id != "J01" else 1
            record = {"identity": ident, "status": status, "selected_label": selected,
                      "failure_reason": "fictional terminal "+status if status != "valid" else None, "probability": p,
                      "usage": {**{k: measurement(v) for k, v in (("input_tokens", 70), ("output_tokens", 100), ("reasoning_tokens", 20), ("final_tokens", 80), ("billed_tokens", 100))},
                                "reasoning_in_output": True, "accounting_definition": "output includes reasoning; billed_tokens is output"},
                      "billing": {"observed_usd": measurement(str(index/1000)), "estimated_usd": measurement(str(index/1000)),
                                  "currency": "USD", "rate_provenance": "fictional rates", "rate_date": "2026-10-04"},
                      "timing": {"started_at": "2026-10-04T10:00:00+00:00", "finished_at": "2026-10-04T10:00:10+00:00", "duration_seconds": measurement(1+index/10)},
                      "route": {"requested_model": "fictional", "returned_model": "fictional", "requested_route": "fictional", "returned_route": "fictional",
                                "effective_controls": {"mock": True}, "provider_request_id": prefix, "finish_reason": "stop"},
                      "artifacts": {"request": request_ref, "raw_response": artifact_ref(raw, package), "final": artifact_ref(final, package), "reasoning": None}}
            records.append(record)
            append({"event": "finish", "identity": ident, "record": record})
    if concurrent_finishes:
        # All starts occur before reverse-order finishes: the consumer must join
        # by attempt identity, never completion position. Every event is resealed.
        events = [events[0]]+[e for e in events if e["event"] == "start"]+list(reversed([e for e in events if e["event"] == "finish"]))
        previous = None
        for event in events:
            event.pop("event_sha256", None)
            event["previous_sha256"] = previous
            previous = fingerprint(event)
            event["event_sha256"] = previous
    write_new(package/"events.jsonl", events, jsonl=True)
    return {"package": package, "questions": qp, "keys": kp, "manifest": manifest, "records": records,
            "coverage": {"expected": len(records), "terminal": len(records), "unstarted": [], "indeterminate": [], "complete": True, "last_event_sha256": previous}}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Fresh fictional fixture directory")
    args = parser.parse_args()
    fixture = write_fixture(args.output)
    print("Fictional consumer fixture:", fixture["package"])
