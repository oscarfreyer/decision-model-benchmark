"""Manual E0 entry. --check never reads credentials, authorizes calls or runs models."""
import argparse
import os
from pathlib import Path

from common import ROOT, config_binding, file_hash, fingerprint, load_config, load_questions, scheduler_policy, strict_json, utc_now, write_new
from benchmarking.adapters import build_request
from benchmarking.legacy.readiness import development_blockers, PURPOSE
from benchmarking.run import build_prompt, make_manifest, run_package, validate_input_manifest


def packet_manifest(config, settings):
    prep = validate_input_manifest(settings['input_path'], settings['questions_path'])
    return make_manifest(config, settings['questions_path'], study_id=settings['study_id'],
        experiment='E0', provenance='development', seed=config['execution']['seed'],
        keys_sha256=prep['keys_sha256'], preparation_sha256=prep['preparation_manifest_sha256'],
        content_sha256=prep.get('content_manifest_sha256'), protocol_path=settings['protocol_path'])


def check_packet(packet):
    settings = strict_json((packet/'launch-settings.json').read_text())
    for path, digest in settings['artifacts_sha256'].items():
        if file_hash(path) != digest:
            raise ValueError('reviewed artifact changed: '+path)
    config = load_config(settings['config_path'])
    manifest = packet_manifest(config, settings)
    if manifest != strict_json((packet/'manifest.candidate.json').read_text()):
        raise ValueError('reviewed code/environment/config/input/protocol bindings changed')
    for question in load_questions(settings['questions_path']):
        for model in config['models'].values():
            build_request(question, model, build_prompt(question, config))
    blockers = development_blockers(config, manifest)
    if blockers:
        raise ValueError('; '.join(blockers))
    return settings, config, manifest, blockers


def approved_config(packet, settings, config):
    """Save exact reviewed launch settings; no funding review or numeric input."""
    path = packet/'config.ready.toml'
    if path.exists():
        if load_config(path) != config:
            raise ValueError('saved launch configuration changed; preserve it and review')
    else:
        with path.open('x') as output:
            output.write(Path(settings['config_path']).read_text())
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', type=Path, default=Path('benchmarking/input/e0-parallel-20'))
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--resume', action='store_true', help='continue the unchanged saved run using its existing authorization')
    args = parser.parse_args(argv)
    os.chdir(ROOT); packet = args.packet.resolve()
    settings, config, manifest, blockers = check_packet(packet)
    policy=scheduler_policy(config) or {'concurrency':4,'provider_concurrency':1}
    overrides=policy.get('provider_concurrency_overrides',{})
    print(f'E0: 12 development questions x 16 configurations = 192 attempts; {policy["concurrency"]} workers, default {policy["provider_concurrency"]} per provider.')
    if overrides:
        print('Provider limits:', ', '.join(f'{name}={limit}' for name,limit in sorted(overrides.items())))
    print('600s timeout; 2s InferenceNet/Mistral spacing; terminal attempts never retry; no fallbacks.')
    print('Manifest:', fingerprint(manifest)); print('Output:', settings['output'])
    for warning in settings.get('warnings', []): print('Review:', warning)
    if args.check:
        if (packet/'config.ready.toml').exists():
            config = approved_config(packet, settings, config)
            blockers = development_blockers(config, packet_manifest(config, settings))
        for blocker in blockers:
            print('Launch blocked:', blocker)
        print('Offline packet/request checks passed; no credentials read and no calls made.')
        return 2 if blockers else 0
    if args.resume:
        if not all(path.exists() for path in (packet/'config.ready.toml',packet/'manifest.ready.json',
                                             packet/'authorization.json',Path(settings['output'])/'manifest.json')):
            raise ValueError('resume requires the saved configuration, manifest, authorization and output package')
        config = approved_config(packet, settings, config)
        manifest = packet_manifest(config, settings)
        if manifest != strict_json((packet/'manifest.ready.json').read_text()):
            raise ValueError('saved run bindings changed; preserve the package and review')
        result = run_package(manifest, settings['questions_path'], config, settings['output'],
                             mock=False, authorization=packet/'authorization.json')
        print(result)
        return 0 if result['complete'] else 2
    if Path(settings['output']).exists() or (packet/'authorization.json').exists():
        raise ValueError('run/authorization already exists; preserve it and review continuation')
    config = approved_config(packet, settings, config)
    manifest = packet_manifest(config, settings)
    errors = development_blockers(config, manifest)
    if errors: raise ValueError('; '.join(errors))
    if not os.environ.get('OPENROUTER_API_KEY'):
        raise ValueError('load OPENROUTER_API_KEY in this console; its value is never printed')
    print('Final reviewed manifest:', fingerprint(manifest))
    print('The listed endpoint status/cadence limitations remain; failures consume attempts and are retained.')
    if input('To approve these exact paid calls and solution-free data use, type START E0 PARALLEL 12: ') != 'START E0 PARALLEL 12':
        print('Cancelled. No call authorization or model calls.'); return 0
    auth = {'actor':'human','approved':True,'reviewer':'Oscar','approved_at':utc_now(),
            'manifest_sha256':fingerprint(manifest),'config_sha256':config_binding(config),
            'purpose':PURPOSE,
            'data_permissions_approved':True,'approval_provenance':'Oscar typed START E0 PARALLEL 12 in the console'}
    write_new(packet/'manifest.ready.json', manifest); write_new(packet/'authorization.json', auth)
    result = run_package(manifest, settings['questions_path'], config, settings['output'],
                         mock=False, authorization=packet/'authorization.json')
    print(result)
    return 0 if result['complete'] else 2


if __name__ == '__main__':
    try: raise SystemExit(main())
    except (OSError, ValueError, KeyError) as exc: raise SystemExit('Blocked: '+str(exc))
