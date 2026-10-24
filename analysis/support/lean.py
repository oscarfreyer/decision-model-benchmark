"""Offline analysis of lean-v1 packages; historical packages use analysis.methods.pipeline."""
import argparse
from pathlib import Path
import sys

from benchmarking.common import ROOT, WORKFLOW, file_hash, fingerprint, read_package, write_new
from analysis.support.source_record import analysis_code_record
from common import environment_record
from analysis.methods.pipeline import analyze, load_study, markdown_tables


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True,action='append')
    parser.add_argument('--questions',type=Path,required=True)
    parser.add_argument('--keys',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--primary-family', choices=['decision_vs_generative_31'])
    parser.add_argument('--d01-sum-tolerance', help='Must equal frozen D01 tolerance; defaults to saved value')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        if args.seed<0: raise ValueError('nonnegative bootstrap seed required')
        if args.output.exists(): raise ValueError('new analysis output directory required')
        for package in args.package:
            read_package(package) # Validates automatic provenance, saved source and journal hashes.
        manifest,terminal,coverage,keys=load_study(args.package,args.questions,args.keys)
        if manifest.get('execution_workflow') not in {WORKFLOW, 'lean-analysis-joined-v1'}: raise ValueError('historical packages require analysis.methods.pipeline and their original evidence')
        # Retain Oscar's accepted distribution normalization; calibration semantics remain unverified.
        methods={'normalization':'native_selected_normalized_distribution',
                 'probability_semantics_verified':False,'bootstrap_seed':args.seed,
                 'primary_family':args.primary_family, 'decision_models':{'D01':{'probability_semantics_verified':False}}}
        if args.d01_sum_tolerance is not None:
            methods['decision_models']['D01']['sum_tolerance'] = args.d01_sum_tolerance
        summary=analyze(manifest,terminal,coverage,keys,seed=args.seed,methods=methods)
        sources=analysis_code_record()
        summary['inputs']={'packages':[{'manifest_sha256':file_hash(package/'manifest.json'),
            'events_sha256':file_hash(package/'events.jsonl')} for package in args.package],
            'questions_sha256':file_hash(args.questions),'keys_sha256':file_hash(args.keys),
            'analysis_manifest_sha256':fingerprint(manifest),'analysis_code':sources,
            'analysis_environment':environment_record(), 'methods':methods,
            'analysis_review_status':'human scientific review required before publication'}
        args.output.mkdir(parents=True,exist_ok=False)
        write_new(args.output/'summary.json',summary)
        (args.output/'tables.md').write_text(markdown_tables(summary),encoding='utf-8')
        print(f'Offline analysis: {args.output}/summary.json; complete={summary["study_complete"]}')
        return 0
    except (ValueError,KeyError,TypeError,OSError) as exc:
        print('Analysis blocked: '+str(exc),file=sys.stderr);return 2

if __name__=='__main__': raise SystemExit(main())
