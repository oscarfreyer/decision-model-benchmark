#!/usr/bin/env python3
"""Prepare seeded MedQA-US development items offline; keep keys out of inference."""

import argparse
import csv
from dataclasses import asdict
from datetime import date, datetime
import io
import json
from pathlib import Path
import platform
import random
import re
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarking.common import (ROOT, Option, Question, content_hash, dataset_audit, file_hash,
                    read_jsonl, strict_json, utc_now, write_new)


def convert(records: list[dict], split: str, digest: str, source: str,
            revision: str, exclusions: dict[int, str]) -> list[dict]:
    result = []
    for row, record in enumerate(records, 1):
        if row in exclusions:
            continue
        try:
            options = record.get('options')
            if not isinstance(options, dict) or set(options) != set('ABCD'):
                raise ValueError('expected the US four-option mapping with labels A–D')
            native_id = record.get('id', record.get('item_id'))
            if 'id' in record and 'item_id' in record and record['id'] != record['item_id']:
                raise ValueError('conflicting native IDs')
            if native_id is None:
                item_id = f'medqa-us-{split}-{digest[:16]}-{row:06d}'
            else:
                if isinstance(native_id, bool) or not isinstance(native_id, (str, int)) or not str(native_id).strip():
                    raise ValueError('invalid native ID')
                item_id = f'medqa-us-id-{native_id}'
            question = Question('medqa', item_id, split, record.get('question'),
                                tuple(Option(label, text) for label, text in options.items()))
            answer = record.get('answer_idx')
            if not isinstance(answer, str) or answer not in question.labels:
                raise ValueError('answer_idx must be one supplied label')
            answer_text = options[answer]
            if 'answer' in record and record['answer'] != answer_text:
                raise ValueError('answer text disagrees with answer_idx; human review required')
            result.append({'question': question, 'source_row': row, 'native_id': native_id,
                           'key': {'benchmark': 'medqa', 'item_id': item_id, 'split': split,
                                   'reference_answer': answer, 'provenance': source,
                                   'revision': revision}, 'answer_text': answer_text})
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{split} source row {row}: {exc}') from exc
    return result


def checked_exclusions(path: Path | None, digest: str, count: int) -> dict[int, str]:
    """Only explicitly reviewed development rows may be excluded; bind to exact bytes."""
    if path is None:
        return {}
    record = strict_json(path.read_text(encoding='utf-8'))
    if not isinstance(record, dict) or set(record) != {'source_sha256', 'rows'} or record['source_sha256'] != digest:
        raise ValueError('exclusions require the development source_sha256 and rows')
    if not isinstance(record['rows'], list):
        raise ValueError('exclusion rows must be a list')
    exclusions = {}
    for entry in record['rows']:
        if not isinstance(entry, dict) or set(entry) != {'source_row', 'reason'}:
            raise ValueError('each exclusion requires source_row and reason')
        row, reason = entry['source_row'], entry['reason']
        if type(row) is not int or not 1 <= row <= count or row in exclusions:
            raise ValueError('exclusion rows must be unique, valid 1-based rows')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('each exclusion needs a nonempty human-reviewed reason')
        exclusions[row] = reason
    return exclusions


def check_pool(development: list[dict], evaluation: list[dict]) -> None:
    ids, contents = {}, {}
    for item in development + evaluation:
        q = item['question']
        digest = content_hash(q)
        other = ids.get(q.item_id) or contents.get(digest)
        if other is not None:
            raise ValueError(f'duplicate/overlap at {q.split} row {item["source_row"]}; '
                             f'first seen at {other}; review before selection')
        location = f'{q.split} row {item["source_row"]}'
        ids[q.item_id] = location
        contents[digest] = location


def question_record(item: dict) -> dict:
    record = asdict(item['question'])
    record.pop('subject')
    return record


def answer_table(selected: list[dict]) -> str:
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(['selection_order', 'benchmark', 'item_id', 'split', 'source_row',
                     'question', 'correct_label', 'correct_answer_text'])
    for order, item in enumerate(selected, 1):
        q = item['question']
        writer.writerow([order, q.benchmark, q.item_id, q.split, item['source_row'],
                         q.stem, item['key']['reference_answer'], item['answer_text']])
    return stream.getvalue()


def prepare(*, dev_source: Path, test_source: Path, seed: int, source_url: str,
            rights_basis: str, acquired_at: str, source_revision: str = '',
            exclusions_file: Path | None = None, name: str | None = None, count: int = 100,
            workspace: Path = ROOT, expected_test_items: int = 1273, train_source: Path | None = None, fictional: bool = False) -> dict[str, Path]:
    if expected_test_items != 1273 and not fictional: raise ValueError('nonstandard count allowed only for explicitly fictional fixtures')
    if fictional and ('fictional' not in rights_basis.casefold() or 'example.invalid' not in source_url): raise ValueError('fictional provenance required')
    if type(seed) is not int or type(expected_test_items) is not int or expected_test_items <= 0:
        raise ValueError('seed must be an integer and expected test count positive')
    if type(count) is not int or count <= 0:
        raise ValueError('count must be a positive integer')
    if name is None:
        name = f'medqa-development-{count}'
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
        raise ValueError('name must be a simple filename prefix')
    for field, value in [('source URL', source_url), ('rights basis', rights_basis)]:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{field} is required')
    if not source_url.startswith(('https://', 'http://')):
        raise ValueError('source URL must be HTTP(S)')
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', acquired_at):
        date.fromisoformat(acquired_at)
        acquisition_precision = 'date_only; exact time and timezone unknown'
    else:
        timestamp = datetime.fromisoformat(acquired_at.replace('Z', '+00:00'))
        if timestamp.tzinfo is None:
            raise ValueError('acquired_at must be a date or a timestamp with timezone')
        acquisition_precision = 'timestamp_with_timezone'
    dev_source, test_source = dev_source.resolve(), test_source.resolve()
    if dev_source == test_source:
        raise ValueError('development and test must be separate original files')
    digests = {'development': file_hash(dev_source), 'evaluation': file_hash(test_source)}
    if not fictional:
        originals={'development':'774bbd90103af43badbe8d5294e336443573fc37b918763920b8fc3b3df581d9',
                   'evaluation':'c3b905ccfa66152dc25afbcb2c10e86c0bdb208824f0658fcdb2c040f60a2beb'}
        if digests!=originals: raise ValueError('original study split SHA-256 mismatch; preserve and resolve source identity')
        if train_source is None or file_hash(train_source)!='3a65baf760f17c395058d6699315a7ce8aa767c50cb193fea87ceb94dd790324':
            raise ValueError('hash-verified original train split required for the complete overlap audit')
    dev_records, test_records = read_jsonl(dev_source), read_jsonl(test_source)
    if len(test_records) != expected_test_items:
        raise ValueError(f'expected {expected_test_items} test rows, observed {len(test_records)}')
    excluded = checked_exclusions(exclusions_file, digests['development'], len(dev_records))
    revision = source_revision.strip() or 'unversioned; exact originals bound by SHA-256'
    dev = convert(dev_records, 'development', digests['development'], source_url, revision, excluded)
    test = convert(test_records, 'evaluation', digests['evaluation'], source_url, revision, {})
    check_pool(dev, test)
    train_audit = {'checked': False, 'reason': 'train source not supplied'}
    if train_source is not None:
        train_source = train_source.resolve()
        if train_source in (dev_source,test_source): raise ValueError('train/development/test sources must be separate')
        train_digest = file_hash(train_source)
        train = convert(read_jsonl(train_source), 'train', train_digest, source_url, revision, {})
        check_pool(train + dev, test)
        train_audit = {'checked': True, 'count': len(train), 'sha256': train_digest, 'path': str(train_source)}
    if len(dev) < count:
        raise ValueError(f'need {count} eligible development items, observed {len(dev)}')
    selected = random.Random(seed).sample(dev, count)
    workspace = workspace.resolve()
    output = workspace / 'preparation/output'
    binput, ainput, boutput = output / 'questions', output / 'answers', output
    paths = {'questions': binput / f'{name}.jsonl',
             'evaluation': binput / f'{name}-evaluation.jsonl',
             'keys': ainput / f'{name}-keys.jsonl',
             'audit_keys': ainput / f'{name}-audit-keys.jsonl',
             'evaluation_keys': ainput / f'{name}-evaluation-keys.jsonl',
             'answers': ainput / f'{name}-answers.csv',
             'provenance': output / f'{name}-provenance.json',
             'selection': output / f'{name}-selection.json',
             'content': output / f'{name}-content.json',
             'audit': boutput / f'{name}-audit.json',
             'evaluation_audit': output / f'{name}-evaluation-audit.json',
             'development_audit': output / f'{name}-development-audit.json',
             'questions_manifest': output / f'{name}-questions-inference-manifest.json',
             'evaluation_manifest': output / f'{name}-evaluation-inference-manifest.json'}
    for path in paths.values():
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'refusing to overwrite {path}; use a fresh --name')
    for original in (dev_source, test_source):
        if original.is_relative_to(workspace / 'benchmarking'):
            raise ValueError('store answer-bearing originals outside benchmarking, under preparation/input')
    provenance = {'source': source_url, 'revision': revision, 'license': rights_basis,
                  'acquired_at': acquired_at, 'acquisition_precision': acquisition_precision,
                  'source_subset': 'US English, four-option',
                  'source_splits': {'development': str(dev_source), 'test': str(test_source)},
                  'originals': {str(dev_source): digests['development'], str(test_source): digests['evaluation']}}
    questions = [question_record(x) for x in selected]
    evaluation = [question_record(x) for x in test]
    keys = [x['key'] for x in selected]
    combined_keys = keys + [x['key'] for x in test]
    expected = {'medqa/development': count, 'medqa/evaluation': expected_test_items}
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        for filename, records in [('dev.jsonl', questions), ('test.jsonl', evaluation), ('keys.jsonl', combined_keys)]:
            write_new(temporary / filename, records, jsonl=True)
        audit = dataset_audit([temporary / 'dev.jsonl', temporary / 'test.jsonl'],
                              temporary / 'keys.jsonl', expected, provenance)
        if not audit['ready']:
            raise ValueError('combined dataset audit failed')
    selection = {'schema_version': 2, 'provenance': 'mock' if fictional else 'development_preparation', 'created_at': utc_now(), 'source_split': 'official development',
                 'source_subset': 'US English, four-option; operator verified from acquisition',
                 'source_sha256': digests['development'], 'evaluation_sha256': digests['evaluation'],
                 'source_counts': {'development': len(dev_records), 'test': len(test_records)},
                 'eligible_development_count': len(dev), 'requested_count': count, 'selected_count': len(selected),
                 'algorithm': 'random.Random(seed).sample(eligible_rows_in_source_order, count)',
                 'seed': seed, 'python': platform.python_version(),
                 'python_implementation': platform.python_implementation(),
                 'script_sha256': file_hash(Path(__file__)), 'common_sha256': file_hash(ROOT / 'benchmarking/common.py'),
                 'ordered_selected_ids': [x['question'].item_id for x in selected],
                 'selected_source_rows': [x['source_row'] for x in selected],
                 'selected_native_ids': [x['native_id'] for x in selected],
                 'exclusions': [{'source_row': row, 'reason': reason} for row, reason in sorted(excluded.items())],
                 'exclusions_file': None if exclusions_file is None else
                     {'path': str(exclusions_file.resolve()), 'sha256': file_hash(exclusions_file)},
                 'not_selected_eligible_rows': [x['source_row'] for x in dev
                     if x['question'].item_id not in {y['question'].item_id for y in selected}],
                 'train_audit': train_audit,
                 'separation': 'full eligible development pool checked against all test items; no automatic exclusions'}
    if train_source is not None: provenance['originals'][str(train_source)] = train_digest
    for original, digest in provenance['originals'].items():
        if file_hash(Path(original)) != digest:
            raise ValueError('original changed during preparation')
    for directory in (binput, ainput, boutput):
        directory.mkdir(parents=True, exist_ok=True)
    for key, records in [('questions', questions), ('evaluation', evaluation), ('keys', keys), ('audit_keys', combined_keys), ('evaluation_keys', [x['key'] for x in test])]:
        write_new(paths[key], records, jsonl=True)
        if key in ('keys', 'audit_keys', 'evaluation_keys'):
            paths[key].chmod(0o600)
    with paths['answers'].open('x', encoding='utf-8', newline='') as stream:
        stream.write(answer_table(selected))
    paths['answers'].chmod(0o600)
    write_new(paths['provenance'], provenance)
    audit['files'] = {str(paths[k]): file_hash(paths[k]) for k in ('questions', 'evaluation')}
    audit['keys_hash'] = file_hash(paths['audit_keys'])
    write_new(paths['audit'], audit)
    for role, qrole, krole, members in [('evaluation_audit','evaluation','evaluation_keys',test),('development_audit','questions','keys',selected)]:
        write_new(paths[role], {'schema_version':2,'ready':True,'benchmark':'medqa','split':members[0]['question'].split,
            'count':len(members),'questions_sha256':file_hash(paths[qrole]),'keys_sha256':file_hash(paths[krole]),
            'cross_split_overlap':False,'normalized_duplicates':False,'train_audit':train_audit,
            'items':[{'id':list((x['question'].benchmark,x['question'].split,x['question'].item_id)),
                      'source_row':x['source_row'],'hash':__import__('benchmarking.common', fromlist=['question_hash']).question_hash(x['question'])} for x in members]})
    content = {'schema_version': 1, 'type': 'preparation_content',
        'algorithm': selection['algorithm'], 'seed': seed,
        'source_sha256': digests,
        'train_sha256': train_audit.get('sha256'),
        'exclusions': selection['exclusions'],
        'ordered_selected_ids': selection['ordered_selected_ids'],
        'selected_source_rows': selection['selected_source_rows'],
        'evaluation_ids': [x['question'].item_id for x in test],
        'evaluation_source_rows': [x['source_row'] for x in test],
        'artifacts': {role: file_hash(paths[role]) for role in ('questions','keys','evaluation','evaluation_keys')}}
    write_new(paths['content'], content)
    selection['content_manifest_sha256'] = file_hash(paths['content'])
    selection['artifacts'] = {key: {'path': str(path.relative_to(output)), 'sha256': file_hash(path)}
                              for key, path in paths.items() if key not in ('selection','questions_manifest','evaluation_manifest')}
    write_new(paths['selection'], selection)
    for role, key_role, split_name, members in [('questions','keys','development',selected),('evaluation','evaluation_keys','evaluation',test)]:
        public = {'schema_version': 2, 'type': 'inference_input', 'provenance': 'mock' if fictional else 'prepared',
                  'benchmark': 'medqa', 'split': split_name, 'count': len(members),
                  'questions_sha256': file_hash(paths[role]), 'keys_sha256': file_hash(paths[key_role]),
                  'preparation_manifest_sha256': file_hash(paths['selection']),
                  'content_manifest_sha256': file_hash(paths['content']),
                  'ordered_ids': [x['question'].item_id for x in members],
                  'source_rows': [x['source_row'] for x in members]}
        public_path = output / f'{name}-{role}-inference-manifest.json'
        write_new(public_path, public); paths[role + '_manifest'] = public_path
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dev-source', type=Path, required=True, help='original US/4_options development JSONL')
    parser.add_argument('--test-source', type=Path, required=True, help='original US/4_options test JSONL')
    parser.add_argument('--workspace', type=Path, default=ROOT, help='output workspace, existing packages never overwritten')
    parser.add_argument('--fictional', action='store_true', help='explicit synthetic fixture provenance; no real data')
    parser.add_argument('--expected-test-items', type=int, default=1273, help='1273 for real MedQA; smaller values require --fictional')
    parser.add_argument('--train-source', type=Path, help='optional original train JSONL for complete cross-split overlap audit')
    parser.add_argument('--seed', type=int, required=True, help='fixed selection seed recorded before outcomes')
    parser.add_argument('--count', type=int, default=100, help='development questions to select (default: 100); must fit eligible pool')
    parser.add_argument('--source-url', required=True, help='actual acquisition URL from the official source')
    parser.add_argument('--rights-basis', required=True, help='documented lawful acquisition/use basis')
    parser.add_argument('--acquired-at', required=True, help='known ISO-8601 acquisition date, or timestamp with timezone; never invent a time')
    parser.add_argument('--source-revision', default='', help='publisher revision, if supplied; otherwise hashes bind identity')
    parser.add_argument('--exclusions-file', type=Path, help='checksum-bound human-reviewed development exclusions JSON')
    parser.add_argument('--name', help='fresh output prefix (default: medqa-development-COUNT); existing files are never replaced')
    parser.add_argument('--subset-count', type=int, help='also export this prefix of the seeded development selection through this primary entry')
    args = parser.parse_args()
    try:
        kwargs = vars(args).copy()
        subset_count = kwargs.pop('subset_count')
        subset_output = None
        if subset_count is not None:
            if not 1 <= subset_count <= args.count:
                raise ValueError('subset-count must be between 1 and development count')
            name = args.name or f'medqa-development-{args.count}'
            subset_output = args.workspace.resolve()/'preparation/output'/f'{name}-subset-{subset_count}'
            if subset_output.exists():
                raise FileExistsError('fresh subset directory required')
        paths = prepare(**kwargs)
        if subset_output is not None:
            from preparation.subset import subset
            paths['subset_manifest'] = subset(paths['selection'], paths['questions'], paths['keys'], subset_output, subset_count)
    except (OSError, ValueError, TypeError) as exc:
        print(f'Preparation failed: {exc}', file=sys.stderr)
        return 1
    print(f'Prepared {args.count} development questions; combined audit ready=true. No model calls.')
    for key, path in paths.items():
        print(f'{key}: {path}')
    print(f'Public handoffs: {paths["questions_manifest"]} and {paths["evaluation_manifest"]}; keys stay isolated.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
