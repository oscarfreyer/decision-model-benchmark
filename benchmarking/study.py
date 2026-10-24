"""Persistent study package claims and duplicate-attempt protection. No provider calls."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path

from benchmarking.common import (ROOT, SCHEMA_VERSION, fingerprint, read_jsonl, read_package,
                    strict_json, study_context, write_new, continuation_parent)


class StudyRegistry:
    def __init__(self, directory, manifest):
        self.directory=Path(directory)
        self.manifest=manifest
        self.entries=[]

    def context(self, manifest):
        return study_context(manifest)

    def header(self):
        return {'schema_version':self.manifest['schema_version'],'study':self.context(self.manifest)}

    def load(self):
        header=self.directory/'study.json'
        expected=self.header()
        if header.exists():
            if strict_json(header.read_text())!=expected: raise ValueError('study registry bindings/metadata changed; review required')
        else:
            if (self.directory/'packages.jsonl').exists(): raise ValueError('study registry header missing; review required')
            write_new(header,expected)
        previous=None
        paths=set();runs=set();parents=set()
        journal=self.directory/'packages.jsonl'
        for entry in read_jsonl(journal) if journal.exists() else []:
            digest=entry.pop('event_sha256',None)
            if entry.get('previous_sha256')!=previous or fingerprint(entry)!=digest: raise ValueError('study registry chain/hash mismatch')
            previous=digest
            overlaps=runs.intersection(entry['run_ids'])
            parent=entry.get('parent_manifest_sha256')
            if parent and parent in parents: raise ValueError('study registry contains duplicate continuation branches')
            if entry['package'] in paths or (overlaps and entry.get('parent_manifest_sha256') not in {prior['manifest_sha256'] for prior in self.entries}):
                raise ValueError('duplicate study package/run claim')
            paths.add(entry['package']);runs.update(entry['run_ids'])
            if parent: parents.add(parent)
            self.entries.append(entry)
        self.previous=previous

    def records(self, *, reviewed_parent=None):
        records=[]
        parents={entry.get('parent_manifest_sha256') for entry in self.entries}
        for entry in self.entries:
            path=Path(entry['package'])
            if not (path/'manifest.json').is_file(): raise ValueError('registered package missing; human review required')
            manifest,terminal,coverage=read_package(path)
            if fingerprint(manifest)!=entry['manifest_sha256'] or self.context(manifest)!=self.context(self.manifest):
                raise ValueError('registered package manifest/bindings changed')
            if sorted(r['run_id'] for r in manifest['runs'])!=entry['run_ids']: raise ValueError('registered run claims changed')
            if manifest.get('continuation',{}).get('manifest_sha256')!=entry.get('parent_manifest_sha256'):
                raise ValueError('registered continuation parent changed')
            if coverage['indeterminate']: raise ValueError('indeterminate study attempts require human review; automatic resend prohibited')
            if any(r['status']=='drift' for r in terminal): raise ValueError('study drift remains unresolved; review required')
            # Parent stops remain evidence; only a leaf's unreviewed stop blocks dispatch.
            if entry['manifest_sha256'] not in parents:
                local=read_package(path,include_parents=False)[1] if 'continuation' in manifest else terminal
                if entry['manifest_sha256']!=reviewed_parent and any(r.get('scheduler',{}).get('stop_reason')=='provider_rate_limit_stop' for r in local):
                    raise ValueError('provider rate-limit stop requires review; automatic continuation prohibited')
                records.extend(terminal)
        identities=[tuple(r['identity'][k] for k in ('benchmark','split','item_id','config_id','run_id')) for r in records]
        if len(set(identities))!=len(identities): raise ValueError('duplicate study attempt across packages')
        return records

    def register(self, output, *, check_only=False):
        output=Path(output).resolve()
        parent=self.manifest.get('continuation',{}).get('manifest_sha256')
        if parent: continuation_parent(output,self.manifest,seen={output})
        self.records(reviewed_parent=parent)
        runs=sorted(r['run_id'] for r in self.manifest['runs'])
        ancestors=set()
        if parent:
            by_hash={entry['manifest_sha256']:entry for entry in self.entries}
            current=parent
            while current:
                if current in ancestors or current not in by_hash: raise ValueError('continuation parent is not a registered package')
                ancestors.add(current);current=by_hash[current].get('parent_manifest_sha256')
            if any(entry.get('parent_manifest_sha256')==parent and entry['package']!=str(output) for entry in self.entries):
                raise ValueError('parent already has a continuation; resume that package')
            parent_path=Path(by_hash[parent]['package']).resolve()
            if (output/self.manifest['continuation']['parent']).resolve()!=parent_path:
                raise ValueError('continuation parent path differs from registered claim')
        for entry in self.entries:
            if entry['package']==str(output):
                if entry['manifest_sha256']!=fingerprint(self.manifest): raise ValueError('resume manifest/bindings changed')
                return False
            if set(runs).intersection(entry['run_ids']) and entry['manifest_sha256'] not in ancestors:
                raise ValueError('study run already registered to another output; repeated attempts prohibited')
        if (output/'manifest.json').exists() or (output/'events.jsonl').exists():
            raise ValueError('unregistered existing package; preserve it for human review')
        if output.exists() and any(output.iterdir()): raise ValueError('fresh output must be empty')
        if check_only: return True
        entry={'package':str(output),'manifest_sha256':fingerprint(self.manifest),
               'run_ids':runs,'previous_sha256':self.previous}
        if parent: entry['parent_manifest_sha256']=parent
        digest=fingerprint(entry)
        with (self.directory/'packages.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({**entry,'event_sha256':digest})+'\n');stream.flush();os.fsync(stream.fileno())
        self.entries.append(entry);self.previous=digest
        return True



@contextmanager
def locked_study(config, manifest):
    root=Path(config['execution']['study_registry'])
    if not root.is_absolute(): root=ROOT/root
    directory=root/fingerprint(manifest['study_id'])
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        registry=StudyRegistry(directory,manifest);registry.load()
        yield registry
