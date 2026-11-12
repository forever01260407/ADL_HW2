#!/usr/bin/env python3
"""Restore bundled references to the paths expected by the unchanged scripts.

The submitted directory has only the six required root files and code/script/.
Data, results and model directories are normal runtime outputs. This script
copies bundled references locally; it does not download, install or train.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

ROOT=Path(__file__).resolve().parents[2]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare(root=ROOT):
    manifest=root/'code/script/SUBMISSION_MANIFEST.json'
    if not manifest.is_file():
        # The development workspace already uses the original runtime paths.
        if all((root/p).is_file() for p in ['retrieval.py','evaluate.py','eval_utils.py']):
            return {'layout':'development','copied':0}
        raise FileNotFoundError('Missing code/script/SUBMISSION_MANIFEST.json')
    mapping=json.loads(manifest.read_text(encoding='utf-8'))['files']
    entries=[]
    for name,expected in mapping.items():
        parts=PurePosixPath(name).parts
        if '..' in parts or PurePosixPath(name).is_absolute():
            raise ValueError('Invalid reference path')
        if parts[:3]==('code','script','records'):
            relative=Path(*parts[3:])
            if not relative.parts or relative.parts[0] not in {'data','results','artifacts'}:
                raise ValueError('Unexpected bundled reference')
        elif parts[:3]==('code','script','official'):
            if len(parts)!=4 or parts[3] not in {'retrieval.py','evaluate.py','eval_utils.py'}:
                raise ValueError('Unexpected official source')
            relative=Path(parts[3])
        else:
            continue
        source=root/name
        destination=root/relative
        if not destination.resolve().is_relative_to(root.resolve()):
            raise ValueError('Reference escapes project directory')
        if digest(source)!=expected:
            raise ValueError(f'Bundled reference changed: {name}')
        if destination.exists() and digest(destination)!=expected:
            raise FileExistsError(f'Existing file differs; preserved without replacement: {relative}')
        entries.append((source,destination))
    copied=0
    for source,destination in entries:
        if not destination.exists():
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination)
            copied+=1
    return {'layout':'submission','copied':copied,'verified_references':len(entries)}


if __name__=='__main__':
    print(json.dumps(prepare()))
