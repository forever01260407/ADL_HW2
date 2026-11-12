#!/usr/bin/env python3
"""Download/verify final H5 and pinned training/generation assets; no installs."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def matches(directory, files):
    return all((directory/p).is_file() and digest(directory/p)==h for p,h in files.items())


def download(url, destination, expected, deadline):
    if destination.is_file() and digest(destination)==expected:
        return
    if urllib.parse.urlparse(url).scheme!='https':
        raise ValueError('Remote asset URLs must use HTTPS')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.part')
    try:
        request=urllib.request.Request(url,headers={'User-Agent':'ADL-HW2-release/1.0'})
        with urllib.request.urlopen(request,timeout=60) as source, temporary.open('wb') as target:
            while True:
                if time.monotonic()>deadline:
                    raise TimeoutError('30-minute download budget exceeded')
                chunk=source.read(1024*1024)
                if not chunk:
                    break
                target.write(chunk)
        if digest(temporary)!=expected:
            raise ValueError(f'Download SHA-256 mismatch: {destination.name}; HTML/share page is not a model archive')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def install_archive(archive,destination,files,expected):
    if digest(archive)!=expected:
        raise ValueError('Final model archive SHA-256 mismatch')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='h5-verify-',dir=destination.parent) as folder:
        staging=Path(folder)
        with zipfile.ZipFile(archive) as z:
            seen=set()
            for info in z.infolist():
                name=info.filename
                path=PurePosixPath(name)
                if ('\\' in name or path.is_absolute() or '..' in path.parts or
                        stat.S_ISLNK(info.external_attr>>16) or name in seen):
                    raise ValueError('Unsafe or duplicate archive member')
                seen.add(name)
                if info.is_dir():
                    continue
                if name not in files:
                    raise ValueError(f'Unexpected model member: {name}')
                target=staging/name
                target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(info) as source,target.open('wb') as out:
                    shutil.copyfileobj(source,out)
        if not matches(staging,files):
            raise ValueError('Extracted final model differs from the frozen H5 files')
        if destination.exists():
            if matches(destination,files):
                return
            raise FileExistsError('Destination contains different weights; preserve it and choose a clean project directory')
        shutil.move(str(staging),str(destination))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,default=ROOT/'code/script/configs/release_assets.json')
    p.add_argument('--final-model-url')
    p.add_argument('--final-model-archive',type=Path)
    p.add_argument('--skip-auxiliary',action='store_true',help='Download final inference weights only')
    args=p.parse_args()
    config=json.loads(args.manifest.read_text(encoding='utf-8'))
    final=config['final_model']
    destination=ROOT/final['path']
    deadline=time.monotonic()+1800
    if not matches(destination,final['files']):
        if args.final_model_archive:
            archive=args.final_model_archive.resolve()
        else:
            url=args.final_model_url or final['url']
            if not url:
                p.error('Final H5 URL is not configured. Upload output/models/h5_final_r15922a14.zip, then supply its direct HTTPS URL using --final-model-url or release_assets.json')
            archive=ROOT/'artifacts/downloads'/final['archive_name']
            download(url,archive,final['archive_sha256'],deadline)
        install_archive(archive,destination,final['files'],final['archive_sha256'])
    print('Final H5: all checkpoint hashes verified',flush=True)
    if not args.skip_auxiliary:
        for label,model in config['auxiliary_models'].items():
            for name,h in model['files'].items():
                url=f"https://huggingface.co/{model['repo_id']}/resolve/{model['revision']}/{urllib.parse.quote(name)}"
                download(url,ROOT/model['path']/name,h,deadline)
            print(f'{label}: pinned model assets verified',flush=True)
    print('Download complete; environment unchanged',flush=True)


if __name__=='__main__':
    main()
