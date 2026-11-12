#!/usr/bin/env python3
"""Stage/zip an explicit source+evidence allowlist; exclude models, raw data and secrets."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
MANIFEST_PATH=Path('code/script/SUBMISSION_MANIFEST.json')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1048576),b''):h.update(c)
    return h.hexdigest()


def source_files():
    paths={ROOT/n for n in ['download.sh','run.sh','retrieval_utils.py','requirements.txt',
        'report.pdf','report.md','README.md','retrieval.py','evaluate.py','eval_utils.py']}
    for directory in ['code/script','results','data/derived']:
        paths.update(p for p in (ROOT/directory).rglob('*') if p.is_file()
            and '__pycache__' not in p.parts and 'w11_pre_editorial' not in p.parts
            and 'w11_pre_strict_layout' not in p.parts
            and p.suffix in {'.py','.json','.jsonl','.md','.log','.txt','.sh'})
    for directory in ['artifacts/training','artifacts/release_reproduction','artifacts/release_smoke']:
        paths.update(p for p in (ROOT/directory).rglob('*') if p.is_file()
            and 'checkpoint' not in p.parts and p.suffix in {'.json','.jsonl','.log'})
    return sorted(paths)


def submission_path(source):
    """Match the slide's six root files plus code/script exactly."""
    relative=source.relative_to(ROOT)
    if relative==Path('report.md'):
        return Path('code/script/docs/report.md')
    if len(relative.parts)==1 and relative.name in {'retrieval.py','evaluate.py','eval_utils.py'}:
        return Path('code/script/official')/relative
    if relative.parts[0] in {'results','data','artifacts'}:
        return Path('code/script/records')/relative
    return relative


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-only',action='store_true')
    parser.add_argument('--allow-pending-url',action='store_true')
    parser.add_argument('--list-files',action='store_true',help='Preview submission paths without staging or creating a ZIP')
    args=parser.parse_args()
    paths=source_files()
    if args.list_files:
        names=[submission_path(p).as_posix() for p in paths]+[MANIFEST_PATH.as_posix()]
        print(json.dumps({'files':names,'root_files':[p for p in names if '/' not in p],
            'file_count':len(names),'archive_created':False},indent=2))
        return
    cfg=json.loads((ROOT/'code/script/configs/release_assets.json').read_text())
    student=cfg['student_id']
    if not student.isalnum() or student!=student.lower():raise ValueError('invalid lowercase student ID')
    remote_path=ROOT/'results/w11_remote_download_validation.json'
    remote=json.loads(remote_path.read_text()) if remote_path.exists() else {}
    ready=bool(cfg['final_model']['url']) and remote.get('status')=='passed' and remote.get('url')==cfg['final_model']['url'] and remote.get('archive_sha256')==cfg['final_model']['archive_sha256']
    # Keep the earlier development-test checkout separate from the clean hand-in.
    stage=ROOT/'output/submission_ready'/student
    stage.mkdir(parents=True,exist_ok=True)
    mapping={}
    for source in paths:
        relative=submission_path(source)
        destination=stage/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
        if destination.suffix=='.sh':destination.chmod(0o755)
        mapping[relative.as_posix()]=sha(destination)
    manifest={'student_id':student,'release_ready':ready,
        'remaining_blockers':[] if ready else ['final model hosting URL and clean HTTPS download verification'],
        'final_model_archive_sha256':cfg['final_model']['archive_sha256'],
        'weights_excluded_from_zip':True,'original_train_public_data_excluded':True,
        'files':mapping}
    (stage/MANIFEST_PATH).write_text(json.dumps(manifest,indent=2)+'\n')
    if args.stage_only:
        print(json.dumps({'staged':str(stage),'files':len(paths),'release_ready':ready}));return
    if not ready and not args.allow_pending_url:
        parser.error('Final model URL or verified remote download missing; staged files saved. Use --allow-pending-url only for a clearly named draft zip.')
    suffix='' if ready else '.pending-url'
    archive=ROOT/'output'/f'{student}{suffix}.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for relative in [*mapping,MANIFEST_PATH.as_posix()]:
            file=stage/relative
            info=zipfile.ZipInfo(f'{student}/{relative}',date_time=(2026,10,25,0,0,0))
            info.create_system=3
            info.external_attr=(0o100755 if relative.endswith('.sh') else 0o100644)<<16
            info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,file.read_bytes())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('zip CRC failure')
        if any(not n.startswith(student+'/') for n in z.namelist()):raise ValueError('missing student root')
        for relative,expected in mapping.items():
            if hashlib.sha256(z.read(student+'/'+relative)).hexdigest()!=expected:
                raise ValueError(f'packaged file changed: {relative}')
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':sha(archive),
        'files':len(paths)+1,'release_ready':ready}))


if __name__=='__main__':main()
