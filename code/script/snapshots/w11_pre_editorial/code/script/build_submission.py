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


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda:f.read(1048576),b''):h.update(c)
    return h.hexdigest()


def source_files():
    paths={ROOT/n for n in ['download.sh','run.sh','retrieval_utils.py','requirements.txt',
        'report.pdf','report.md','README.md','retrieval.py','evaluate.py','eval_utils.py',
        'PROJECT.md','PLAN.md','DECISIONS.md','PROGRESS.md']}
    paths.update(ROOT.glob('*_RESULTS.md'))
    for directory in ['code/script','results','data/derived']:
        paths.update(p for p in (ROOT/directory).rglob('*') if p.is_file()
            and '__pycache__' not in p.parts and p.suffix in {'.py','.json','.jsonl','.md','.log','.txt','.sh'})
    for directory in ['artifacts/training','artifacts/release_reproduction','artifacts/release_smoke']:
        paths.update(p for p in (ROOT/directory).rglob('*') if p.is_file()
            and 'checkpoint' not in p.parts and p.suffix in {'.json','.jsonl','.log'})
    return sorted(paths)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-only',action='store_true')
    parser.add_argument('--allow-pending-url',action='store_true')
    args=parser.parse_args()
    cfg=json.loads((ROOT/'code/script/configs/release_assets.json').read_text())
    student=cfg['student_id']
    if not student.isalnum() or student!=student.lower():raise ValueError('invalid lowercase student ID')
    remote_path=ROOT/'results/w11_remote_download_validation.json'
    remote=json.loads(remote_path.read_text()) if remote_path.exists() else {}
    ready=bool(cfg['final_model']['url']) and remote.get('status')=='passed' and remote.get('url')==cfg['final_model']['url'] and remote.get('archive_sha256')==cfg['final_model']['archive_sha256']
    stage=ROOT/'output/submission'/student
    stage.mkdir(parents=True,exist_ok=True)
    paths=source_files()
    mapping={}
    for source in paths:
        relative=source.relative_to(ROOT)
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
    (stage/'SUBMISSION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if args.stage_only:
        print(json.dumps({'staged':str(stage),'files':len(paths),'release_ready':ready}));return
    if not ready and not args.allow_pending_url:
        parser.error('Final model URL or verified remote download missing; staged files saved. Use --allow-pending-url only for a clearly named draft zip.')
    suffix='' if ready else '.pending-url'
    archive=ROOT/'output'/f'{student}{suffix}.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for relative in [*mapping,'SUBMISSION_MANIFEST.json']:
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
