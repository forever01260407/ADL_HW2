#!/usr/bin/env python3
"""Validate the configured final archive over HTTPS in an empty destination.

No credentials, hosting changes, inference, training or Public evaluation.
"""
import argparse
import json
import time
from pathlib import Path
from download_release import ROOT, download, digest, install_archive, matches


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'output/remote-download-check')
    args=parser.parse_args()
    cfg=json.loads((ROOT/'code/script/configs/release_assets.json').read_text())['final_model']
    if not cfg['url']:parser.error('final_model.url has not been supplied')
    out=args.output.resolve()
    if not out.is_relative_to(ROOT/'output'):parser.error('output must be inside project output/')
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    archive=out/cfg['archive_name']
    download(cfg['url'],archive,cfg['archive_sha256'],started+1800)
    install_archive(archive,out/'checkpoint',cfg['files'],cfg['archive_sha256'])
    if not matches(out/'checkpoint',cfg['files']):raise ValueError('remote model file verification failed')
    report={'status':'passed','url':cfg['url'],'archive_sha256':digest(archive),
        'archive_bytes':archive.stat().st_size,'wall_seconds':time.monotonic()-started,
        'empty_destination':True,'checkpoint_files_verified':len(cfg['files']),
        'public_runs':0,'training_runs':0}
    target=ROOT/'results/w11_remote_download_validation.json'
    with target.open('x',encoding='utf-8') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
