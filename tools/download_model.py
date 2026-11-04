#!/usr/bin/env python3
"""Resume the pinned target/shared-MTP downloads and verify every full-file SHA256."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

REPOSITORY = "unsloth/Qwen3.8-Flash-Next-GGUF"
REVISION = "38bb39ee97821de2c9009abb7e93950eec396e66"
FILES = [
    ("UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf", 10946624, "4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082"),
    ("UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf", 49859583136, "3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9"),
    ("UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf", 49376141504, "56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3"),
    ("UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf", 12087983520, "753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a"),
    ("MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf", 2786568256, "5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6"),
]

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(8<<20):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--endpoint',default='https://huggingface.co')
    ap.add_argument('--jobs',type=int,default=2)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if not 1<=a.jobs<=4:ap.error('jobs must be 1..4')
    remaining=sum(max(0,n-next((p.stat().st_size for p in [a.out/name,a.out/(name+'.part')] if p.exists()),0)) for name,n,_ in FILES)
    if shutil.disk_usage(a.out).free<remaining+(2<<30):raise RuntimeError('insufficient disk space, including 2 GiB reserve')
    def fetch(entry):
        name,size,sha=entry;final=a.out/name;partial=a.out/(name+'.part');final.parent.mkdir(parents=True,exist_ok=True)
        if final.exists():
            if final.stat().st_size!=size or digest(final)!=sha:raise RuntimeError(f'pre-existing file failed verification: {name}')
        else:
            print(json.dumps({'downloading':name,'expected_bytes':size}),flush=True)
            subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','5',
                '--retry-all-errors','--retry-delay','3','--connect-timeout','30','--speed-time','120',
                '--speed-limit','1024','--continue-at','-','--output',str(partial),
                f'{a.endpoint.rstrip("/")}/{REPOSITORY}/resolve/{REVISION}/{name}?download=true'],check=True)
            if partial.stat().st_size!=size or digest(partial)!=sha:raise RuntimeError(f'download failed full-file verification: {name}')
            partial.replace(final)
        row={'file':name,'bytes':size,'sha256':sha};print(json.dumps({'verified':row}),flush=True);return row
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:rows=list(pool.map(fetch,FILES))
    (a.out/'verified.json').write_text(json.dumps({'repository':REPOSITORY,'revision':REVISION,'files':rows},indent=2)+'\n')

if __name__=='__main__':main()
