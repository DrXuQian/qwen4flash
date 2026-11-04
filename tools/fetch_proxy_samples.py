#!/usr/bin/env python3
"""Fetch exact, bounded weight ranges from the pinned GGUFs for GPU format checks.

These are explicitly cropped expert/row samples, never full-model substitutes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

REVISION='38bb39ee97821de2c9009abb7e93950eec396e66'
REPOSITORY='unsloth/Qwen3.8-Flash-Next-GGUF'

def fetch(endpoint,path,start,size,file_size):
    end=start+size-1
    with tempfile.TemporaryDirectory(prefix='q4e-sample-') as tmp:
        body=Path(tmp)/'body';headers=Path(tmp)/'headers'
        url=f'{endpoint.rstrip("/")}/{REPOSITORY}/resolve/{REVISION}/{path}?download=true&q4e_range={start}-{end}'
        r=subprocess.run(['curl','--fail','--location','--silent','--show-error','--retry','3','--retry-all-errors',
            '--connect-timeout','20','--max-time','120','--range',f'{start}-{end}','--max-filesize',str(size+65536),
            '--dump-header',str(headers),'--output',str(body),'--write-out','%{http_code}',url],capture_output=True,text=True,check=True)
        final=next(s for s in reversed(headers.read_text().lower().split('\n\n')) if s.startswith('http/'))
        if r.stdout!='206' or f'content-range: bytes {start}-{end}/{file_size}' not in final.splitlines():raise ValueError('server did not honor the exact range')
        b=body.read_bytes()
        if len(b)!=size:raise ValueError('short weight range')
        return b

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--endpoint',default='https://huggingface.co');a=ap.parse_args()
    root=Path(__file__).resolve().parents[1];a.out.mkdir(parents=True,exist_ok=True)
    entries={}
    for name in ['target-index','mtp-index']:
        for shard in json.loads((root/'fixtures'/f'{name}.json').read_text())['shards']:
            for t in shard['header']['tensors']:entries[t['name']]=(shard,t)
    requests=[
        ('blk.0.ffn_gate_exps.weight',[0,17,511],None),('blk.0.ffn_up_exps.weight',[0,17,511],None),
        ('blk.2.ffn_gate_exps.weight',[0,511],None),('blk.2.ffn_up_exps.weight',[0,511],None),
        ('blk.0.ffn_down_exps.weight',[0,511],None),('blk.2.ffn_down_exps.weight',[0,511],None),
        ('blk.48.ffn_gate_exps.weight',[0,511],None),('blk.48.ffn_up_exps.weight',[0,511],None),
        ('blk.48.ffn_down_exps.weight',[0,511],None),('blk.48.indexer.q_proj.weight',[0],128),
        ('per_layer_token_embd.weight',[0],32),('output.weight',[0],256),
    ]
    def one(request):
        name,experts,crop=request;s,t=entries[name];k=t['dimensions'][0];n=t['dimensions'][1];total_e=t['dimensions'][2] if len(t['dimensions'])>2 else 1
        size=t['bytes']//total_e;rows=crop or n;row_bytes=size//n;parts=[];ranges=[]
        for e in experts:
            start=s['header']['data_offset']+t['offset']+size*e;count=rows*row_bytes
            data=fetch(a.endpoint,s['path'],start,count,s['header']['file_bytes']);parts.append(data)
            ranges.append({'expert':e,'start':start,'bytes':count,'sha256':hashlib.sha256(data).hexdigest()})
        filename=name+'.bin';data=b''.join(parts);(a.out/filename).write_bytes(data)
        result={'tensor':name,'file':filename,'qtype':t['kind'],'n':rows,'k':k,'experts':len(experts),'source_path':s['path'],'ranges':ranges,'sha256':hashlib.sha256(data).hexdigest()}
        print(json.dumps({'sample':name,'bytes':len(data)}),flush=True);return result
    with ThreadPoolExecutor(max_workers=4) as pool:samples=list(pool.map(one,requests))
    (a.out/'samples.json').write_text(json.dumps({'repository':REPOSITORY,'revision':REVISION,'cropped_samples':True,'samples':samples},indent=2)+'\n')

if __name__=='__main__':main()
