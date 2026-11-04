#!/usr/bin/env python3
"""Fetch GGUF descriptors with bounded HTTP ranges, without full weight downloads.

Writes a metadata-only index for `q4e inspect-index`. It cannot be used as
model weights. Prefix reads may include leading payload bytes; only descriptors
are saved. Every CDN response must honor Range and its exact file size.
"""
import argparse
import concurrent.futures
import hashlib
import io
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import urllib.request

LAYOUT = {0:(1,4),1:(1,2),2:(32,18),3:(32,20),6:(32,22),7:(32,24),8:(32,34),
          10:(256,84),11:(256,110),12:(256,144),13:(256,176),14:(256,210),
          16:(256,66),17:(256,74),18:(256,98),19:(256,50),20:(32,18),
          21:(256,110),22:(256,82),23:(256,136),29:(256,56),30:(1,2),42:(64,18)}
SCALAR = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}

def request_json(url):
    with urllib.request.urlopen(url, timeout=45) as f:
        return json.load(f)

def read_range(url, end, size):
    # Distinct range URLs avoid intermediaries replaying a differently sized range.
    # curl uses the system TLS stack; certificate validation remains enabled.
    with tempfile.TemporaryDirectory(prefix='q4e-range-') as tmp:
        body, headers = Path(tmp)/'body', Path(tmp)/'headers'
        result=subprocess.run([
            'curl','--fail','--location','--silent','--show-error',
            '--retry','2','--retry-all-errors','--max-time','45',
            '--range',f'0-{end}','--header','Accept-Encoding: identity',
            '--max-filesize',str(end+1+65536),'--dump-header',str(headers),
            '--output',str(body),'--write-out','%{http_code}',
            url+f'?download=true&q4e_range=0-{end}',
        ],capture_output=True,text=True,check=True)
        response=headers.read_text().lower().split('\n\n')
        final=next(block for block in reversed(response) if block.startswith('http/'))
        expected=f'content-range: bytes 0-{end}/{size}'
        if result.stdout!='206' or expected not in final.splitlines():
            raise ValueError('CDN did not honor the exact bounded Range request')
        data=body.read_bytes()
        if len(data)!=end+1:raise ValueError('range response length mismatch')
        return data

def parse(data, size):
    f=io.BytesIO(data)
    def read(n):
        b=f.read(n)
        if len(b)!=n: raise EOFError('header needs a larger range')
        return b
    def u(fmt): return struct.unpack('<'+fmt,read(struct.calcsize('<'+fmt)))[0]
    def string():
        n=u('Q')
        if n>16<<20: raise ValueError('GGUF string too large')
        return read(n).decode('utf8')
    def value(t):
        if t in SCALAR:return u(SCALAR[t])
        if t==8:return string()
        if t==9:
            et,n=u('I'),u('Q')
            if et==9 or n>2_000_000:raise ValueError('invalid GGUF array')
            return [value(et) for _ in range(n)]
        raise ValueError(f'unknown metadata type {t}')
    if read(4)!=b'GGUF':raise ValueError('not GGUF')
    version,nt,nm=u('I'),u('Q'),u('Q')
    if version not in (2,3) or nt>1_000_000 or nm>100_000:raise ValueError('invalid GGUF counts/version')
    metadata={}
    for _ in range(nm):
        key=string()
        if key in metadata:raise ValueError('duplicate metadata')
        metadata[key]=value(u('I'))
    tensors=[]
    for _ in range(nt):
        name=string();nd=u('I')
        if not 1<=nd<=4:raise ValueError('invalid tensor rank')
        dims=[u('Q') for _ in range(nd)];kind,offset=u('I'),u('Q')
        block,nbytes=LAYOUT[kind];elements=1
        for d in dims:elements*=d
        if not all(dims) or dims[0]%block:raise ValueError('invalid tensor dimensions')
        tensors.append(dict(name=name,dimensions=dims,kind=kind,offset=offset,bytes=elements//block*nbytes))
    alignment=metadata.get('general.alignment',32)
    if alignment<=0 or alignment&(alignment-1):raise ValueError('invalid alignment')
    offset=(f.tell()+alignment-1)//alignment*alignment
    for t in tensors:
        if offset+t['offset']+t['bytes']>size:raise ValueError('tensor outside source file')
    return dict(version=version,metadata=metadata,tensors=tensors,data_offset=offset,file_bytes=size), f.tell()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repository',default='unsloth/Qwen3.8-Flash-Next-GGUF')
    ap.add_argument('--revision',default='38bb39ee97821de2c9009abb7e93950eec396e66')
    ap.add_argument('--directory',default='UD-Q4_K_XL')
    ap.add_argument('--filename',help='inspect only this file within the directory')
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    files=request_json(f'https://huggingface.co/api/models/{a.repository}/tree/{a.revision}/{a.directory}')
    files=sorted((x for x in files if x['path'].endswith('.gguf')),key=lambda x:x['path'])
    if a.filename:files=[x for x in files if Path(x['path']).name==a.filename]
    if not files:raise ValueError('no GGUF shards')
    def one(entry):
        size=entry['size'];n=min(size,1<<20)
        while True:
            data=read_range(f'https://huggingface.co/{a.repository}/resolve/{a.revision}/{entry["path"]}',n-1,size)
            try:
                header,used=parse(data,size)
                return {'path':entry['path'],'header':header}, {'path':entry['path'],'header_sha256':hashlib.sha256(data[:used]).hexdigest(),'header_bytes':used,'file_bytes':size}
            except EOFError:
                if n>=min(size,16<<20):raise
                n=min(size,16<<20,n*2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: pairs=list(pool.map(one,files))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps({'shards':[x[0] for x in pairs]},ensure_ascii=False))
    receipt={'schema':'q4e.remote-descriptors.v1','metadata_only':True,'repository':a.repository,'revision':a.revision,'files':[x[1] for x in pairs]}
    a.out.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2))
    all_tensors=[t for x,_ in pairs for t in x['header']['tensors']]
    groups={}
    for t in all_tensors:
        role='ple_table' if t['name']=='per_layer_token_embd.weight' else 'experts' if '_exps.' in t['name'] else 'other_weights'
        groups[role]=groups.get(role,0)+t['bytes']
    print(json.dumps({'index':str(a.out),'tensors':len(all_tensors),'bytes_by_role':groups,'mtp_tensors':[t['name'] for t in all_tensors if 'nextn' in t['name'] or t['name'].startswith('mtp.')]},indent=2))

if __name__=='__main__':main()
