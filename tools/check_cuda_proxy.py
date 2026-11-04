#!/usr/bin/env python3
"""Independent CUDA format/compute checks: gguf-py, cuBLAS, PyTorch and FA4.

All comparisons name their precision. No FP8/FP4 conversion is called GGUF.
"""
import argparse
import ctypes as C
import hashlib
import importlib.metadata
import json
from pathlib import Path
import numpy as np
import torch
import gguf

P=C.c_void_p;I=C.c_int;U=C.c_uint64
class Arrangement(C.Structure):
    _fields_=[(x,I) for x in ['version','layout','bits','high_bits','artifact_tile_k','transport_tile_k','group_size','reserved']]+[('mapping_id',U)]
class Sizes(C.Structure):
    _fields_=[(x,U) for x in ['raw_bytes','low_bytes','high_bytes','units_bytes']]
assert C.sizeof(Arrangement)==40 and C.sizeof(Sizes)==32
def ptr(a):return P(a.ctypes.data) if a is not None and a.size else None

class Native:
    def __init__(self,path):
        self.lib=C.CDLL(str(path.resolve()));l=self.lib
        l.q4e_cuda_error_v1.restype=C.c_char_p
        signatures={
            'q4e_cuda_create_v1':[I,U,C.POINTER(P)],'q4e_cuda_destroy_v1':[P],
            'q4e_cuda_upload_v1':[P,I,I,I,I,P,U,C.POINTER(P)],
            'q4e_cuda_matmul_v1':[P,P,I,I,P,P],'q4e_cuda_dequant_v1':[P,P,I,P],
            'q4e_cuda_gemm_reference_v1':[P,P,I,I,P,P],
            'q4e_cuda_attention_v1':[P,P,P,P,I,I,I,I,P],
            'q4e_cuda_delta_v1':[P,P,P,P,I,I,I,P,P],
            'quactlize_ppu_kpack_canonical_arrangement_v1':[I,C.POINTER(Arrangement)],
            'quactlize_ppu_kpack_sizes_for_arrangement_v1':[I,I,I,I,C.POINTER(Arrangement),C.POINTER(Sizes)],
            'q4e_cuda_pack_v1':[P,P,C.POINTER(Arrangement),C.POINTER(P)],
            'q4e_cuda_packed_bytes_v1':[P,P,P,P,P],
            'q4e_cuda_host_pack_v1':[I,I,I,I,P,C.POINTER(Arrangement),P,P,P],
        }
        for name,args in signatures.items():getattr(l,name).argtypes=args
        l.q4e_cuda_destroy_v1.restype=None
        if l.q4e_cuda_abi_v1()!=1 or l.q4e_cuda_features_v1()&31!=31:raise RuntimeError('build with QUactlize and cuBLAS features')
        self.context=P();self.ok(l.q4e_cuda_create_v1(0,4<<30,C.byref(self.context)))
    def ok(self,rc):
        if rc:raise RuntimeError(self.lib.q4e_cuda_error_v1().decode())
    def close(self):
        if self.context:self.lib.q4e_cuda_destroy_v1(self.context);self.context=P()
    def upload(self,q,n,k,e,raw):
        handle=P();self.ok(self.lib.q4e_cuda_upload_v1(self.context,q,n,k,e,ptr(raw),raw.nbytes,C.byref(handle)));return handle
    def dequant(self,w,e,n,k):
        y=np.empty((n,k),np.float32);self.ok(self.lib.q4e_cuda_dequant_v1(self.context,w,e,ptr(y)));return y
    def mm(self,w,e,x,n,oracle=False):
        y=np.empty((len(x),n),np.float32);f=self.lib.q4e_cuda_gemm_reference_v1 if oracle else self.lib.q4e_cuda_matmul_v1
        self.ok(f(self.context,w,e,len(x),ptr(x),ptr(y)));return y

def metric(actual,expected,atol,rtol):
    a=np.asarray(actual,dtype=np.float64);b=np.asarray(expected,dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise AssertionError('nonfinite result')
    diff=np.abs(a-b);limit=atol+rtol*np.abs(b)
    if np.any(diff>limit):
        i=np.unravel_index(np.argmax(diff-limit),a.shape);raise AssertionError(f'mismatch at {i}: {a[i]} versus {b[i]} (limit {limit[i]})')
    return {'max_abs':float(diff.max()),'relative_l2':float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-30))}

def data(q,n,k,e,seed=17):
    rng=np.random.default_rng(seed);dtype=gguf.GGMLQuantizationType(q);block,size=gguf.GGML_QUANT_SIZES[dtype]
    if q in [0,1,30]:
        x=(rng.standard_normal((e,n,k))*0.1).astype('<f4')
        if q==1:return x.astype('<f2').view('u1').reshape(e,n,-1)
        if q==30:return (x.view('<u4')>>16).astype('<u2').view('u1').reshape(e,n,-1)
        return x.view('u1').reshape(e,n,-1)
    raw=rng.integers(0,256,(e,n,k//block,size),dtype=np.uint8)
    d=(rng.integers(1,17,(e,n,k//block,1))*0.0009765625).astype('<f2').view('u1')
    if q==14:raw[...,208:210]=d
    else:
        raw[...,:2]=d
        if q in [3,7,12,13]:raw[...,2:4]=(d.copy())
    return raw.reshape(e,n,-1)

def check_weight(lib,raw,q,n,k,e,label,packed=True,expert_ids=None):
    native=Native(lib)
    try:
        raw=np.ascontiguousarray(raw,dtype=np.uint8).reshape(e,n,-1);expected=gguf.quants.dequantize(raw,gguf.GGMLQuantizationType(q)).reshape(e,n,k)
        w=native.upload(q,n,k,e,raw);rows=[];ids=expert_ids or list(range(e))
        for expert in ids:
            rows.append({'operation':'raw_dequant','expert':expert,**metric(native.dequant(w,expert,n,k),expected[expert],1e-7,1e-6)})
            for m in [1,4,17,65]:
                x=np.random.default_rng(m+13).standard_normal((m,k)).astype(np.float32)*0.1
                y=native.mm(w,expert,x,n);cpu=x@expected[expert].T;gpu=native.mm(w,expert,x,n,oracle=True)
                rows.append({'operation':'raw_matmul','m':m,'expert':expert,'gguf_numpy':metric(y,cpu,2e-4,3e-4),'cublas_f32':metric(y,gpu,2e-4,3e-4)})
        pack_status='not_requested'
        if packed:
            a=Arrangement();rc=native.lib.quactlize_ppu_kpack_canonical_arrangement_v1(q,C.byref(a));sizes=Sizes()
            admitted=rc==0 and native.lib.quactlize_ppu_kpack_sizes_for_arrangement_v1(n,k,e,q,C.byref(a),C.byref(sizes))==0
            if admitted:
                p=P();native.ok(native.lib.q4e_cuda_pack_v1(native.context,w,C.byref(a),C.byref(p)))
                host=[np.zeros(getattr(sizes,f'{x}_bytes'),np.uint8) for x in ['low','high','units']];device=[np.empty_like(x) for x in host]
                native.ok(native.lib.q4e_cuda_host_pack_v1(q,n,k,e,ptr(raw),C.byref(a),*[ptr(x) for x in host]))
                native.ok(native.lib.q4e_cuda_packed_bytes_v1(native.context,p,*[ptr(x) for x in device]))
                if not all(np.array_equal(a,b) for a,b in zip(host,device)):raise AssertionError('canonical CPU/GPU packed bytes differ')
                for expert in ids:
                    rows.append({'operation':'canonical_dequant','expert':expert,**metric(native.dequant(p,expert,n,k),expected[expert],2e-6,2e-6)})
                    x=np.random.default_rng(89).standard_normal((17,k)).astype(np.float32)*0.1
                    rows.append({'operation':'canonical_matmul','m':17,'expert':expert,**metric(native.mm(p,expert,x,n),x@expected[expert].T,3e-4,5e-4)})
                wrong=Arrangement.from_buffer_copy(a);wrong.mapping_id^=1;bad=P()
                if native.lib.q4e_cuda_pack_v1(native.context,w,C.byref(wrong),C.byref(bad))==0 or bad.value:raise AssertionError('wrong mapping ID was not rejected')
                pack_status={'mapping_id':hex(a.mapping_id),'layout':a.layout,'byte_exact':True,'plane_sha256':[hashlib.sha256(x).hexdigest() for x in device]}
            else:pack_status='rejected_by_canonical_producer_raw_GGUF_path_verified'
        # Shape/type mismatches must fail before device reads.
        bad=P()
        if native.lib.q4e_cuda_upload_v1(native.context,q,n,k,e,ptr(raw),raw.nbytes-1,C.byref(bad))==0:raise AssertionError('wrong payload extent accepted')
        return {'tensor':label,'qtype':q,'shape':[e,n,k],'canonical':pack_status,'comparisons':rows}
    finally:native.close()

def check_attention(lib):
    from flash_attn.cute.interface import _flash_attn_fwd
    native=Native(lib);results=[];torch.manual_seed(47)
    try:
        for length in [1,4,17,513]:
            q=torch.randn(1,1,24,256,device='cuda',dtype=torch.bfloat16)*0.1
            k=torch.randn(1,length,2,256,device='cuda',dtype=torch.bfloat16)*0.1;v=torch.randn_like(k)
            # These rows are already the selected causal KV set; applying a new
            # contiguous causal mask after gather would change the QSA operation.
            with torch.no_grad():fa=_flash_attn_fwd(q,k,v,causal=False,tile_mn=(64,64),pack_gqa=False)[0]
            qh=q.float().cpu().numpy().reshape(24,256);kh=k.float().cpu().numpy().reshape(length,2,256);vh=v.float().cpu().numpy().reshape(length,2,256)
            out=np.empty((24,256),np.float32);native.ok(native.lib.q4e_cuda_attention_v1(native.context,ptr(qh),ptr(kh),ptr(vh),length,24,2,256,ptr(out)))
            qf=q.float().transpose(1,2);kf=k.float().repeat_interleave(12,dim=2).transpose(1,2);vf=v.float().repeat_interleave(12,dim=2).transpose(1,2)
            reference=(torch.softmax(qf@kf.transpose(-1,-2)/16.,-1)@vf).transpose(1,2).cpu().numpy().reshape(24,256)
            results.append({'selected_tokens':length,'fp32_torch':metric(out,reference,2e-6,2e-5),'fa4_bf16_output':metric(out,fa.float().cpu().numpy().reshape(24,256),8e-3,1e-2)})
        return {'provider':'flash-attn-4','version':importlib.metadata.version('flash-attn-4'),'tile':[64,64],'dtype':'BF16 QKV/output; explicit F32 oracle on rounded QKV','comparisons':results}
    finally:native.close()

def check_gdn(lib):
    native=Native(lib);rng=np.random.default_rng(71);hk,hv,d=16,48,128
    q=rng.standard_normal((hk,d)).astype(np.float32);q/=np.linalg.norm(q,axis=1,keepdims=True)
    k=rng.standard_normal((hk,d)).astype(np.float32);k/=np.linalg.norm(k,axis=1,keepdims=True)
    v=rng.standard_normal((hv,d)).astype(np.float32)*0.1;qkv=np.concatenate([q.ravel(),k.ravel(),v.ravel()])
    decay=np.linspace(.8,.99,hv,dtype=np.float32);beta=np.linspace(.1,.9,hv,dtype=np.float32);state=np.zeros((hv,d,d),np.float32);out=np.empty((hv,d),np.float32)
    reference=torch.zeros((hv,d,d),device='cuda',dtype=torch.float32);qh=torch.from_numpy(q[np.arange(hv)%hk]).cuda();kh=torch.from_numpy(k[np.arange(hv)%hk]).cuda();vh=torch.from_numpy(v).cuda();dh=torch.from_numpy(decay).cuda();bh=torch.from_numpy(beta).cuda()
    try:
        results=[]
        for step in range(9):
            native.ok(native.lib.q4e_cuda_delta_v1(native.context,ptr(qkv),ptr(decay),ptr(beta),hk,hv,d,ptr(state),ptr(out)))
            reference=reference*dh[:,None,None];prediction=(kh[:,:,None]*reference).sum(dim=1);delta=(vh-prediction)*bh[:,None];reference=reference+kh[:,:,None]*delta[:,None,:];expected=(qh[:,:,None]*reference).sum(dim=1)/(d**.5)
            results.append({'step':step,'output':metric(out,expected.cpu().numpy(),2e-6,2e-5),'state':metric(state,reference.cpu().numpy(),2e-6,2e-5)})
        return {'state_dtype':'F32','head_map':'value_head % key_heads','comparisons':results}
    finally:native.close()

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--library',type=Path,required=True);ap.add_argument('--samples',type=Path);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32=False
    result={'schema':'q4e.cuda-proxy-validation.v1','gpu':torch.cuda.get_device_name(0),'compute_capability':torch.cuda.get_device_capability(0),'torch':torch.__version__,'cuda':torch.version.cuda,'gguf_py':importlib.metadata.version('gguf'),'library_sha256':hashlib.sha256(a.library.read_bytes()).hexdigest(),'weights':[]}
    shapes=[(0,17,256,2),(1,17,256,2),(30,17,256,2),(7,2560,640,2),(8,2560,640,2),(8,1280,2560,2),(12,1280,2560,2),(13,1280,2560,2),(14,256,512,2),(20,32,160,1),(12,16,256,512)]
    for q,n,k,e in shapes:
        label=f'synthetic-{q}-{n}-{k}-{e}';result['weights'].append(check_weight(a.library,data(q,n,k,e),q,n,k,e,label,expert_ids=[0,17,511] if e==512 else None));print(label,'PASS',flush=True)
    if a.samples:
        manifest=json.loads((a.samples/'samples.json').read_text());result['sample_revision']=manifest['revision'];samples={s['tensor']:s for s in manifest['samples']}
        for s in manifest['samples']:
            raw=(a.samples/s['file']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=s['sha256']:raise RuntimeError('sample checksum mismatch')
            result['weights'].append(check_weight(a.library,np.frombuffer(raw,np.uint8),s['qtype'],s['n'],s['k'],s['experts'],s['tensor']));print(s['tensor'],'PASS',flush=True)
        for layer in [0,2,48]:
            gate=samples[f'blk.{layer}.ffn_gate_exps.weight'];up=samples[f'blk.{layer}.ffn_up_exps.weight'];e,n,k,q=(gate[x] for x in ['experts','n','k','qtype'])
            if any(gate[x]!=up[x] for x in ['experts','n','k','qtype']):raise RuntimeError('incompatible gate/up samples')
            if [x['expert'] for x in gate['ranges']]!=[x['expert'] for x in up['ranges']]:raise RuntimeError('gate/up expert order differs')
            g=np.fromfile(a.samples/gate['file'],np.uint8).reshape(e,n,-1);u=np.fromfile(a.samples/up['file'],np.uint8).reshape(e,n,-1);joined=np.concatenate([g,u],axis=1)
            result['weights'].append(check_weight(a.library,joined,q,2*n,k,e,f'blk.{layer}.gate_up_joined'));print(f'blk.{layer}.gate_up_joined','PASS',flush=True)
    result['gdn']=check_gdn(a.library);result['attention']=check_attention(a.library)
    major,minor=result['compute_capability'];sm=major*10+minor
    result['deepgemm']={'status':f'not_launched_unsupported_sm{sm}' if sm not in [90,100] else 'not_configured_for_this_run','upstream_required_sm':[90,100],'upstream_min_cuda':'12.9','upstream':'https://github.com/deepseek-ai/DeepGEMM','alternative_oracle':'cuBLAS pedantic F32 (no TF32/FP8 conversion)'}
    result['status']='passed';a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(a.out,flush=True)

if __name__=='__main__':main()
