#!/usr/bin/env python3
"""Build the CUDA validation backend; optional QUactlize uses its unchanged format sources."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--nvcc',type=Path,default=Path('/usr/local/cuda/bin/nvcc'))
    ap.add_argument('--arch',default='sm_120')
    ap.add_argument('--out',type=Path,default=Path('native/build/cuda'))
    ap.add_argument('--quactlize',type=Path)
    ap.add_argument('--cutlass',type=Path)
    a=ap.parse_args();root=Path(__file__).resolve().parents[1];a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    if a.arch not in ['sm_80','sm_86','sm_89','sm_90','sm_120']:ap.error('choose an explicit supported CUDA proxy architecture')
    cmd=[str(a.nvcc),'-std=c++17','-O3','--fmad=false','--shared','-Xcompiler=-fPIC','--expt-relaxed-constexpr','--expt-extended-lambda','-lineinfo',f'-arch={a.arch}',str(root/'native/cuda/proxy.cu'),'-lcublas','-o',str(a.out/'libq4e_cuda_proxy.so')]
    dependencies={}
    if a.quactlize:
        if not a.cutlass:ap.error('--cutlass is required with --quactlize (use NVIDIA CUTLASS, not the PPU fork)')
        q=a.quactlize.resolve();cut=a.cutlass.resolve()
        for f in ['quactlize/packing/ppu_pack.cu','quactlize/packing/sizes.cpp']:
            cmd.append(str(q/f))
        cmd+=['-DQ4E_WITH_QUACTLIZE=1','-I'+str(root/'native/cuda/compat'),'-I'+str(q),'-I'+str(q/'quactlize/include'),'-I'+str(cut/'include')]
        for directory in ['quactlize/include','quactlize/packing','quactlize/fusion','quactlize/backends/ppu17']:
            for p in sorted((q/directory).rglob('*')):
                if p.is_file():dependencies[str(p.relative_to(q))]=hashlib.sha256(p.read_bytes()).hexdigest()
    subprocess.run(cmd,check=True)
    manifest={'abi':1,'arch':a.arch,'purpose':'framework_validation','command':cmd,'nvcc':subprocess.check_output([str(a.nvcc),'--version'],text=True),'quactlize_source_sha256':dependencies,'binary_sha256':hashlib.sha256((a.out/'libq4e_cuda_proxy.so').read_bytes()).hexdigest()}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(a.out/'libq4e_cuda_proxy.so')

if __name__=='__main__':main()
