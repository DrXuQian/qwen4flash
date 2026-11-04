#pragma once
// Compile QUactlize's portable packing translation unit with CUDA. This bridge
// contains only runtime spelling aliases; no PPU/CUDA architecture macros.
#include <cuda_runtime.h>
using hggcStream_t = cudaStream_t;
using hggcError_t = cudaError_t;
constexpr auto hggcSuccess = cudaSuccess;
inline cudaError_t hggcGetLastError() { return cudaGetLastError(); }
