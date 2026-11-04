#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Explicit synchronous host-buffer validation ABI, NOT an optimized serving ABI.
uint32_t q4e_cuda_abi_v1(void);
uint32_t q4e_cuda_features_v1(void); // 1 raw, 2 attention, 4 GDN, 8 QUactlize formats, 16 cuBLAS oracle
const char* q4e_cuda_error_v1(void);
int q4e_cuda_create_v1(int device,uint64_t weight_budget,void** context);
void q4e_cuda_destroy_v1(void* context);
int q4e_cuda_upload_v1(void* context,int qtype,int n,int k,int experts,const void* raw,uint64_t bytes,void** weight);
int q4e_cuda_matmul_v1(void* context,void* weight,int expert,int m,const float* input,float* output);
int q4e_cuda_dequant_v1(void* context,void* weight,int expert,float* output);
int q4e_cuda_gemm_reference_v1(void* context,void* weight,int expert,int m,const float* input,float* output);
int q4e_cuda_attention_v1(void* context,const float* q,const float* k,const float* v,int tokens,int heads,int kv_heads,int dim,float* output);
int q4e_cuda_delta_v1(void* context,const float* qkv,const float* decay,const float* beta,int key_heads,int value_heads,int dim,float* state,float* output);
// Counts: matmul, attention, GDN, resident weights, uploads. Synchronous query.
int q4e_cuda_stats_v1(void* context,uint64_t* counts);
#ifdef __cplusplus
}
#endif
