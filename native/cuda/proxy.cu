#include "api.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef Q4E_WITH_QUACTLIZE
#include "quactlize/packing/api.h"
#include "quactlize/packing/word_pack.hpp"
// Only the portable format traits are reused. No SM90 kernel/target header.
#include "quactlize/backends/ppu17/format.cuh"
#endif

namespace {
thread_local std::string error_message;
void require(bool b,const char* message){if(!b)throw std::runtime_error(message);}
void check(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
void blas(cublasStatus_t rc){if(rc!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("cuBLAS oracle failed: "+std::to_string(int(rc)));}
template<class F>int guard(F f){try{f();error_message.clear();return 0;}catch(const std::exception& e){error_message=e.what();return 1;}}
uint64_t product(uint64_t a,uint64_t b){require(b==0||a<=UINT64_MAX/b,"byte extent overflow");return a*b;}
struct Buffer{
    void* ptr=nullptr;uint64_t capacity=0;
    Buffer()=default;Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    ~Buffer(){if(ptr)cudaFree(ptr);}
    void reserve(uint64_t n){if(n<=capacity)return;void* p=nullptr;check(cudaMalloc(&p,n));if(ptr)check(cudaFree(ptr));ptr=p;capacity=n;}
    float* f(){return static_cast<float*>(ptr);}
};
struct Weight{
    int qtype=0,n=0,k=0,experts=0;bool packed=false;uint64_t bytes=0;
    Buffer raw,low,high,units;
#ifdef Q4E_WITH_QUACTLIZE
    quactlize_ppu_placed_arrangement_v2 arrangement{};
    quactlize_ppu_kpack_sizes_v1 sizes{};
#endif
};
struct Context{
    int device;uint64_t budget,used=0,matmuls=0,attentions=0,deltas=0,uploads=0;
    cudaStream_t stream=nullptr;cublasHandle_t handle=nullptr;
    Buffer a,b,c,d,e,f;std::vector<std::unique_ptr<Weight>> weights;
    Context(int dev,uint64_t cap):device(dev),budget(cap){check(cudaSetDevice(dev));check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));try{blas(cublasCreate(&handle));blas(cublasSetStream(handle,stream));blas(cublasSetMathMode(handle,CUBLAS_PEDANTIC_MATH));}catch(...){cudaStreamDestroy(stream);throw;}}
    ~Context(){cudaSetDevice(device);if(stream)cudaStreamSynchronize(stream);weights.clear();if(handle)cublasDestroy(handle);if(stream)cudaStreamDestroy(stream);}
    void active(){check(cudaSetDevice(device));}
    void room(uint64_t n){require(n<=budget&&used<=budget-n,"CUDA proxy weight budget exceeded");}
    void put(Buffer& b,const void* p,uint64_t n){require(p||n==0,"null host input");b.reserve(n);check(cudaMemcpyAsync(b.ptr,p,n,cudaMemcpyHostToDevice,stream));}
    void get(void* p,Buffer& b,uint64_t n){require(p||n==0,"null host output");check(cudaMemcpyAsync(p,b.ptr,n,cudaMemcpyDeviceToHost,stream));check(cudaStreamSynchronize(stream));}
};
Context& ctx(void* p){require(p,"null context");auto& c=*static_cast<Context*>(p);c.active();return c;}
Weight& weight(Context& c,void* p){auto i=std::find_if(c.weights.begin(),c.weights.end(),[&](auto& w){return w.get()==p;});require(i!=c.weights.end(),"weight belongs to another context or is invalid");return **i;}
int block_size(int q){switch(q){case 0:case 1:case 30:return 1;case 2:case 3:case 6:case 7:case 8:case 20:return 32;case 12:case 13:case 14:return 256;case 42:return 64;default:throw std::runtime_error("raw GGUF type is not supported by CUDA proxy");}}
int block_bytes(int q){switch(q){case 0:return 4;case 1:case 30:return 2;case 2:case 20:case 42:return 18;case 3:return 20;case 6:return 22;case 7:return 24;case 8:return 34;case 12:return 144;case 13:return 176;case 14:return 210;default:throw std::runtime_error("unknown GGUF block");}}
__device__ float half_at(const uint8_t* p){return __half2float(__ushort_as_half(uint16_t(p[0])|(uint16_t(p[1])<<8)));}
__device__ float raw_value(const uint8_t* data,int q,uint64_t i){
    if(q==0)return reinterpret_cast<const float*>(data)[i];
    if(q==1)return half_at(data+2*i);
    if(q==30)return __uint_as_float(uint32_t(data[2*i])<<16|uint32_t(data[2*i+1])<<24);
    if(q==8){auto p=data+(i/32)*34;return half_at(p)*float(int8_t(p[2+i%32]));}
    if(q==42){auto p=data+(i/64)*18;int j=i%64;return half_at(p)*float(((p[2+j/4]>>(2*(j%4)))&3)-1);}
    if(q==12||q==13){
        int j=i%256,g=j/32,l=j%32;auto p=data+(i/256)*(q==12?144:176);auto s=p+4;
        int sc=g<4?s[g]&63:(s[g+4]&15)|((s[g-4]>>6)<<4);
        int mn=g<4?s[g+4]&63:(s[g+4]>>4)|((s[g]>>6)<<4);
        auto qs=p+(q==12?16:48);int code=(qs[(g/2)*32+l]>>(4*(g%2)))&15;
        if(q==13)code|=((p[16+l]>>g)&1)<<4;
        return (half_at(p)*float(sc))*float(code)-half_at(p+2)*float(mn);
    }
    if(q==14){
        auto p=data+(i/256)*210;int j=i%256,half=j/128,l=j%32,g=(j%128)/32;
        int lo=half*64+l+(g%2)*32,hi=p[128+half*32+l];
        int code=((p[lo]>>(g>=2?4:0))&15)|(((hi>>(2*g))&3)<<4);
        return half_at(p+208)*float(int8_t(p[192+half*8+l/16+g*2]))*float(code-32);
    }
    int bytes=q==3?20:q==6?22:q==7?24:18;auto p=data+(i/32)*bytes;int j=i%32;
    bool affine=q==3||q==7,five=q==6||q==7;int off=affine?4:2;
    uint32_t high=five?(uint32_t(p[off])|uint32_t(p[off+1])<<8|uint32_t(p[off+2])<<16|uint32_t(p[off+3])<<24):0;
    int code=(p[off+(five?4:0)+j%16]>>(4*(j/16)))&15;if(five)code|=((high>>j)&1)<<4;
    if(q==20){constexpr int iq[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};return half_at(p)*float(iq[code]);}
    return half_at(p)*float(code-(affine?0:five?16:8))+(affine?half_at(p+2):0.f);
}
struct RawReader{const uint8_t* data;int q,k;__device__ float operator()(int row,int col)const{return raw_value(data,q,uint64_t(row)*k+col);}};
template<class Reader>__global__ void matmul_kernel(Reader w,const float* a,float* y,int n,int k){
    int row=blockIdx.x,m=blockIdx.y;float sum=0.f;
    for(int col=threadIdx.x;col<k;col+=blockDim.x)sum+=w(row,col)*a[uint64_t(m)*k+col];
    __shared__ float sums[256];sums[threadIdx.x]=sum;__syncthreads();
    for(int step=blockDim.x/2;step;step/=2){if(threadIdx.x<step)sums[threadIdx.x]+=sums[threadIdx.x+step];__syncthreads();}
    if(threadIdx.x==0)y[uint64_t(m)*n+row]=sums[0];
}
template<class Reader>__global__ void dequant_kernel(Reader w,float* y,int n,int k){
    for(uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<uint64_t(n)*k;i+=uint64_t(gridDim.x)*blockDim.x)y[i]=w(i/k,i%k);
}
RawReader raw_reader(Weight& w,int expert){int block=block_size(w.qtype),bytes=block_bytes(w.qtype);return {static_cast<const uint8_t*>(w.raw.ptr)+uint64_t(expert)*w.n*(w.k/block)*bytes,w.qtype,w.k};}
#ifdef Q4E_WITH_QUACTLIZE
template<int Q>struct PackedReader{
    const uint16_t* low;const uint16_t* high;const uint8_t* units;int n;
    __device__ float operator()(int row,int col)const{using F=quactlize::ppu17::Format<Q>;auto a=F::template affine<float>(units,n,row,col/F::group_size);return float(F::code(low,high,n,row,col))*a.scale+a.zero;}
};
template<int Q>void packed_op(Context& c,Weight& w,int expert,int m,bool decode){
    PackedReader<Q> r{static_cast<const uint16_t*>(w.low.ptr)+uint64_t(expert)*w.sizes.low_bytes/w.experts/2,w.high.ptr?static_cast<const uint16_t*>(w.high.ptr)+uint64_t(expert)*w.sizes.high_bytes/w.experts/2:nullptr,static_cast<const uint8_t*>(w.units.ptr)+uint64_t(expert)*w.sizes.units_bytes/w.experts,w.n};
    if(decode)dequant_kernel<<<std::min<uint64_t>((uint64_t(w.n)*w.k+255)/256,65535),256,0,c.stream>>>(r,c.c.f(),w.n,w.k);
    else matmul_kernel<<<dim3(w.n,m),256,0,c.stream>>>(r,c.a.f(),c.c.f(),w.n,w.k);
}
#endif
void launch(Context& c,Weight& w,int expert,int m,bool decode){
    require(expert>=0&&expert<w.experts,"expert index outside tensor");require(m>0&&m<=65535,"invalid M");
    if(!w.packed){auto r=raw_reader(w,expert);if(decode)dequant_kernel<<<std::min<uint64_t>((uint64_t(w.n)*w.k+255)/256,65535),256,0,c.stream>>>(r,c.c.f(),w.n,w.k);else matmul_kernel<<<dim3(w.n,m),256,0,c.stream>>>(r,c.a.f(),c.c.f(),w.n,w.k);}
#ifdef Q4E_WITH_QUACTLIZE
    else{switch(w.qtype){case 8:packed_op<8>(c,w,expert,m,decode);break;case 12:packed_op<12>(c,w,expert,m,decode);break;case 13:packed_op<13>(c,w,expert,m,decode);break;case 14:packed_op<14>(c,w,expert,m,decode);break;default:throw std::runtime_error("packed proxy reader not admitted for this type");}}
#endif
    check(cudaGetLastError());
}
__global__ void attention_scores(const float* q,const float* k,float* scores,int tokens,int heads,int kv,int dim){
    int h=blockIdx.x,t=blockIdx.y,kh=h/(heads/kv);float sum=0;for(int j=threadIdx.x;j<dim;j+=blockDim.x)sum+=q[h*dim+j]*k[(uint64_t(t)*kv+kh)*dim+j];
    __shared__ float s[256];s[threadIdx.x]=sum;__syncthreads();for(int r=128;r;r/=2){if(threadIdx.x<r)s[threadIdx.x]+=s[threadIdx.x+r];__syncthreads();}if(threadIdx.x==0)scores[h*tokens+t]=s[0]/sqrtf(float(dim));
}
__global__ void attention_values(float* scores,const float* v,float* out,int tokens,int heads,int kv,int dim){
    int h=blockIdx.x,kh=h/(heads/kv);__shared__ float mass;
    if(threadIdx.x==0){float top=-INFINITY;for(int t=0;t<tokens;t++)top=fmaxf(top,scores[h*tokens+t]);float sum=0;for(int t=0;t<tokens;t++){float p=expf(scores[h*tokens+t]-top);scores[h*tokens+t]=p;sum+=p;}mass=sum;}__syncthreads();
    for(int j=threadIdx.x;j<dim;j+=blockDim.x){float sum=0;for(int t=0;t<tokens;t++)sum+=(scores[h*tokens+t]/mass)*v[(uint64_t(t)*kv+kh)*dim+j];out[h*dim+j]=sum;}
}
__global__ void delta_kernel(const float* qkv,const float* decay,const float* beta,float* state,float* out,int hk,int hv,int dim){
    int h=blockIdx.x,col=threadIdx.x;if(col>=dim)return;int kh=h%hk;auto q=qkv+kh*dim;auto k=qkv+hk*dim+kh*dim;auto v=qkv+2*hk*dim+h*dim;auto s=state+uint64_t(h)*dim*dim;
    float pred=0;for(int r=0;r<dim;r++)pred+=k[r]*s[r*dim+col];float delta=(v[col]-decay[h]*pred)*beta[h];float y=0;
    for(int r=0;r<dim;r++){float next=decay[h]*s[r*dim+col]+k[r]*delta;s[r*dim+col]=next;y+=q[r]*next;}out[h*dim+col]=y/sqrtf(float(dim));
}
}

extern "C" uint32_t q4e_cuda_abi_v1(){return 1;}
extern "C" uint32_t q4e_cuda_features_v1(){return 1|2|4|16
#ifdef Q4E_WITH_QUACTLIZE
|8
#endif
;}
extern "C" const char* q4e_cuda_error_v1(){return error_message.c_str();}
extern "C" int q4e_cuda_create_v1(int device,uint64_t budget,void** output){return guard([&]{require(output&&budget,"invalid context arguments");*output=nullptr;*output=new Context(device,budget);});}
extern "C" void q4e_cuda_destroy_v1(void* context){delete static_cast<Context*>(context);}
extern "C" int q4e_cuda_upload_v1(void* p,int q,int n,int k,int experts,const void* raw,uint64_t bytes,void** result){return guard([&]{auto& c=ctx(p);require(raw&&result&&n>0&&k>0&&experts>0,"invalid upload arguments");*result=nullptr;int block=block_size(q);require(k%block==0,"quantized row is not block aligned");uint64_t expected=product(product(product(n,k/block),experts),block_bytes(q));require(bytes==expected,"GGUF type/shape/payload byte mismatch");c.room(bytes);auto w=std::make_unique<Weight>();w->qtype=q;w->n=n;w->k=k;w->experts=experts;w->bytes=bytes;w->raw.reserve(bytes);
    constexpr uint64_t chunk=16<<20;for(uint64_t off=0;off<bytes;off+=chunk){check(cudaMemcpyAsync(static_cast<uint8_t*>(w->raw.ptr)+off,static_cast<const uint8_t*>(raw)+off,std::min(chunk,bytes-off),cudaMemcpyHostToDevice,c.stream));check(cudaStreamSynchronize(c.stream));}
    *result=w.get();c.weights.push_back(std::move(w));c.used+=bytes;c.uploads++;});}
extern "C" int q4e_cuda_matmul_v1(void* p,void* w,int expert,int m,const float* input,float* output){return guard([&]{auto& c=ctx(p);auto& t=weight(c,w);require(m>0&&m<=65535,"invalid M");c.put(c.a,input,product(product(m,t.k),4));c.c.reserve(product(product(m,t.n),4));launch(c,t,expert,m,false);c.get(output,c.c,product(product(m,t.n),4));c.matmuls++;});}
extern "C" int q4e_cuda_dequant_v1(void* p,void* w,int expert,float* output){return guard([&]{auto& c=ctx(p);auto& t=weight(c,w);uint64_t bytes=product(product(t.n,t.k),4);c.c.reserve(bytes);launch(c,t,expert,1,true);c.get(output,c.c,bytes);});}
extern "C" int q4e_cuda_gemm_reference_v1(void* p,void* w,int expert,int m,const float* input,float* output){return guard([&]{auto& c=ctx(p);auto& t=weight(c,w);require(m>0&&m<=65535,"invalid M");c.c.reserve(product(product(t.n,t.k),4));launch(c,t,expert,1,true);c.put(c.a,input,product(product(m,t.k),4));c.b.reserve(product(product(m,t.n),4));float one=1,zero=0;blas(cublasSgemm(c.handle,CUBLAS_OP_T,CUBLAS_OP_N,t.n,m,t.k,&one,c.c.f(),t.k,c.a.f(),t.k,&zero,c.b.f(),t.n));c.get(output,c.b,product(product(m,t.n),4));});}
extern "C" int q4e_cuda_attention_v1(void* p,const float* q,const float* k,const float* v,int tokens,int heads,int kv,int dim,float* out){return guard([&]{auto& c=ctx(p);require(tokens>0&&tokens<=65535&&heads>0&&kv>0&&heads%kv==0&&dim>0&&dim<=256,"invalid attention shape");uint64_t qb=product(product(heads,dim),4),kb=product(product(product(tokens,kv),dim),4);c.put(c.a,q,qb);c.put(c.b,k,kb);c.put(c.c,v,kb);c.d.reserve(product(product(heads,tokens),4));c.e.reserve(qb);attention_scores<<<dim3(heads,tokens),256,0,c.stream>>>(c.a.f(),c.b.f(),c.d.f(),tokens,heads,kv,dim);attention_values<<<heads,256,0,c.stream>>>(c.d.f(),c.c.f(),c.e.f(),tokens,heads,kv,dim);check(cudaGetLastError());c.get(out,c.e,qb);c.attentions++;});}
extern "C" int q4e_cuda_delta_v1(void* p,const float* qkv,const float* decay,const float* beta,int hk,int hv,int dim,float* state,float* output){return guard([&]{auto& c=ctx(p);require(hk>0&&hv>0&&hv%hk==0&&dim>0&&dim<=256,"invalid GDN shape");uint64_t sb=product(product(product(hv,dim),dim),4),ob=product(product(hv,dim),4);c.put(c.a,qkv,product(product(2ull*hk+hv,dim),4));c.put(c.b,decay,product(hv,4));c.put(c.c,beta,product(hv,4));c.put(c.d,state,sb);c.e.reserve(ob);delta_kernel<<<hv,256,0,c.stream>>>(c.a.f(),c.b.f(),c.c.f(),c.d.f(),c.e.f(),hk,hv,dim);check(cudaGetLastError());c.get(state,c.d,sb);c.get(output,c.e,ob);c.deltas++;});}
extern "C" int q4e_cuda_stats_v1(void* p,uint64_t* counts){return guard([&]{auto& c=ctx(p);require(counts,"null stats");counts[0]=c.matmuls;counts[1]=c.attentions;counts[2]=c.deltas;counts[3]=c.used;counts[4]=c.uploads;});}

#ifdef Q4E_WITH_QUACTLIZE
// Optional format compatibility API. The original producer's descriptor and byte
// maps are authoritative; this is NOT the optimized PPU dense execution library.
extern "C" int q4e_cuda_pack_v1(void* p,void* raw,const quactlize_ppu_placed_arrangement_v2* a,void** result){return guard([&]{auto& c=ctx(p);auto& source=weight(c,raw);require(a&&result&&!source.packed,"invalid packing source");*result=nullptr;require(source.qtype==8||source.qtype==12||source.qtype==13||source.qtype==14,"packed proxy format not admitted");auto w=std::make_unique<Weight>();w->qtype=source.qtype;w->n=source.n;w->k=source.k;w->experts=source.experts;w->packed=true;w->arrangement=*a;int rc=quactlize_ppu_kpack_sizes_for_arrangement_v1(w->n,w->k,w->experts,w->qtype,a,&w->sizes);require(rc==0,"QUactlize canonical descriptor/shape mismatch");w->bytes=w->sizes.low_bytes+w->sizes.high_bytes+w->sizes.units_bytes;c.room(w->bytes);w->low.reserve(w->sizes.low_bytes);w->high.reserve(w->sizes.high_bytes);w->units.reserve(w->sizes.units_bytes);rc=quactlize_ppu_prepare_fully_quantized_dev_for_arrangement_v2(static_cast<const uint8_t*>(source.raw.ptr),static_cast<uint8_t*>(w->low.ptr),static_cast<uint8_t*>(w->high.ptr),static_cast<uint8_t*>(w->units.ptr),w->n,w->k,w->experts,w->qtype,a,c.stream);require(rc==0,"QUactlize CUDA packing launch failed");check(cudaStreamSynchronize(c.stream));*result=w.get();c.used+=w->bytes;c.weights.push_back(std::move(w));});}
extern "C" int q4e_cuda_packed_bytes_v1(void* p,void* packed,void* low,void* high,void* units){return guard([&]{auto& c=ctx(p);auto& w=weight(c,packed);require(w.packed,"weight is not canonical K-pack");c.get(low,w.low,w.sizes.low_bytes);if(w.sizes.high_bytes)c.get(high,w.high,w.sizes.high_bytes);c.get(units,w.units,w.sizes.units_bytes);});}
template<gguf_scale::KType Q>void host_pack(const uint8_t* raw,int n,int k,int experts,uint8_t* low,uint8_t* high,uint8_t* units){using U=gguf_scale::packed_unit::Unit<Q>;using C=gguf_scale::CodeTraits<Q>;for(uint64_t i=0;i<uint64_t(n)*k*experts*C::Lo::kWidth/16;i++){auto x=quactlize_pack::Plane<Q,false>::word(raw,n,k,i);low[2*i]=x;low[2*i+1]=x>>8;}if constexpr(C::kHiBytes)for(uint64_t i=0;i<uint64_t(n)*k*experts*C::Hi::kWidth/16;i++){auto x=quactlize_pack::Plane<Q,true>::word(raw,n,k,i);high[2*i]=x;high[2*i+1]=x>>8;}for(uint64_t i=0;i<uint64_t(n)*experts*(k/(256*U::kSbPerUnit));i++)quactlize_pack::metadata<Q>(raw,units,n,k,i);}
extern "C" int q4e_cuda_host_pack_v1(int q,int n,int k,int experts,const uint8_t* raw,const quactlize_ppu_placed_arrangement_v2* a,uint8_t* low,uint8_t* high,uint8_t* units){return guard([&]{quactlize_ppu_kpack_sizes_v1 s{};require(raw&&low&&units,"null host packing span");require(quactlize_ppu_kpack_sizes_for_arrangement_v1(n,k,experts,q,a,&s)==0,"invalid canonical arrangement");require(!s.high_bytes||high,"null high plane");if(q==8){for(uint64_t i=0;i<s.low_bytes/2;i++){auto x=quactlize_pack::Q8::word(raw,n,k,i);low[2*i]=x;low[2*i+1]=x>>8;}for(uint64_t i=0;i<s.units_bytes/2;i++)quactlize_pack::Q8::metadata(raw,units,n,k,i);}else if(q==12)host_pack<gguf_scale::KType::Q4_K>(raw,n,k,experts,low,high,units);else if(q==13)host_pack<gguf_scale::KType::Q5_K>(raw,n,k,experts,low,high,units);else if(q==14)host_pack<gguf_scale::KType::Q6_K>(raw,n,k,experts,low,high,units);else throw std::runtime_error("host packed type not admitted");});}
#endif
