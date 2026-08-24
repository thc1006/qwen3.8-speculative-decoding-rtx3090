// What the guard costs, measured at the shipped launch configuration.
//
// The dispatch in speculative_sampling.cuh is nblks(batch_size), nthrs(1024), and the walk is
// not guarded by `tx == 0`, so all 1024 threads run it redundantly. Timing it at <<<1,1>>> would
// understate both the work and the guard, so the shapes here are the real ones.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>
#include <cuda_runtime.h>
#define CUDA_OK(x) do{cudaError_t e_=(x);if(e_!=cudaSuccess){printf("CUDA %s @%d\n",cudaGetErrorString(e_),__LINE__);exit(1);}}while(0)

__global__ void walk(const int64_t* ri,const int64_t* nt,const int64_t* ns,const int64_t* cd,
                     const float* tp,const float* us,float* dp,
                     uint32_t nspec,uint32_t ndraft,uint32_t d,float ts,float ta,int bounded,
                     int32_t* out_n){
  const uint32_t bx=blockIdx.x;
  float prob_acc=0.f; uint32_t cpo=bx*ndraft*d; float coin=us[bx*ndraft];
  int64_t last=ri[bx*ndraft]; uint32_t nacc=0; int64_t cur=0;
  for(uint32_t j=1;j<nspec;++j){
    cur=nt[bx*ndraft+cur]; uint32_t steps=0;
    while(cur!=-1){
      if(bounded && steps>=ndraft){cur=-1;break;}
      ++steps;
      int64_t di=ri[bx*ndraft+cur]; int64_t tid=cd[bx*ndraft+cur];
      float p=tp[cpo+tid]; prob_acc+=p;
      if(coin<=prob_acc/ta||p>=ts){prob_acc=0.f;cpo=(bx*ndraft+cur)*d;coin=us[bx*ndraft+cur];
        ++nacc;last=di;break;}
      else{dp[cpo+tid]=tp[cpo+tid];cur=ns[bx*ndraft+cur];}
    }
    if(cur==-1)break;
  }
  if(threadIdx.x==0) out_n[bx]=(int32_t)nacc+(int32_t)(last&0);
}

static void gen(uint32_t N,std::mt19937&rng,std::vector<int64_t>&nx,std::vector<int64_t>&sb){
  nx.assign(N,-1); sb.assign(N,-1); std::vector<int> par(N,-1);
  for(uint32_t i=1;i<N;++i) par[i]=std::uniform_int_distribution<int>(0,i-1)(rng);
  for(int i=(int)N-1;i>=1;--i){int p=par[i]; sb[i]=nx[p]; nx[p]=i;}
}

int main(){
  const uint32_t D=64, BLOCK=1024, ITERS=300, WARM=50;
  const uint32_t BATCHES[]={1,8,32,64,128,256};
  const uint32_t WIDTHS[]={8,16,64};
  std::mt19937 rng(20260825);
  printf("shipped launch shape: nthrs=%u, walk runs on every thread\n", BLOCK);
  printf("%u timed iterations after %u warmup, per configuration\n\n", ITERS, WARM);
  printf("  %-8s %-8s %14s %14s %12s\n","batch","width","as-is (us)","bounded (us)","delta");

  for(uint32_t wi=0;wi<3;++wi) for(uint32_t bi=0;bi<6;++bi){
    const uint32_t N=WIDTHS[wi], BS=BATCHES[bi], NSPEC=std::min(8u,N);
    std::vector<int64_t> hi(BS*N),hn(BS*N),hs(BS*N),hc(BS*N);
    std::vector<float> ht(BS*N*D),hu(BS*N);
    for(uint32_t b=0;b<BS;++b){ std::vector<int64_t> nx,sb; gen(N,rng,nx,sb);
      for(uint32_t i=0;i<N;++i){hi[b*N+i]=i;hn[b*N+i]=nx[i];hs[b*N+i]=sb[i];
        hc[b*N+i]=std::uniform_int_distribution<int>(0,D-1)(rng);
        hu[b*N+i]=std::uniform_real_distribution<float>(0,1)(rng);}
      for(uint32_t i=0;i<N*D;++i) ht[b*N*D+i]=std::uniform_real_distribution<float>(0,1)(rng); }
    int64_t *di,*dn,*ds,*dc; float *dt,*du,*dp; int32_t* on;
    CUDA_OK(cudaMalloc(&di,BS*N*8));CUDA_OK(cudaMalloc(&dn,BS*N*8));
    CUDA_OK(cudaMalloc(&ds,BS*N*8));CUDA_OK(cudaMalloc(&dc,BS*N*8));
    CUDA_OK(cudaMalloc(&dt,BS*N*D*4));CUDA_OK(cudaMalloc(&du,BS*N*4));
    CUDA_OK(cudaMalloc(&dp,BS*N*D*4));CUDA_OK(cudaMalloc(&on,BS*4));
    CUDA_OK(cudaMemcpy(di,hi.data(),BS*N*8,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dn,hn.data(),BS*N*8,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(ds,hs.data(),BS*N*8,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dc,hc.data(),BS*N*8,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dt,ht.data(),BS*N*D*4,cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(du,hu.data(),BS*N*4,cudaMemcpyHostToDevice));
    float ms[2];
    for(int b=0;b<2;++b){
      cudaEvent_t s,e; CUDA_OK(cudaEventCreate(&s)); CUDA_OK(cudaEventCreate(&e));
      for(uint32_t k=0;k<WARM;++k) walk<<<BS,BLOCK>>>(di,dn,ds,dc,dt,du,dp,NSPEC,N,D,1.f,1.f,b,on);
      CUDA_OK(cudaDeviceSynchronize());
      CUDA_OK(cudaEventRecord(s));
      for(uint32_t k=0;k<ITERS;++k) walk<<<BS,BLOCK>>>(di,dn,ds,dc,dt,du,dp,NSPEC,N,D,1.f,1.f,b,on);
      CUDA_OK(cudaEventRecord(e)); CUDA_OK(cudaEventSynchronize(e));
      CUDA_OK(cudaEventElapsedTime(&ms[b],s,e));
      cudaEventDestroy(s); cudaEventDestroy(e);
    }
    double a=ms[0]*1000.0/ITERS, g=ms[1]*1000.0/ITERS;
    printf("  %-8u %-8u %14.2f %14.2f %11.1f%%\n",BS,N,a,g,(g-a)/a*100.0);
    cudaFree(di);cudaFree(dn);cudaFree(ds);cudaFree(dc);
    cudaFree(dt);cudaFree(du);cudaFree(dp);cudaFree(on);
  }
  return 0;
}
