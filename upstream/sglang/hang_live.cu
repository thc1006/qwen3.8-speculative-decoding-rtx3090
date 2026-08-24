// Standalone reproduction of the sibling walk in TreeSpeculativeSamplingTargetOnly.
//
// Transcribed verbatim from
// python/sglang/kernels/aot/csrc/speculative/speculative_sampling.cuh (main), lines 63-96,
// with two changes and no others:
//   1. an iteration counter, so a non-terminating walk reports itself instead of wedging the GPU
//   2. an optional bound, which is the proposed fix
// Types match the real instantiation: DType=float, IdType2=int64_t.
//
// Built and run without SGLang or sgl-kernel: the claim is about this loop, and the loop is
// self-contained.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); return 1;} } while(0)

__global__ void walk(
    const int64_t* retrive_index, const int64_t* retrive_next_token,
    const int64_t* retrive_next_sibling, const int64_t* candidates,
    const float* target_probs, const float* uniform_samples, float* draft_probs,
    uint32_t num_speculative_tokens, uint32_t num_draft_tokens, uint32_t d,
    float threshold_single, float threshold_acc,
    unsigned long long cap, int bounded,
    unsigned long long* out_iters, int* out_accepted, int* out_hit_cap) {
  const uint32_t bx = 0;
  float prob_acc = 0.0f;
  uint32_t cur_prob_offset = bx * num_draft_tokens * d;
  float coin = uniform_samples[bx * num_draft_tokens];
  int64_t last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens];
  uint32_t num_accepted_tokens = 0;
  int64_t cur_index = 0;
  unsigned long long iters = 0;
  int hit_cap = 0;

  for (uint32_t j = 1; j < num_speculative_tokens; ++j) {
    cur_index = retrive_next_token[bx * num_draft_tokens + cur_index];
    uint32_t steps = 0;
    while (cur_index != -1) {
      if (bounded && steps >= num_draft_tokens) { cur_index = -1; break; }   // proposed fix
      ++steps;
      if (++iters > cap) { hit_cap = 1; break; }                             // escape, not a fix
      int64_t draft_index    = retrive_index[bx * num_draft_tokens + cur_index];
      int64_t draft_token_id = candidates[bx * num_draft_tokens + cur_index];
      float target_prob_single = target_probs[cur_prob_offset + draft_token_id];
      prob_acc += target_prob_single;

      if (coin <= prob_acc / threshold_acc || target_prob_single >= threshold_single) {
        prob_acc = 0.f;
        cur_prob_offset = (bx * num_draft_tokens + cur_index) * d;
        coin = uniform_samples[bx * num_draft_tokens + cur_index];
        ++num_accepted_tokens;
        last_accepted_retrive_idx = draft_index;
        break;
      } else {
        draft_probs[cur_prob_offset + draft_token_id] =
            target_probs[cur_prob_offset + draft_token_id];
        cur_index = retrive_next_sibling[bx * num_draft_tokens + cur_index];
      }
    }
    if (hit_cap) break;
    if (cur_index == -1) break;
  }
  *out_iters = iters; *out_accepted = (int)num_accepted_tokens; *out_hit_cap = hit_cap;
  (void)last_accepted_retrive_idx;
}

// ---------------------------------------------------------------------------------------------
// The kernel above is byte-identical to hang_repro.cu's, extracted rather than retyped. The only
// difference here is the value passed for `cap`: ULLONG_MAX, so the escape hatch never fires and
// the walk runs exactly as the shipped kernel would. The point is no longer the iteration count,
// which hang_repro.cu already established; it is whether the GPU state matches what #35822
// describes -- utilisation pinned while power stays at idle.
//
// Case: "cycle 1 <-> 3, no NaN, no accept", the minimal sufficient one from the factorial.
#include <limits.h>

int main() {
  const uint32_t N = 4, D = 8, NSPEC = 3;
  const int64_t nxt[4]  = {1,-1,-1,-1};
  const int64_t sib[4]  = {-1,3,-1,1};      // 1 -> 3 -> 1
  const int64_t cand[4] = {0,1,2,3};
  const float   coin    = 1.0f;             // nothing is ever accepted

  int64_t *d_idx,*d_nxt,*d_sib,*d_cand; float *d_tp,*d_us,*d_dp;
  unsigned long long *d_it; int *d_acc,*d_cap;
  CUDA_OK(cudaMalloc(&d_idx,N*sizeof(int64_t)));  CUDA_OK(cudaMalloc(&d_nxt,N*sizeof(int64_t)));
  CUDA_OK(cudaMalloc(&d_sib,N*sizeof(int64_t)));  CUDA_OK(cudaMalloc(&d_cand,N*sizeof(int64_t)));
  CUDA_OK(cudaMalloc(&d_tp,N*D*sizeof(float)));   CUDA_OK(cudaMalloc(&d_us,N*sizeof(float)));
  CUDA_OK(cudaMalloc(&d_dp,N*D*sizeof(float)));
  CUDA_OK(cudaMalloc(&d_it,sizeof(unsigned long long)));
  CUDA_OK(cudaMalloc(&d_acc,sizeof(int)));        CUDA_OK(cudaMalloc(&d_cap,sizeof(int)));

  int64_t h_idx[N]; for (uint32_t i=0;i<N;++i) h_idx[i]=(int64_t)i;
  float h_tp[N*D];  for (uint32_t i=0;i<N*D;++i) h_tp[i]=0.0f;
  float h_us[N];    for (uint32_t i=0;i<N;++i) h_us[i]=coin;

  CUDA_OK(cudaMemcpy(d_idx,h_idx,sizeof(h_idx),cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_nxt,nxt,N*sizeof(int64_t),cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_sib,sib,N*sizeof(int64_t),cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_cand,cand,N*sizeof(int64_t),cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_tp,h_tp,sizeof(h_tp),cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_us,h_us,sizeof(h_us),cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemset(d_dp,0,N*D*sizeof(float)));

  printf("launching walk<<<1,1>>> with cap=ULLONG_MAX, bounded=0\n");
  printf("this call does not return; kill the process to release the device\n");
  fflush(stdout);

  walk<<<1,1>>>(d_idx,d_nxt,d_sib,d_cand,d_tp,d_us,d_dp,NSPEC,N,D,1.0f,1.0f,
                ULLONG_MAX,0,d_it,d_acc,d_cap);
  cudaDeviceSynchronize();

  printf("UNEXPECTED: the walk returned\n");
  return 0;
}
