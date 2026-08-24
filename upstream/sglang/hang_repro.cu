// Standalone reproduction of the sibling walk in TreeSpeculativeSamplingTargetOnly.
//
// Transcribed verbatim from
// python/sglang/kernels/aot/csrc/speculative/speculative_sampling.cuh (main), lines 63-96,
// with two changes and no others:
//   1. an iteration counter, so a non-terminating walk reports itself instead of wedging the GPU
//   2. an optional bound, which is the proposed fix
// Everything else, including the two writes in the accept branch, is as shipped. An earlier
// version dropped those two writes; they sit in the branch that breaks, so they could not affect
// termination, but `predicts` is indexed by a value read from `retrive_index` and that is its own
// out-of-bounds path, which dropping them would have hidden.
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
    int32_t* predicts, int32_t* accept_index,
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
        predicts[last_accepted_retrive_idx] = (int32_t)draft_token_id;
        ++num_accepted_tokens;
        accept_index[bx * num_speculative_tokens + num_accepted_tokens] = (int32_t)draft_index;
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

struct Case { const char* name; int64_t nxt[4]; int64_t sib[4]; int64_t cand[4];
              int nan_at; float coin; };

int main() {
  const uint32_t N = 4, D = 8, NSPEC = 3;
  const unsigned long long CAP = 1000000ULL;   // a valid walk needs at most N steps

  // Factorial over the two candidate causes, a cycle in the sibling chain and a NaN reaching
  // prob_acc, so neither gets credit for the other's effect. An earlier version of this test put
  // a cycle in the "NaN" case and would have supported the wrong conclusion.
  Case cases[] = {
    // acyclic sibling chains, ending at -1
    {"acyclic, no NaN, nothing accepted", {1,2,-1,-1}, {-1,3,-1,-1}, {0,1,2,3}, -1, 1.0f},
    {"acyclic, no NaN, accepts",          {1,2,-1,-1}, {-1,3,-1,-1}, {0,1,2,3}, -1, 0.0f},
    {"acyclic, WITH NaN",                 {1,-1,-1,-1},{-1,3,-1,-1},{1,1,1,1},  1, 0.5f},
    // cyclic sibling chains
    {"cycle 1 <-> 3, no NaN, no accept",  {1,-1,-1,-1},{-1,3,-1,1}, {0,1,2,3}, -1, 1.0f},
    {"cycle 1 <-> 3, no NaN, accepts",    {1,-1,-1,-1},{-1,3,-1,1}, {0,1,2,3}, -1, 0.0f},
    {"cycle 1 <-> 3, WITH NaN",           {1,-1,-1,-1},{-1,3,-1,1}, {1,1,1,1},  1, 0.5f},
    {"self sibling 1 -> 1, no NaN",       {1,-1,-1,-1},{-1,1,-1,-1},{0,1,2,3}, -1, 1.0f},
  };
  const int NC = sizeof(cases)/sizeof(cases[0]);

  int64_t *d_idx,*d_nxt,*d_sib,*d_cand; float *d_tp,*d_us,*d_dp;
  unsigned long long *d_it; int *d_acc,*d_cap; int32_t *d_pred,*d_ai;
  CUDA_OK(cudaMalloc(&d_idx,N*sizeof(int64_t)));  CUDA_OK(cudaMalloc(&d_nxt,N*sizeof(int64_t)));
  CUDA_OK(cudaMalloc(&d_sib,N*sizeof(int64_t)));  CUDA_OK(cudaMalloc(&d_cand,N*sizeof(int64_t)));
  CUDA_OK(cudaMalloc(&d_tp,N*D*sizeof(float)));   CUDA_OK(cudaMalloc(&d_us,N*sizeof(float)));
  CUDA_OK(cudaMalloc(&d_dp,N*D*sizeof(float)));
  CUDA_OK(cudaMalloc(&d_it,sizeof(unsigned long long)));
  CUDA_OK(cudaMalloc(&d_acc,sizeof(int)));        CUDA_OK(cudaMalloc(&d_cap,sizeof(int)));
  CUDA_OK(cudaMalloc(&d_pred,N*sizeof(int32_t)));  CUDA_OK(cudaMalloc(&d_ai,NSPEC*sizeof(int32_t)));

  int64_t h_idx[N]; for (uint32_t i=0;i<N;++i) h_idx[i]=(int64_t)i;
  printf("iteration cap %llu; a valid sibling walk needs at most %u steps\n\n", CAP, N);
  printf("  %-32s %14s %10s %9s   %s\n", "case", "iters (as-is)", "accepted", "hit cap", "iters (bounded)");

  int fail = 0;
  for (int c=0;c<NC;++c) {
    float h_tp[N*D]; for (uint32_t i=0;i<N*D;++i) h_tp[i]=0.0f;
    if (cases[c].nan_at >= 0) h_tp[cases[c].nan_at] = nanf("");
    if (cases[c].coin == 0.0f) for (uint32_t i=0;i<N*D;++i) h_tp[i]=1.0f;
    float h_us[N]; for (uint32_t i=0;i<N;++i) h_us[i]=cases[c].coin;

    CUDA_OK(cudaMemcpy(d_idx,h_idx,sizeof(h_idx),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_nxt,cases[c].nxt,N*sizeof(int64_t),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_sib,cases[c].sib,N*sizeof(int64_t),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_cand,cases[c].cand,N*sizeof(int64_t),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_tp,h_tp,sizeof(h_tp),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_us,h_us,sizeof(h_us),cudaMemcpyHostToDevice));

    unsigned long long it[2]; int acc[2], cap[2];
    for (int bounded=0; bounded<2; ++bounded) {
      CUDA_OK(cudaMemset(d_dp,0,N*D*sizeof(float)));
      CUDA_OK(cudaMemset(d_pred,0,N*sizeof(int32_t)));
      CUDA_OK(cudaMemset(d_ai,-1,NSPEC*sizeof(int32_t)));
      walk<<<1,1>>>(d_idx,d_nxt,d_sib,d_cand,d_tp,d_us,d_dp,d_pred,d_ai,NSPEC,N,D,1.0f,1.0f,
                    CAP,bounded,d_it,d_acc,d_cap);
      CUDA_OK(cudaDeviceSynchronize());
      CUDA_OK(cudaMemcpy(&it[bounded],d_it,sizeof(it[0]),cudaMemcpyDeviceToHost));
      CUDA_OK(cudaMemcpy(&acc[bounded],d_acc,sizeof(int),cudaMemcpyDeviceToHost));
      CUDA_OK(cudaMemcpy(&cap[bounded],d_cap,sizeof(int),cudaMemcpyDeviceToHost));
    }
    printf("  %-32s %14llu %10d %9s   %llu\n", cases[c].name, it[0], acc[0],
           cap[0] ? "YES" : "no", it[1]);
    if (cap[1]) { printf("     BOUNDED VERSION ALSO HIT THE CAP\n"); fail = 1; }
    if (!cap[0] && acc[0] != acc[1]) {
      printf("     bounded result differs on a terminating case: %d vs %d\n", acc[0], acc[1]);
      fail = 1;
    }
  }
  printf("\n%s\n", fail ? "CHECK FAILED" : "every non-terminating case is released by the bound, "
         "and terminating cases are unchanged");
  return fail;
}
