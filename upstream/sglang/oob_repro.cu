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

// ---------------------------------------------------------------------------------------------
// Same kernel, extracted not retyped. This case asks a different question from hang_repro.cu.
//
// The walk's only exit test is `cur_index != -1`. It never checks that cur_index is a valid index
// into the row. A sibling entry that is neither -1 nor in [0, num_draft_tokens) therefore indexes
// retrive_index, candidates and target_probs out of bounds, and the else branch WRITES to
// draft_probs at an offset derived from an unvalidated candidates value.
//
// Run this under compute-sanitizer. The iteration cap is small, so the kernel terminates and the
// question is only whether the accesses it makes are legal.
struct OobCase { const char* name; int64_t nxt[4]; int64_t sib[4]; int64_t cand[4]; };

int main(int argc, char** argv) {
  const uint32_t N = 4, D = 8, NSPEC = 3;
  const unsigned long long CAP = 8ULL;
  const int only = (argc > 1) ? atoi(argv[1]) : -1;   // terminate quickly; OOB is the subject, not liveness

  OobCase cases[] = {
    // control: every sibling entry is -1 or a valid index
    {"in-range sibling chain",        {1,2,-1,-1}, {-1,3,-1,-1}, {0,1,2,3}},
    // sibling points outside the row
    {"sibling index 99, row holds 4", {1,-1,-1,-1},{-1,99,-1,-1},{0,1,2,3}},
    // sibling is a large negative that is not the -1 sentinel
    {"sibling index -7, not -1",      {1,-1,-1,-1},{-1,-7,-1,-1},{0,1,2,3}},
    // in-range walk, but a candidate token id outside the vocab dimension D
    {"candidate token id 4096, D=8",  {1,2,-1,-1}, {-1,3,-1,-1}, {0,4096,2,3}},
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
  CUDA_OK(cudaMemset(d_pred,0,N*sizeof(int32_t))); CUDA_OK(cudaMemset(d_ai,-1,NSPEC*sizeof(int32_t)));

  int64_t h_idx[N]; for (uint32_t i=0;i<N;++i) h_idx[i]=(int64_t)i;
  float h_tp[N*D];  for (uint32_t i=0;i<N*D;++i) h_tp[i]=0.0f;
  float h_us[N];    for (uint32_t i=0;i<N;++i) h_us[i]=1.0f;   // accept nothing, keep walking

  for (int c=0;c<NC;++c) {
    if (only >= 0 && c != only) continue;
    CUDA_OK(cudaMemcpy(d_idx,h_idx,sizeof(h_idx),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_nxt,cases[c].nxt,N*sizeof(int64_t),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_sib,cases[c].sib,N*sizeof(int64_t),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_cand,cases[c].cand,N*sizeof(int64_t),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_tp,h_tp,sizeof(h_tp),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_us,h_us,sizeof(h_us),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(d_dp,0,N*D*sizeof(float)));

    printf("\n=== %s ===\n", cases[c].name); fflush(stdout);
    walk<<<1,1>>>(d_idx,d_nxt,d_sib,d_cand,d_tp,d_us,d_dp,d_pred,d_ai,NSPEC,N,D,1.0f,1.0f,
                  CAP,0,d_it,d_acc,d_cap);
    cudaError_t e = cudaDeviceSynchronize();
    printf("  sync: %s\n", cudaGetErrorString(e)); fflush(stdout);
  }
  return 0;
}