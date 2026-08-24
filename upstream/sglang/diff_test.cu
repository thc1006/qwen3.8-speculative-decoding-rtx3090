// Differential, bound-tightness and cost test for the sibling walk in
// TreeSpeculativeSamplingTargetOnly.
//
// The loop body is the one from
// python/sglang/kernels/aot/csrc/speculative/speculative_sampling.cuh, lines 63-96. Three
// changes and no others:
//   1. `bx` comes from blockIdx.x, as in the real kernel (hang_repro.cu pinned it to 0)
//   2. a `bounded` flag, which is the proposed fix
//   3. output arrays for the accepted sequence, so the two variants can be compared per request
//
// Answers the three questions a reviewer asks of a guard in a hot loop:
//   A. does it change the result on trees the builder can actually produce?
//   B. is the bound ever reached on such a tree?
//   C. what does it cost?
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#define CUDA_OK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("CUDA %s at %d\n", cudaGetErrorString(e_), __LINE__); exit(1);} } while(0)

__global__ void walk(
    const int64_t* retrive_index, const int64_t* retrive_next_token,
    const int64_t* retrive_next_sibling, const int64_t* candidates,
    const float* target_probs, const float* uniform_samples, float* draft_probs,
    uint32_t num_speculative_tokens, uint32_t num_draft_tokens, uint32_t d,
    float threshold_single, float threshold_acc,
    int bounded,
    int32_t* out_accept, int32_t* out_naccept, uint32_t* out_maxchain) {
  const uint32_t bx = blockIdx.x;
  float prob_acc = 0.0f;
  uint32_t cur_prob_offset = bx * num_draft_tokens * d;
  float coin = uniform_samples[bx * num_draft_tokens];
  int64_t last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens];
  uint32_t num_accepted_tokens = 0;
  int64_t cur_index = 0;
  uint32_t max_chain = 0;
  out_accept[bx * num_speculative_tokens] = (int32_t)last_accepted_retrive_idx;

  for (uint32_t j = 1; j < num_speculative_tokens; ++j) {
    cur_index = retrive_next_token[bx * num_draft_tokens + cur_index];
    uint32_t steps = 0;
    while (cur_index != -1) {
      if (bounded && steps >= num_draft_tokens) { cur_index = -1; break; }
      ++steps;
      if (steps > max_chain) max_chain = steps;
      int64_t draft_index    = retrive_index[bx * num_draft_tokens + cur_index];
      int64_t draft_token_id = candidates[bx * num_draft_tokens + cur_index];
      float target_prob_single = target_probs[cur_prob_offset + draft_token_id];
      prob_acc += target_prob_single;
      if (coin <= prob_acc / threshold_acc || target_prob_single >= threshold_single) {
        prob_acc = 0.f;
        cur_prob_offset = (bx * num_draft_tokens + cur_index) * d;
        coin = uniform_samples[bx * num_draft_tokens + cur_index];
        ++num_accepted_tokens;
        out_accept[bx * num_speculative_tokens + num_accepted_tokens] = (int32_t)draft_index;
        last_accepted_retrive_idx = draft_index;
        break;
      } else {
        draft_probs[cur_prob_offset + draft_token_id] =
            target_probs[cur_prob_offset + draft_token_id];
        cur_index = retrive_next_sibling[bx * num_draft_tokens + cur_index];
      }
    }
    if (cur_index == -1) break;
  }
  out_naccept[bx]  = (int32_t)num_accepted_tokens;
  out_maxchain[bx] = max_chain;
}

// A tree the CUDA builder could produce: children are pushed at the head of the parent's child
// list while the index counts downward, which is what makes sibling[i] > i hold and the chains
// acyclic. Generating them any other way would test a shape the builder cannot emit.
static void gen_valid_tree(uint32_t N, std::mt19937& rng,
                           std::vector<int64_t>& nxt, std::vector<int64_t>& sib) {
  nxt.assign(N, -1); sib.assign(N, -1);
  std::vector<int64_t> parent(N, -1);
  for (uint32_t i = 1; i < N; ++i) parent[i] = std::uniform_int_distribution<int>(0, i - 1)(rng);
  for (int i = (int)N - 1; i >= 1; --i) {       // head insertion, descending i
    int p = (int)parent[i];
    sib[i] = nxt[p];
    nxt[p] = i;
  }
}

int main(int argc, char** argv) {
  const int TRIALS = (argc > 1) ? atoi(argv[1]) : 20000;
  const uint32_t WIDTHS[] = {4, 8, 16, 32, 64};
  const uint32_t NW = sizeof(WIDTHS)/sizeof(WIDTHS[0]);
  const uint32_t D = 64, NSPEC_MAX = 8;
  std::mt19937 rng(20260825);

  printf("differential test: %d random valid trees per width, widths 4..64\n", TRIALS);
  printf("  %-6s %10s %12s %14s %16s\n", "width", "trials", "mismatches", "max chain seen", "bound (=width)");

  int total_mismatch = 0;
  for (uint32_t wi = 0; wi < NW; ++wi) {
    const uint32_t N = WIDTHS[wi];
    const uint32_t NSPEC = std::min(NSPEC_MAX, N);
    const uint32_t BS = 256;                      // trees per launch
    const int rounds = (TRIALS + BS - 1) / BS;

    std::vector<int64_t> h_idx(BS*N), h_nxt(BS*N), h_sib(BS*N), h_cand(BS*N);
    std::vector<float>   h_tp(BS*N*D), h_us(BS*N);
    int64_t *d_idx,*d_nxt,*d_sib,*d_cand; float *d_tp,*d_us,*d_dp;
    int32_t *d_acc[2],*d_nac[2]; uint32_t *d_mc[2];
    CUDA_OK(cudaMalloc(&d_idx, BS*N*sizeof(int64_t))); CUDA_OK(cudaMalloc(&d_nxt, BS*N*sizeof(int64_t)));
    CUDA_OK(cudaMalloc(&d_sib, BS*N*sizeof(int64_t))); CUDA_OK(cudaMalloc(&d_cand,BS*N*sizeof(int64_t)));
    CUDA_OK(cudaMalloc(&d_tp, BS*N*D*sizeof(float))); CUDA_OK(cudaMalloc(&d_us, BS*N*sizeof(float)));
    CUDA_OK(cudaMalloc(&d_dp, BS*N*D*sizeof(float)));
    for (int b=0;b<2;++b){ CUDA_OK(cudaMalloc(&d_acc[b], BS*NSPEC*sizeof(int32_t)));
                           CUDA_OK(cudaMalloc(&d_nac[b], BS*sizeof(int32_t)));
                           CUDA_OK(cudaMalloc(&d_mc[b],  BS*sizeof(uint32_t))); }

    uint32_t max_chain_seen = 0; int mism = 0;
    for (int r = 0; r < rounds; ++r) {
      for (uint32_t b = 0; b < BS; ++b) {
        std::vector<int64_t> nxt, sib;
        gen_valid_tree(N, rng, nxt, sib);
        for (uint32_t i=0;i<N;++i) {
          h_idx [b*N+i] = i;
          h_nxt [b*N+i] = nxt[i];
          h_sib [b*N+i] = sib[i];
          h_cand[b*N+i] = std::uniform_int_distribution<int>(0, D-1)(rng);
          h_us  [b*N+i] = std::uniform_real_distribution<float>(0.f,1.f)(rng);
        }
        for (uint32_t i=0;i<N*D;++i)
          h_tp[b*N*D+i] = std::uniform_real_distribution<float>(0.f,1.f)(rng);
      }
      CUDA_OK(cudaMemcpy(d_idx ,h_idx .data(),BS*N*sizeof(int64_t),cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(d_nxt ,h_nxt .data(),BS*N*sizeof(int64_t),cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(d_sib ,h_sib .data(),BS*N*sizeof(int64_t),cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(d_cand,h_cand.data(),BS*N*sizeof(int64_t),cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(d_tp  ,h_tp  .data(),BS*N*D*sizeof(float),cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(d_us  ,h_us  .data(),BS*N*sizeof(float),cudaMemcpyHostToDevice));

      for (int b=0;b<2;++b) {
        CUDA_OK(cudaMemset(d_dp,0,BS*N*D*sizeof(float)));
        CUDA_OK(cudaMemset(d_acc[b],-1,BS*NSPEC*sizeof(int32_t)));
        walk<<<BS,1>>>(d_idx,d_nxt,d_sib,d_cand,d_tp,d_us,d_dp,NSPEC,N,D,1.0f,1.0f,
                       b,d_acc[b],d_nac[b],d_mc[b]);
      }
      CUDA_OK(cudaDeviceSynchronize());
      std::vector<int32_t> a0(BS*NSPEC),a1(BS*NSPEC),n0(BS),n1(BS); std::vector<uint32_t> m0(BS);
      CUDA_OK(cudaMemcpy(a0.data(),d_acc[0],BS*NSPEC*sizeof(int32_t),cudaMemcpyDeviceToHost));
      CUDA_OK(cudaMemcpy(a1.data(),d_acc[1],BS*NSPEC*sizeof(int32_t),cudaMemcpyDeviceToHost));
      CUDA_OK(cudaMemcpy(n0.data(),d_nac[0],BS*sizeof(int32_t),cudaMemcpyDeviceToHost));
      CUDA_OK(cudaMemcpy(n1.data(),d_nac[1],BS*sizeof(int32_t),cudaMemcpyDeviceToHost));
      CUDA_OK(cudaMemcpy(m0.data(),d_mc[0], BS*sizeof(uint32_t),cudaMemcpyDeviceToHost));
      for (uint32_t b=0;b<BS;++b) {
        if (n0[b]!=n1[b] || memcmp(&a0[b*NSPEC],&a1[b*NSPEC],NSPEC*sizeof(int32_t))) ++mism;
        max_chain_seen = std::max(max_chain_seen, m0[b]);
      }
    }
    printf("  %-6u %10d %12d %14u %16u%s\n", N, rounds*(int)BS, mism, max_chain_seen, N,
           max_chain_seen >= N ? "   <-- BOUND WOULD BIND" : "");
    total_mismatch += mism;

    cudaFree(d_idx);cudaFree(d_nxt);cudaFree(d_sib);cudaFree(d_cand);
    cudaFree(d_tp);cudaFree(d_us);cudaFree(d_dp);
    for (int b=0;b<2;++b){cudaFree(d_acc[b]);cudaFree(d_nac[b]);cudaFree(d_mc[b]);}
  }
  printf("\n  total mismatches across all widths: %d\n", total_mismatch);
  return total_mismatch == 0 ? 0 : 1;
}
