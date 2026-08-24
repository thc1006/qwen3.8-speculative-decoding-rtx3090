// Standalone reproduction of the tid != 0 branch of build_tree_kernel.
//
// Transcribed from python/sglang/kernels/aot/csrc/speculative/eagle_utils.cu (main), lines
// 104-122, QLEN_ONLY mask layout. The only additions are an iteration counter so a
// non-terminating walk reports itself, and the `guarded` variant, which is the proposed fix.
//
// The point of comparison is the tid == 0 branch of the same kernel, which runs the same search
// over selected_index and then checks for not-found, prints
//   "WARNING: invalid eagle tree!!! Detected a token with no parent token selected.
//    Please check if the logprob has nan. The token will be ignored to keep proceeding."
// and skips the token. The tid != 0 branch has no such check.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA error: %s (line %d)\n", cudaGetErrorString(e), __LINE__); return 1;} } while(0)

__global__ void build_walk(
    const int64_t* parent_list, const int64_t* selected_index, const int64_t* verified_seq_len,
    bool* tree_mask, int64_t* positions,
    int topk, int depth, int draft_token_num,
    int guarded, unsigned long long cap,
    unsigned long long* out_iters, int* out_hit_cap, int* out_max_offset) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;
  if (tid >= draft_token_num) return;

  int seq_len = (int)verified_seq_len[bid];
  int token_tree_idx = draft_token_num * draft_token_num * bid + draft_token_num * tid + 1;

  // The real kernel initialises every thread's mask row before the tid branch.
  tree_mask[token_tree_idx - 1] = true;
  for (int i = 0; i < draft_token_num - 1; i++) {
    tree_mask[token_tree_idx + i] = false;
  }
  if (tid == 0) {                             // the guarded branch is not under test here
    positions[bid * draft_token_num] = seq_len;
    out_iters[0] = 0; out_hit_cap[0] = 0; out_max_offset[0] = 0;
    return;
  }

  int position = 0;
  int cur_position = tid - 1;
  unsigned long long iters = 0;
  int hit_cap = 0;
  int max_offset = cur_position;

  if (!guarded) {
    // ---- as shipped on main ----
    while (true) {
      if (++iters > cap) { hit_cap = 1; break; }
      position += 1;
      if (cur_position > max_offset) max_offset = cur_position;
      tree_mask[token_tree_idx + cur_position] = true;
      int parent_tb_idx = (int)(selected_index[bid * (draft_token_num - 1) + cur_position] / topk);
      if (parent_tb_idx == 0) break;
      int64_t token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
      for (cur_position = 0; cur_position < draft_token_num; ++cur_position) {
        if (selected_index[bid * (draft_token_num - 1) + cur_position] == token_idx) break;
      }
    }
  } else {
    // ---- verbatim copy of the proposed patch ----
    while (position < depth) {
      if (++iters > cap) { hit_cap = 1; break; }        // escape only, not part of the patch
      position += 1;
      if (cur_position > max_offset) max_offset = cur_position;
      tree_mask[token_tree_idx + cur_position] = true;
      int parent_tb_idx = (int)(selected_index[bid * (draft_token_num - 1) + cur_position] / topk);
      if (parent_tb_idx == 0) break;
      int64_t token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
      int found = -1;
      for (int p = 0; p < draft_token_num - 1; ++p) {
        if (selected_index[bid * (draft_token_num - 1) + p] == token_idx) { found = p; break; }
      }
      if (found < 0) break;                              // the patch prints a warning here
      cur_position = found;
    }
  }
  positions[bid * draft_token_num + tid] = position + seq_len;
  out_iters[tid] = iters;
  out_hit_cap[tid] = hit_cap;
  out_max_offset[tid] = max_offset;
}

int main(int argc, char** argv) {
  // 0 = as shipped, 1 = with the guard, absent = both
  int only = (argc > 1) ? atoi(argv[1]) : -1;
  const int N = 4;        // draft_token_num
  const int TOPK = 2, DEPTH = 3, BS = 2;
  const unsigned long long CAP = 200000ULL;
  const int PL = TOPK * (DEPTH - 1) + 1;

  // selected_index is (BS, N-1). parent_list is (BS, PL).
  // The walk only advances while selected_index[cur_position] / topk is non-zero, so the entries
  // have to be at least topk for anything to happen; an entry below topk is the root and stops it.
  //   selected_index = {4, 2, 0}: parent_tb_idx comes out 2, 1, 0 with topk = 2.
  // Case A: the looked-up token is present, the walk reaches the root and stops.
  // Case B: the looked-up token is absent, which is exactly the state the tid == 0 branch warns
  //         about and skips.
  int64_t h_sel_A[BS*(N-1)] = {4, 2, 0,  2, 4, 0};
  int64_t h_par_A[BS*PL]    = {0, 0, 2, 0, 0,   0, 4, 0, 0, 0};
  int64_t h_sel_B[BS*(N-1)] = {4, 2, 0,  2, 4, 0};
  int64_t h_par_B[BS*PL]    = {0, 99, 99, 99, 99,  0, 99, 99, 99, 99};
  int64_t h_seq[BS] = {7, 11};

  int64_t *d_par,*d_sel,*d_seq,*d_pos; bool* d_mask;
  unsigned long long* d_it; int *d_cap,*d_max;
  CUDA_OK(cudaMalloc(&d_par, BS*PL*sizeof(int64_t)));
  CUDA_OK(cudaMalloc(&d_sel, BS*(N-1)*sizeof(int64_t)));
  CUDA_OK(cudaMalloc(&d_seq, BS*sizeof(int64_t)));
  CUDA_OK(cudaMalloc(&d_pos, BS*N*sizeof(int64_t)));
  CUDA_OK(cudaMalloc(&d_mask, N*BS*N*sizeof(bool)));      // the real QLEN_ONLY size
  CUDA_OK(cudaMalloc(&d_it,N*sizeof(unsigned long long)));
  CUDA_OK(cudaMalloc(&d_cap,N*sizeof(int))); CUDA_OK(cudaMalloc(&d_max,N*sizeof(int)));

  printf("draft_token_num %d, tree_mask holds %d bools, selected_index is %d wide per batch\n",
         N, N*BS*N, N-1);
  printf("the search loop bound is draft_token_num, one wider than selected_index\n\n");
  printf("  %-40s %9s %8s %8s\n", "case", "iters", "hit cap", "max off / row end");

  int fail = 0;
  int64_t h_sel_C[BS*(N-1)] = {4, 2, 0,  2, 4, 0};
  int64_t h_par_C[BS*PL]    = {0, 2, 2, 0, 0,   0, 4, 4, 0, 0};
  struct { const char* n; int64_t* sel; int64_t* par; } cases[] = {
    {"valid ancestor chain", h_sel_A, h_par_A},
    {"ancestor not selected", h_sel_B, h_par_B},
    {"ancestor chain loops", h_sel_C, h_par_C},
  };
  for (int c=0;c<3;++c) {
    for (int guarded=0; guarded<2; ++guarded) {
      if (only >= 0 && guarded != only) continue;
      CUDA_OK(cudaMemcpy(d_sel,cases[c].sel,BS*(N-1)*sizeof(int64_t),cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(d_par,cases[c].par,BS*PL*sizeof(int64_t),cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(d_seq,h_seq,BS*sizeof(int64_t),cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemset(d_mask,0,N*BS*N*sizeof(bool)));
      CUDA_OK(cudaMemset(d_it,0,N*sizeof(unsigned long long)));
      CUDA_OK(cudaMemset(d_cap,0,N*sizeof(int))); CUDA_OK(cudaMemset(d_max,0,N*sizeof(int)));
      build_walk<<<BS,N>>>(d_par,d_sel,d_seq,d_mask,d_pos,TOPK,DEPTH,N,guarded,CAP,d_it,d_cap,d_max);
      CUDA_OK(cudaDeviceSynchronize());
      int64_t posa[BS*N];
      unsigned long long ita[N]; int capa[N], mxa[N];
      CUDA_OK(cudaMemcpy(ita,d_it,N*sizeof(unsigned long long),cudaMemcpyDeviceToHost));
      CUDA_OK(cudaMemcpy(capa,d_cap,N*sizeof(int),cudaMemcpyDeviceToHost));
      CUDA_OK(cudaMemcpy(mxa,d_max,N*sizeof(int),cudaMemcpyDeviceToHost));
      CUDA_OK(cudaMemcpy(posa,d_pos,BS*N*sizeof(int64_t),cudaMemcpyDeviceToHost));
      unsigned long long it=0; int cap_=0, mx=0, over=0;
      for (int q=1;q<N;++q){ if(ita[q]>it) it=ita[q]; cap_|=capa[q];
                             if(mxa[q]>mx) mx=mxa[q]; if(mxa[q] > N-2) over=1; }
      char label[80];
      snprintf(label,sizeof(label),"%s%s", cases[c].n, guarded ? " [guarded]" : "");
      printf("  %-40s %9llu %8s %8d / %d   positions [", label, it, cap_?"YES":"no", mx, N-2);
      for (int q=0;q<BS*N;++q) printf("%s%lld", q?", ":"", (long long)posa[q]);
      printf("]%s\n", over ? "  <-- outside its row" : "");
      bool hm[N*BS*N];
      CUDA_OK(cudaMemcpy(hm,d_mask,N*BS*N*sizeof(bool),cudaMemcpyDeviceToHost));
      printf("      mask ");
      for (int r=0;r<BS*N;++r){ printf("["); for(int q=0;q<N;++q) printf("%d",(int)hm[r*N+q]); printf("]"); }
      printf("\n");
      if (guarded && (cap_ || over)) fail = 1;
    }
  }
  printf("\n%s\n", fail ? "THE GUARD DID NOT HOLD"
        : "the guard stops both the runaway walk and the write past the row");
  return fail;
}
