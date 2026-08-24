// The search in the tid == 0 branch of build_tree_efficient reads one element past its row.
//
// Transcribed from python/sglang/kernels/aot/csrc/speculative/eagle_utils.cu (main), lines 69-92.
// selected_index holds draft_token_num - 1 entries per request, indexed 0 .. N-2 within a row,
// but the search runs `parent_position < draft_token_num`, so its last iteration reads index
// N-1. For the last request in the batch that is past the end of the tensor.
//
// The value read cannot corrupt the tree: a match there sets parent_position to N via the ++,
// which is exactly what not-found produces, so both take the warning path. The defect is the
// read itself, which is undefined behaviour and faults on the last request.
//
// PR #36201 fixes the identical off-by-one in the tid != 0 branch thirty lines below. This one
// is untouched by it.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#define CUDA_OK(x) do{cudaError_t e_=(x);if(e_!=cudaSuccess){printf("CUDA %s @%d\n",cudaGetErrorString(e_),__LINE__);exit(1);}}while(0)

__global__ void tid0_search(const int64_t* selected_index, const int64_t* parent_list,
                            int64_t* retrive_index, int64_t* retrive_next_token,
                            int64_t* retrive_next_sibling, int64_t* positions,
                            int topk, int depth, int draft_token_num, int seq_len, int bounded) {
  int bid = blockIdx.x;
  if (threadIdx.x != 0) return;
  positions[bid * draft_token_num] = seq_len;
  int retrive_index_offset = bid * draft_token_num;
  for (int i = draft_token_num - 1; i > 0; --i) {
    retrive_index[bid * draft_token_num + i] = retrive_index_offset + i;
    int parent_tb_idx = selected_index[bid * (draft_token_num - 1) + i - 1] / topk;
    int parent_position = 0;
    if (parent_tb_idx > 0) {
      int parent_token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
      // the one-character difference under test
      const int limit = bounded ? draft_token_num - 1 : draft_token_num;
      for (; parent_position < limit; ++parent_position) {
        if (selected_index[bid * (draft_token_num - 1) + parent_position] == parent_token_idx) {
          ++parent_position;
          break;
        }
      }
      if (bounded && parent_position == draft_token_num - 1) parent_position = draft_token_num;
    }
    if (parent_position == draft_token_num) continue;
    if (retrive_next_token[bid * draft_token_num + parent_position] == -1) {
      retrive_next_token[bid * draft_token_num + parent_position] = i;
    } else {
      int64_t o = retrive_next_token[bid * draft_token_num + parent_position];
      retrive_next_token[bid * draft_token_num + parent_position] = i;
      retrive_next_sibling[bid * draft_token_num + i] = o;
    }
  }
  retrive_index[bid * draft_token_num] = bid * draft_token_num;
}

int main(int argc, char** argv) {
  const int bounded = (argc > 1) ? atoi(argv[1]) : 0;
  const int BS = 1, N = 4, TOPK = 2, DEPTH = 3, SEQ = 7;   // BS = 1 so bid 0 is also the last
  // every parent lookup misses, which is what drives the search to its last iteration
  const int64_t h_sel[BS*(N-1)] = {1, 2, 3};
  const int64_t h_par[BS*(TOPK*(DEPTH-1)+1)] = {0, 99, 98, 97, 96};

  int64_t *d_sel,*d_par,*d_ri,*d_nt,*d_ns,*d_pos;
  CUDA_OK(cudaMalloc(&d_sel, BS*(N-1)*sizeof(int64_t)));     // exactly N-1 per request
  CUDA_OK(cudaMalloc(&d_par, BS*(TOPK*(DEPTH-1)+1)*sizeof(int64_t)));
  CUDA_OK(cudaMalloc(&d_ri,  BS*N*sizeof(int64_t)));
  CUDA_OK(cudaMalloc(&d_nt,  BS*N*sizeof(int64_t)));
  CUDA_OK(cudaMalloc(&d_ns,  BS*N*sizeof(int64_t)));
  CUDA_OK(cudaMalloc(&d_pos, BS*N*sizeof(int64_t)));
  CUDA_OK(cudaMemcpy(d_sel,h_sel,sizeof(h_sel),cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_par,h_par,sizeof(h_par),cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemset(d_nt,-1,BS*N*sizeof(int64_t)));
  CUDA_OK(cudaMemset(d_ns,-1,BS*N*sizeof(int64_t)));

  printf("=== tid == 0 search, %s (selected_index holds %d entries, loop limit %d) ===\n",
         bounded ? "bounded to N-1" : "as shipped, limit N", N-1, bounded ? N-1 : N);
  tid0_search<<<BS,1>>>(d_sel,d_par,d_ri,d_nt,d_ns,d_pos,TOPK,DEPTH,N,SEQ,bounded);
  cudaError_t e = cudaDeviceSynchronize();
  printf("  sync: %s\n", cudaGetErrorString(e));
  if (e == cudaSuccess) {
    int64_t nt[BS*N], ns[BS*N];
    CUDA_OK(cudaMemcpy(nt,d_nt,sizeof(nt),cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(ns,d_ns,sizeof(ns),cudaMemcpyDeviceToHost));
    printf("  retrive_next_token  ");
    for (int i=0;i<BS*N;++i) printf("%3lld ",(long long)nt[i]);
    printf("\n  retrive_next_sibling");
    for (int i=0;i<BS*N;++i) printf("%3lld ",(long long)ns[i]);
    printf("\n");
  }
  return 0;
}
