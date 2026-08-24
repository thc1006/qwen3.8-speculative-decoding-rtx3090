#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cstring>
#include <cuda_runtime.h>
enum TreeMaskMode { FULL_MASK = 0, QLEN_ONLY = 1, QLEN_ONLY_BITPACKING = 2 };
#define CUDA_OK(x) do{cudaError_t e_=(x);if(e_!=cudaSuccess){printf("CUDA %s @%d\n",cudaGetErrorString(e_),__LINE__);exit(1);} }while(0)

namespace before {
__global__ void build_tree_efficient(
    int64_t* parent_list,
    int64_t* selected_index,
    int64_t* verified_seq_len,
    bool* tree_mask,
    int64_t* positions,
    int64_t* retrive_index,
    int64_t* retrive_next_token,
    int64_t* retrive_next_sibling,
    int topk,
    int depth,
    int draft_token_num,
    int tree_mask_mode) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid >= draft_token_num) {
    return;
  }
  int seq_tree_idx = draft_token_num * draft_token_num * bid;
  for (int i = 0; i < bid; i++) {
    seq_tree_idx += verified_seq_len[i] * draft_token_num;
  }
  int seq_len = verified_seq_len[bid];
  int token_tree_idx;
  if (tree_mask_mode == FULL_MASK) {
    token_tree_idx = seq_tree_idx + (seq_len + draft_token_num) * tid + seq_len + 1;
  } else {
    token_tree_idx = draft_token_num * draft_token_num * bid + draft_token_num * tid + 1;
  }
  tree_mask[token_tree_idx - 1] = true;
  for (int i = 0; i < draft_token_num - 1; i++) {
    tree_mask[token_tree_idx + i] = false;
  }

  int position = 0;
  if (tid == 0) {
    positions[bid * draft_token_num] = seq_len;

    int retrive_index_offset = bid * draft_token_num;
    for (int i = draft_token_num - 1; i > 0; --i) {
      int current_token_idx = retrive_index_offset + i;
      retrive_index[bid * draft_token_num + i] = current_token_idx;
      int parent_tb_idx = selected_index[bid * (draft_token_num - 1) + i - 1] / topk;
      int parent_position = 0;
      if (parent_tb_idx > 0) {
        int parent_token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
        for (; parent_position < draft_token_num; ++parent_position) {
          if (selected_index[bid * (draft_token_num - 1) + parent_position] == parent_token_idx) {
            ++parent_position;
            break;
          }
        }
      }
      if (parent_position == draft_token_num) {
        printf(
            "WARNING: invalid eagle tree!!! Detected a token with no parent token selected. "
            "Please check if the logprob has nan. The token will be ignored to keep proceeding.\n");
        continue;
      }

      if (retrive_next_token[bid * draft_token_num + parent_position] == -1) {
        retrive_next_token[bid * draft_token_num + parent_position] = i;
      } else {
        int origin_next_token = retrive_next_token[bid * draft_token_num + parent_position];
        retrive_next_token[bid * draft_token_num + parent_position] = i;
        retrive_next_sibling[bid * draft_token_num + i] = origin_next_token;
      }
    }
    retrive_index[bid * draft_token_num] = bid * draft_token_num;
  } else {
    int cur_position = tid - 1;
    while (true) {
      position += 1;
      tree_mask[token_tree_idx + cur_position] = true;
      int parent_tb_idx = selected_index[bid * (draft_token_num - 1) + cur_position] / topk;
      if (parent_tb_idx == 0) {
        break;
      }

      int token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
      for (cur_position = 0; cur_position < draft_token_num; ++cur_position) {
        if (selected_index[bid * (draft_token_num - 1) + cur_position] == token_idx) {
          break;
        }
      }
    }
    positions[bid * draft_token_num + tid] = position + seq_len;
  }
}

// parent_list [bs, topk * (depth - 1) + 1)]
// selected_index [bs, draft_token_num - 1]
// verified_seq_len [bs]
// tree_mask: [draft_token*num_bytes_per_item | .. ] = [bs*draft_token*num_bytes_per_item]
// positions [bs * draft_token]
// retrive_index [bs, draft_token]
// retrive_next_token [bs, draft_token]
// retrive_next_sibling [bs, draft_token]
__global__ void build_tree_efficient_partial_packed(
    int64_t* parent_list,
    int64_t* selected_index,
    int64_t* verified_seq_len,
    uint8_t* tree_mask,
    int64_t* positions,
    int64_t* retrive_index,
    int64_t* retrive_next_token,
    int64_t* retrive_next_sibling,
    int topk,
    int depth,
    int draft_token_num,
    size_t num_bytes_per_item) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid >= draft_token_num) {
    return;
  }
  int seq_len = verified_seq_len[bid];
  int token_tree_idx = (bid * draft_token_num + tid) * num_bytes_per_item;
  tree_mask[token_tree_idx] = 1;  // little endian

  int position = 0;
  if (tid == 0) {
    positions[bid * draft_token_num] = seq_len;

    int retrive_index_offset = bid * draft_token_num;
    for (int i = draft_token_num - 1; i > 0; --i) {
      int current_token_idx = retrive_index_offset + i;
      retrive_index[bid * draft_token_num + i] = current_token_idx;
      int parent_tb_idx = selected_index[bid * (draft_token_num - 1) + i - 1] / topk;
      int parent_position = 0;
      if (parent_tb_idx > 0) {
        int parent_token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
        for (; parent_position < draft_token_num; ++parent_position) {
          if (selected_index[bid * (draft_token_num - 1) + parent_position] == parent_token_idx) {
            ++parent_position;
            break;
          }
        }
      }
      if (parent_position == draft_token_num) {
        printf(
            "WARNING: invalid eagle tree!!! Detected a token with no parent token selected. "
            "Please check if the logprob has nan. The token will be ignored to keep proceeding.\n");
        continue;
      }

      if (retrive_next_token[bid * draft_token_num + parent_position] == -1) {
        retrive_next_token[bid * draft_token_num + parent_position] = i;
      } else {
        int origin_next_token = retrive_next_token[bid * draft_token_num + parent_position];
        retrive_next_token[bid * draft_token_num + parent_position] = i;
        retrive_next_sibling[bid * draft_token_num + i] = origin_next_token;
      }
    }
    retrive_index[bid * draft_token_num] = bid * draft_token_num;
  } else {
    int cur_position = tid - 1;
    while (true) {
      position += 1;
      int byte_idx = (cur_position + 1) / 8;
      int bit_idx = (cur_position + 1) % 8;
      tree_mask[token_tree_idx + byte_idx] |= (1 << bit_idx);
      int parent_tb_idx = selected_index[bid * (draft_token_num - 1) + cur_position] / topk;
      if (parent_tb_idx == 0) {
        break;
      }

      int token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
      for (cur_position = 0; cur_position < draft_token_num; ++cur_position) {
        if (selected_index[bid * (draft_token_num - 1) + cur_position] == token_idx) {
          break;
        }
      }
    }
    positions[bid * draft_token_num + tid] = position + seq_len;
  }
}

}  // namespace before
namespace after {
__device__ __forceinline__ int find_selected_position(
    const int64_t* selected_index, int sel_off, int sel_stride, int64_t token_idx) {
  for (int p = 0; p < sel_stride; ++p) {
    if (selected_index[sel_off + p] == token_idx) {
      return p;
    }
  }
  return -1;
}

__global__ void build_tree_efficient(
    int64_t* parent_list,
    int64_t* selected_index,
    int64_t* verified_seq_len,
    bool* tree_mask,
    int64_t* positions,
    int64_t* retrive_index,
    int64_t* retrive_next_token,
    int64_t* retrive_next_sibling,
    int topk,
    int depth,
    int draft_token_num,
    int tree_mask_mode) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid >= draft_token_num) {
    return;
  }
  int seq_tree_idx = draft_token_num * draft_token_num * bid;
  for (int i = 0; i < bid; i++) {
    seq_tree_idx += verified_seq_len[i] * draft_token_num;
  }
  int seq_len = verified_seq_len[bid];
  int token_tree_idx;
  if (tree_mask_mode == FULL_MASK) {
    token_tree_idx = seq_tree_idx + (seq_len + draft_token_num) * tid + seq_len + 1;
  } else {
    token_tree_idx = draft_token_num * draft_token_num * bid + draft_token_num * tid + 1;
  }
  tree_mask[token_tree_idx - 1] = true;
  for (int i = 0; i < draft_token_num - 1; i++) {
    tree_mask[token_tree_idx + i] = false;
  }

  int position = 0;
  if (tid == 0) {
    positions[bid * draft_token_num] = seq_len;

    int retrive_index_offset = bid * draft_token_num;
    for (int i = draft_token_num - 1; i > 0; --i) {
      int current_token_idx = retrive_index_offset + i;
      retrive_index[bid * draft_token_num + i] = current_token_idx;
      int parent_tb_idx = selected_index[bid * (draft_token_num - 1) + i - 1] / topk;
      int parent_position = 0;
      if (parent_tb_idx > 0) {
        int parent_token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
        const int found = find_selected_position(
            selected_index, bid * (draft_token_num - 1), draft_token_num - 1, parent_token_idx);
        if (found < 0) {
          printf(
              "WARNING: invalid eagle tree!!! Detected a token with no parent token selected. "
              "Please check if the logprob has nan. The token will be ignored to keep proceeding.\n");
          continue;
        }
        parent_position = found + 1;
      }

      if (retrive_next_token[bid * draft_token_num + parent_position] == -1) {
        retrive_next_token[bid * draft_token_num + parent_position] = i;
      } else {
        int origin_next_token = retrive_next_token[bid * draft_token_num + parent_position];
        retrive_next_token[bid * draft_token_num + parent_position] = i;
        retrive_next_sibling[bid * draft_token_num + i] = origin_next_token;
      }
    }
    retrive_index[bid * draft_token_num] = bid * draft_token_num;
  } else {
    int cur_position = tid - 1;
    // a malformed tree can loop back on itself and never reach the root
    while (position < depth) {
      position += 1;
      tree_mask[token_tree_idx + cur_position] = true;
      int parent_tb_idx = selected_index[bid * (draft_token_num - 1) + cur_position] / topk;
      if (parent_tb_idx == 0) {
        break;
      }

      int token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
      const int found = find_selected_position(
          selected_index, bid * (draft_token_num - 1), draft_token_num - 1, token_idx);
      if (found < 0) {
        printf(
            "WARNING: invalid eagle tree!!! Detected a token whose ancestor was not selected. "
            "Please check if the logprob has nan. The walk stops here to keep proceeding.\n");
        break;
      }
      cur_position = found;
    }
    positions[bid * draft_token_num + tid] = position + seq_len;
  }
}

// parent_list [bs, topk * (depth - 1) + 1)]
// selected_index [bs, draft_token_num - 1]
// verified_seq_len [bs]
// tree_mask: [draft_token*num_bytes_per_item | .. ] = [bs*draft_token*num_bytes_per_item]
// positions [bs * draft_token]
// retrive_index [bs, draft_token]
// retrive_next_token [bs, draft_token]
// retrive_next_sibling [bs, draft_token]
__global__ void build_tree_efficient_partial_packed(
    int64_t* parent_list,
    int64_t* selected_index,
    int64_t* verified_seq_len,
    uint8_t* tree_mask,
    int64_t* positions,
    int64_t* retrive_index,
    int64_t* retrive_next_token,
    int64_t* retrive_next_sibling,
    int topk,
    int depth,
    int draft_token_num,
    size_t num_bytes_per_item) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (tid >= draft_token_num) {
    return;
  }
  int seq_len = verified_seq_len[bid];
  int token_tree_idx = (bid * draft_token_num + tid) * num_bytes_per_item;
  tree_mask[token_tree_idx] = 1;  // little endian

  int position = 0;
  if (tid == 0) {
    positions[bid * draft_token_num] = seq_len;

    int retrive_index_offset = bid * draft_token_num;
    for (int i = draft_token_num - 1; i > 0; --i) {
      int current_token_idx = retrive_index_offset + i;
      retrive_index[bid * draft_token_num + i] = current_token_idx;
      int parent_tb_idx = selected_index[bid * (draft_token_num - 1) + i - 1] / topk;
      int parent_position = 0;
      if (parent_tb_idx > 0) {
        int parent_token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
        const int found = find_selected_position(
            selected_index, bid * (draft_token_num - 1), draft_token_num - 1, parent_token_idx);
        if (found < 0) {
          printf(
              "WARNING: invalid eagle tree!!! Detected a token with no parent token selected. "
              "Please check if the logprob has nan. The token will be ignored to keep proceeding.\n");
          continue;
        }
        parent_position = found + 1;
      }

      if (retrive_next_token[bid * draft_token_num + parent_position] == -1) {
        retrive_next_token[bid * draft_token_num + parent_position] = i;
      } else {
        int origin_next_token = retrive_next_token[bid * draft_token_num + parent_position];
        retrive_next_token[bid * draft_token_num + parent_position] = i;
        retrive_next_sibling[bid * draft_token_num + i] = origin_next_token;
      }
    }
    retrive_index[bid * draft_token_num] = bid * draft_token_num;
  } else {
    int cur_position = tid - 1;
    // a malformed tree can loop back on itself and never reach the root
    while (position < depth) {
      position += 1;
      int byte_idx = (cur_position + 1) / 8;
      int bit_idx = (cur_position + 1) % 8;
      tree_mask[token_tree_idx + byte_idx] |= (1 << bit_idx);
      int parent_tb_idx = selected_index[bid * (draft_token_num - 1) + cur_position] / topk;
      if (parent_tb_idx == 0) {
        break;
      }

      int token_idx = parent_list[bid * (topk * (depth - 1) + 1) + parent_tb_idx];
      const int found = find_selected_position(
          selected_index, bid * (draft_token_num - 1), draft_token_num - 1, token_idx);
      if (found < 0) {
        printf(
            "WARNING: invalid eagle tree!!! Detected a token whose ancestor was not selected. "
            "Please check if the logprob has nan. The walk stops here to keep proceeding.\n");
        break;
      }
      cur_position = found;
    }
    positions[bid * draft_token_num + tid] = position + seq_len;
  }
}

}  // namespace after

// A tree the producer can emit: every node's parent sits strictly earlier in selected_index, so
// the ancestor chain is strictly decreasing and cannot cycle. Feeding a cycle here would hang
// the "before" kernel, which is the defect under test, not the thing being differenced.
static void gen(int N, int topk, int pstride, std::mt19937& rng,
                std::vector<int64_t>& sel, std::vector<int64_t>& par) {
  sel.assign(N - 1, 0); par.assign(pstride, 0);
  for (int p = 0; p < N - 1; ++p) sel[p] = (int64_t) p * topk;   // sel[p] / topk == p
  for (int i = 1; i < N - 1; ++i) {                              // node i+1's lookup slot is i
    int j = std::uniform_int_distribution<int>(0, i - 1)(rng);   // strictly earlier
    if (i < pstride) par[i] = sel[j];
  }
}

int main(int argc, char** argv) {
  const int TRIALS = (argc > 1) ? atoi(argv[1]) : 4000;
  const int BS = 2, N = 8, TOPK = 2, DEPTH = 8;
  const int PSTRIDE = TOPK * (DEPTH - 1) + 1;
  std::mt19937 rng(20260825);
  size_t mask_n = (size_t) BS * (N + 32) * N;   // enough for FULL_MASK's seq_len rows

  int64_t *d_par, *d_sel, *d_len, *d_pos[2], *d_ri[2], *d_nt[2], *d_ns[2];
  bool *d_mask[2];
  CUDA_OK(cudaMalloc(&d_par, (size_t) BS * PSTRIDE * 8));
  CUDA_OK(cudaMalloc(&d_sel, (size_t) BS * (N - 1) * 8));
  CUDA_OK(cudaMalloc(&d_len, (size_t) BS * 8));
  for (int v = 0; v < 2; ++v) {
    CUDA_OK(cudaMalloc(&d_mask[v], mask_n));
    CUDA_OK(cudaMalloc(&d_pos[v], (size_t) BS * N * 8));
    CUDA_OK(cudaMalloc(&d_ri[v],  (size_t) BS * N * 8));
    CUDA_OK(cudaMalloc(&d_nt[v],  (size_t) BS * N * 8));
    CUDA_OK(cudaMalloc(&d_ns[v],  (size_t) BS * N * 8));
  }

  const int MODES[3] = {FULL_MASK, QLEN_ONLY, QLEN_ONLY_BITPACKING};
  const char* MODE_NAME[3] = {"FULL_MASK", "QLEN_ONLY", "QLEN_ONLY_BITPACKING"};
  std::vector<int64_t> hp(BS * PSTRIDE), hs(BS * (N - 1)), hl(BS, 5);
  int total_mism = 0;
  for (int mi = 0; mi < 3; ++mi) {
  const int MODE = MODES[mi];
  int mism = 0;
  for (int t = 0; t < TRIALS; ++t) {
    for (int b = 0; b < BS; ++b) {
      std::vector<int64_t> sel, par; gen(N, TOPK, PSTRIDE, rng, sel, par);
      memcpy(&hs[b * (N - 1)], sel.data(), (N - 1) * 8);
      memcpy(&hp[b * PSTRIDE], par.data(), PSTRIDE * 8);
      hl[b] = 3 + (t + b) % 9;
    }
    CUDA_OK(cudaMemcpy(d_par, hp.data(), hp.size() * 8, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_sel, hs.data(), hs.size() * 8, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_len, hl.data(), hl.size() * 8, cudaMemcpyHostToDevice));
    size_t this_mask_n = mask_n;
    if (MODE == FULL_MASK) {
      this_mask_n = 0;
      for (int b = 0; b < BS; ++b) this_mask_n += (size_t)(hl[b] + N) * N;
    } else if (MODE == QLEN_ONLY_BITPACKING) {
      this_mask_n = (size_t) BS * N * (N > 16 ? 4 : N > 8 ? 2 : 1);
    }
    for (int v = 0; v < 2; ++v) {
      CUDA_OK(cudaMemset(d_mask[v], 0, mask_n));
      CUDA_OK(cudaMemset(d_pos[v], 0, (size_t) BS * N * 8));
      CUDA_OK(cudaMemset(d_ri[v], -1, (size_t) BS * N * 8));
      CUDA_OK(cudaMemset(d_nt[v], -1, (size_t) BS * N * 8));
      CUDA_OK(cudaMemset(d_ns[v], -1, (size_t) BS * N * 8));
    }
    if (MODE == QLEN_ONLY_BITPACKING) {
      size_t bpi = (N > 16 ? 4 : N > 8 ? 2 : 1);
      before::build_tree_efficient_partial_packed<<<BS, N>>>(d_par, d_sel, d_len,
          (uint8_t*) d_mask[0], d_pos[0], d_ri[0], d_nt[0], d_ns[0], TOPK, DEPTH, N, bpi);
      after ::build_tree_efficient_partial_packed<<<BS, N>>>(d_par, d_sel, d_len,
          (uint8_t*) d_mask[1], d_pos[1], d_ri[1], d_nt[1], d_ns[1], TOPK, DEPTH, N, bpi);
    } else {
      before::build_tree_efficient<<<BS, N>>>(d_par, d_sel, d_len, d_mask[0], d_pos[0],
          d_ri[0], d_nt[0], d_ns[0], TOPK, DEPTH, N, MODE);
      after ::build_tree_efficient<<<BS, N>>>(d_par, d_sel, d_len, d_mask[1], d_pos[1],
          d_ri[1], d_nt[1], d_ns[1], TOPK, DEPTH, N, MODE);
    }
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<int64_t> a(BS * N), b2(BS * N);
    std::vector<uint8_t> m0(mask_n), m1(mask_n);
    bool bad = false;
    CUDA_OK(cudaMemcpy(m0.data(), d_mask[0], this_mask_n, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(m1.data(), d_mask[1], this_mask_n, cudaMemcpyDeviceToHost));
    if (memcmp(m0.data(), m1.data(), this_mask_n)) bad = true;
    for (int k = 0; k < 4 && !bad; ++k) {
      int64_t *p0 = (k==0?d_pos[0]:k==1?d_ri[0]:k==2?d_nt[0]:d_ns[0]);
      int64_t *p1 = (k==0?d_pos[1]:k==1?d_ri[1]:k==2?d_nt[1]:d_ns[1]);
      CUDA_OK(cudaMemcpy(a.data(),  p0, (size_t) BS*N*8, cudaMemcpyDeviceToHost));
      CUDA_OK(cudaMemcpy(b2.data(), p1, (size_t) BS*N*8, cudaMemcpyDeviceToHost));
      if (memcmp(a.data(), b2.data(), (size_t) BS*N*8)) bad = true;
    }
    if (bad) ++mism;
  }
  printf("  %-22s %6d trees   differing outputs: %d\n", MODE_NAME[mi], TRIALS, mism);
  total_mism += mism;
  }
  printf("\n  bs=%d draft_token_num=%d depth=%d, upstream main against this PR\n", BS, N, DEPTH);
  printf("  total differing across all three mask modes: %d\n", total_mism);
  return total_mism == 0 ? 0 : 1;
}
