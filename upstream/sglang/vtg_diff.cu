// VerifyTreeGreedy before and after the sibling-walk bound, differenced on well-formed trees
// and exercised on the malformed ones. Both kernels are the real ones, extracted from the two
// source files rather than retyped.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cstring>
#include <string>
#include <cuda_runtime.h>
#define CUDA_OK(x) do{cudaError_t e_=(x);if(e_!=cudaSuccess){\
  printf("CUDA %s @%d\n",cudaGetErrorString(e_),__LINE__);exit(1);} }while(0)

namespace before {
template <typename IdType, typename IdType2>
__global__ void VerifyTreeGreedy(
    IdType* predicts,
    IdType* accept_index,
    IdType* accept_token_num,  // mutable
    IdType2* candidates,
    IdType2* retrive_index,
    IdType2* retrive_next_token,
    IdType2* retrive_next_sibling,
    IdType2* target_predict,
    uint32_t batch_size,
    uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens) {
  uint32_t bx = blockIdx.x;

  IdType2 last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens];
  accept_index[bx * num_speculative_tokens] = last_accepted_retrive_idx;
  uint32_t num_accepted_tokens = 0;
  IdType2 cur_index = 0;

  for (uint32_t j = 1; j < num_speculative_tokens; ++j) {
    cur_index = retrive_next_token[bx * num_draft_tokens + cur_index];
    while (cur_index != -1) {
      IdType2 draft_index = retrive_index[bx * num_draft_tokens + cur_index];
      IdType2 draft_token_id = candidates[bx * num_draft_tokens + cur_index];
      IdType2 target_token_id = target_predict[last_accepted_retrive_idx];

      if (draft_token_id == target_token_id) {
        // accept token
        predicts[last_accepted_retrive_idx] = target_token_id;
        ++num_accepted_tokens;
        accept_index[bx * num_speculative_tokens + num_accepted_tokens] = draft_index;
        last_accepted_retrive_idx = draft_index;
        break;
      } else {
        cur_index = retrive_next_sibling[bx * num_draft_tokens + cur_index];
      }
    }
    if (cur_index == -1) break;
  }
  accept_token_num[bx] = num_accepted_tokens;
  predicts[last_accepted_retrive_idx] = target_predict[last_accepted_retrive_idx];
}

// predicts: [tot_num_draft_tokens]
// accept_index: [bs, num_spec_step]
// accept_token_num: [bs]
// candidates: [bs, num_draft_tokens]
// retrive_index: [bs, num_draft_tokens]
// retrive_next_token: [bs, num_draft_tokens]
// retrive_next_sibling: [bs, num_draft_tokens]
// target_predict: [bs, num_draft_tokens]
}  // namespace before
namespace after {
template <typename IdType, typename IdType2>
__global__ void VerifyTreeGreedy(
    IdType* predicts,
    IdType* accept_index,
    IdType* accept_token_num,  // mutable
    IdType2* candidates,
    IdType2* retrive_index,
    IdType2* retrive_next_token,
    IdType2* retrive_next_sibling,
    IdType2* target_predict,
    uint32_t batch_size,
    uint32_t num_speculative_tokens,
    uint32_t num_draft_tokens) {
  uint32_t bx = blockIdx.x;

  IdType2 last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens];
  accept_index[bx * num_speculative_tokens] = last_accepted_retrive_idx;
  uint32_t num_accepted_tokens = 0;
  IdType2 cur_index = 0;

  for (uint32_t j = 1; j < num_speculative_tokens; ++j) {
    cur_index = retrive_next_token[bx * num_draft_tokens + cur_index];
    // A sibling chain is one node's child list, so it cannot be longer than the tree. Without
    // the bound a chain that loops back on itself never returns, and without the range check an
    // entry that is neither -1 nor a position in this row indexes outside it.
    uint32_t hops = 0;
    while (cur_index != -1) {
      if (cur_index < 0 || (uint32_t) cur_index >= num_draft_tokens ||
          hops++ >= num_draft_tokens) {
        printf(
            "WARNING: invalid eagle tree!!! The sibling chain of request %u does not terminate "
            "inside the tree. Please check if the logprob has nan. Verification stops here to "
            "keep proceeding.\n", bx);
        cur_index = -1;
        break;
      }
      IdType2 draft_index = retrive_index[bx * num_draft_tokens + cur_index];
      IdType2 draft_token_id = candidates[bx * num_draft_tokens + cur_index];
      IdType2 target_token_id = target_predict[last_accepted_retrive_idx];

      if (draft_token_id == target_token_id) {
        // accept token
        predicts[last_accepted_retrive_idx] = target_token_id;
        ++num_accepted_tokens;
        accept_index[bx * num_speculative_tokens + num_accepted_tokens] = draft_index;
        last_accepted_retrive_idx = draft_index;
        break;
      } else {
        cur_index = retrive_next_sibling[bx * num_draft_tokens + cur_index];
      }
    }
    if (cur_index == -1) break;
  }
  accept_token_num[bx] = num_accepted_tokens;
  predicts[last_accepted_retrive_idx] = target_predict[last_accepted_retrive_idx];
}

// predicts: [tot_num_draft_tokens]
// accept_index: [bs, num_spec_step]
// accept_token_num: [bs]
// candidates: [bs, num_draft_tokens]
// retrive_index: [bs, num_draft_tokens]
// retrive_next_token: [bs, num_draft_tokens]
// retrive_next_sibling: [bs, num_draft_tokens]
// target_predict: [bs, num_draft_tokens]
}  // namespace after

struct Tree { std::vector<int64_t> nxt, sib, cand, ridx, tgt; };

// first-child / next-sibling with head insertion, which is what the builder emits and what makes
// the chains strictly increasing and acyclic
static Tree gen(int N, std::mt19937& rng, bool cyclic, int bad_sibling) {
  Tree t; t.nxt.assign(N,-1); t.sib.assign(N,-1); t.cand.resize(N); t.ridx.resize(N); t.tgt.resize(N);
  std::vector<int> par(N,-1);
  for (int i=1;i<N;++i) par[i]=std::uniform_int_distribution<int>(0,i-1)(rng);
  for (int i=N-1;i>=1;--i){ t.sib[i]=t.nxt[par[i]]; t.nxt[par[i]]=i; }
  for (int i=0;i<N;++i){ t.ridx[i]=i; t.cand[i]=std::uniform_int_distribution<int>(0,50)(rng);
                         t.tgt[i]=std::uniform_int_distribution<int>(0,50)(rng); }
  if (cyclic && N>3){ t.nxt[0]=1; t.sib[1]=3; t.sib[3]=1;
                      for(int i=0;i<N;++i) t.tgt[i]=999; }      // nothing can be accepted
  if (bad_sibling>=0 && N>1){ t.nxt[0]=1; t.sib[1]=bad_sibling;
                              for(int i=0;i<N;++i) t.tgt[i]=999; }
  return t;
}

static void upload(const Tree& t, int N, int64_t* d_nxt,int64_t* d_sib,int64_t* d_cand,
                   int64_t* d_ridx,int64_t* d_tgt){
  CUDA_OK(cudaMemcpy(d_nxt ,t.nxt .data(),N*8,cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_sib ,t.sib .data(),N*8,cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_cand,t.cand.data(),N*8,cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_ridx,t.ridx.data(),N*8,cudaMemcpyHostToDevice));
  CUDA_OK(cudaMemcpy(d_tgt ,t.tgt .data(),N*8,cudaMemcpyHostToDevice));
}

int main(int argc,char** argv){
  // argv: [trials]  |  "before" <cycle|oob>  to run the unfixed kernel on one malformed tree
  const bool run_before = (argc>2 && std::string(argv[1])=="before");
  const bool want_cycle = run_before && std::string(argv[2])=="cycle";
  const int TRIALS=(argc>1 && !run_before)?atoi(argv[1]):20000;
  const int N=8, NSPEC=4;
  std::mt19937 rng(20260825);
  int64_t *d_nxt,*d_sib,*d_cand,*d_ridx,*d_tgt;
  int32_t *d_pred[2],*d_ai[2],*d_an[2];
  CUDA_OK(cudaMalloc(&d_nxt,N*8)); CUDA_OK(cudaMalloc(&d_sib,N*8));
  CUDA_OK(cudaMalloc(&d_cand,N*8));CUDA_OK(cudaMalloc(&d_ridx,N*8));
  CUDA_OK(cudaMalloc(&d_tgt,N*8));
  for(int v=0;v<2;++v){ CUDA_OK(cudaMalloc(&d_pred[v],N*4));
    CUDA_OK(cudaMalloc(&d_ai[v],NSPEC*4)); CUDA_OK(cudaMalloc(&d_an[v],4)); }

  if (run_before) {
    Tree tr=gen(N,rng,want_cycle,want_cycle?-1:99);
    upload(tr,N,d_nxt,d_sib,d_cand,d_ridx,d_tgt);
    CUDA_OK(cudaMemset(d_pred[0],0,N*4)); CUDA_OK(cudaMemset(d_ai[0],-1,NSPEC*4));
    CUDA_OK(cudaMemset(d_an[0],0,4));
    printf("  unfixed kernel on a %s tree; launching\n", want_cycle?"cyclic":"out-of-range");
    fflush(stdout);
    before::VerifyTreeGreedy<int32_t,int64_t><<<1,1>>>(d_pred[0],d_ai[0],d_an[0],d_cand,
        d_ridx,d_nxt,d_sib,d_tgt,1,NSPEC,N);
    cudaError_t e=cudaDeviceSynchronize();
    printf("  sync: %s\n", cudaGetErrorString(e));
    return 0;
  }
  int mism=0;
  for(int t=0;t<TRIALS;++t){
    Tree tr=gen(N,rng,false,-1); upload(tr,N,d_nxt,d_sib,d_cand,d_ridx,d_tgt);
    for(int v=0;v<2;++v){ CUDA_OK(cudaMemset(d_pred[v],0,N*4));
      CUDA_OK(cudaMemset(d_ai[v],-1,NSPEC*4)); CUDA_OK(cudaMemset(d_an[v],0,4)); }
    before::VerifyTreeGreedy<int32_t,int64_t><<<1,1>>>(d_pred[0],d_ai[0],d_an[0],d_cand,
        d_ridx,d_nxt,d_sib,d_tgt,1,NSPEC,N);
    after ::VerifyTreeGreedy<int32_t,int64_t><<<1,1>>>(d_pred[1],d_ai[1],d_an[1],d_cand,
        d_ridx,d_nxt,d_sib,d_tgt,1,NSPEC,N);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<int32_t> p0(N),p1(N),a0(NSPEC),a1(NSPEC),n0(1),n1(1);
    CUDA_OK(cudaMemcpy(p0.data(),d_pred[0],N*4,cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(p1.data(),d_pred[1],N*4,cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(a0.data(),d_ai[0],NSPEC*4,cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(a1.data(),d_ai[1],NSPEC*4,cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(n0.data(),d_an[0],4,cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(n1.data(),d_an[1],4,cudaMemcpyDeviceToHost));
    if(p0!=p1||a0!=a1||n0!=n1) ++mism;
  }
  printf("  well-formed trees : %d   outputs differing before/after : %d\n",TRIALS,mism);

  // the malformed cases run only on the fixed kernel; the old one does not return
  const char* names[]={"two-node sibling cycle","sibling index 99 (row holds 8)"};
  for(int c=0;c<2;++c){
    Tree tr=gen(N,rng,c==0,c==1?99:-1); upload(tr,N,d_nxt,d_sib,d_cand,d_ridx,d_tgt);
    CUDA_OK(cudaMemset(d_pred[1],0,N*4)); CUDA_OK(cudaMemset(d_ai[1],-1,NSPEC*4));
    CUDA_OK(cudaMemset(d_an[1],0,4));
    after::VerifyTreeGreedy<int32_t,int64_t><<<1,1>>>(d_pred[1],d_ai[1],d_an[1],d_cand,
        d_ridx,d_nxt,d_sib,d_tgt,1,NSPEC,N);
    cudaError_t e=cudaDeviceSynchronize();
    printf("  %-32s fixed kernel: %s\n",names[c],cudaGetErrorString(e));
  }
  return mism==0?0:1;
}
