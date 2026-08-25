// Ordering probe for llama.cpp `output_reorder()`. Paste into tests/test-llama-archs.cpp above
// main(), and call it from main() behind an environment variable:
//
//     if (getenv("LAYER_INP_PROBE")) { return probe_layer_inp_order(seed); }
//
// It asks one question: after a decode, is row i of embd_layer_inp the layer-0 input of the token
// the caller put at batch position i? That is what common/speculative.cpp:1115 reads it as.
//
// The answer is decidable without knowing any model arithmetic, because for LLM_ARCH_LLAMA
// t_layer_inp[0] is inpL straight out of build_inp_embd (src/models/llama.cpp:108,127): a pure
// embedding lookup, so a row depends on the token and on nothing else, and two rows holding the
// same token are bit-identical. The reference table is built by decoding every token id once in
// one sequence, where nothing is permuted.
//
// Not for commit to llama.cpp. This lives here so the numbers in the PR notes can be re-derived.

static int probe_layer_inp_order(const size_t seed) {
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    GGML_ASSERT(cpu != nullptr);

    gguf_context_ptr gguf_ctx = get_gguf_ctx(LLM_ARCH_LLAMA, false);

    llama_model_params mparams = llama_model_default_params();
    mparams.progress_callback = silent_model_load_progress;
    std::vector<ggml_backend_dev_t> devs = { cpu, nullptr };
    mparams.devices = devs.data();

    size_t tmp = seed;
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &tmp, mparams));
    GGML_ASSERT(model);

    const uint32_t n_embd  = llama_model_n_embd(model.get());
    const uint32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model.get()));

    auto make_ctx = [&](uint32_t n_ubatch, uint32_t n_seq_max, uint32_t n_batch) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx           = 128*n_seq_max;
        cp.n_batch         = n_batch;
        cp.n_ubatch        = n_ubatch;
        cp.n_seq_max       = n_seq_max;
        cp.n_threads       = 2;
        cp.n_threads_batch = 2;
        llama_context_ptr c(llama_init_from_model(model.get(), cp));
        GGML_ASSERT(c);
        llama_set_embeddings_layer_inp(c.get(), 0, true);
        return c;
    };

    // reference: one sequence of every token id, so row i is the embedding of token i
    const uint32_t n_ref = 64;
    std::vector<float> ref(n_ref*n_embd);
    {
        llama_context_ptr c = make_ctx(n_ref, 1, n_ref);
        llama_batch b = llama_batch_init(n_ref, 0, 1);
        for (uint32_t i = 0; i < n_ref; i++) {
            common_batch_add(b, (llama_token) i, (llama_pos) i, { 0 }, true);
        }
        GGML_ASSERT(llama_decode(c.get(), b) == 0);
        llama_batch_free(b);
        const float * li = llama_get_embeddings_layer_inp(c.get(), 0);
        std::copy(li, li + (size_t) n_ref*n_embd, ref.begin());
    }
    GGML_ASSERT(n_vocab >= n_ref);

    // every reference row distinct, or a wrong row could match by accident
    for (uint32_t a = 0; a < n_ref; a++) {
        for (uint32_t b2 = a + 1; b2 < n_ref; b2++) {
            bool eq = true;
            for (uint32_t k = 0; k < n_embd; k++) {
                if (ref[(size_t) a*n_embd + k] != ref[(size_t) b2*n_embd + k]) { eq = false; break; }
            }
            GGML_ASSERT(!eq && "two reference rows are identical");
        }
    }

    std::mt19937 gen(seed);
    int n_cases = 0, n_fail = 0, n_skip = 0;

    for (int trial = 0; trial < 400; trial++) {
        const uint32_t n_seq    = 1 + gen() % 4;
        const uint32_t per_seq  = 1 + gen() % 5;
        const uint32_t n_tokens = n_seq*per_seq;
        const uint32_t ub_pick[] = { 1, 2, 3, 4, 8, 16, 32 };
        const uint32_t n_ubatch = ub_pick[gen() % 7];

        std::vector<llama_token>  tok(n_tokens);
        std::vector<llama_pos>    pos(n_tokens);
        std::vector<llama_seq_id> seq(n_tokens);
        std::vector<bool>         out(n_tokens);

        // round-robin the sequences at random, keeping each sequence's positions ascending
        std::vector<uint32_t> next_pos(n_seq, 0), pool;
        for (uint32_t s = 0; s < n_seq; s++) {
            for (uint32_t k = 0; k < per_seq; k++) { pool.push_back(s); }
        }
        std::shuffle(pool.begin(), pool.end(), gen);

        for (uint32_t i = 0; i < n_tokens; i++) {
            const uint32_t s = pool[i];
            seq[i] = (llama_seq_id) s;
            pos[i] = (llama_pos) next_pos[s]++;
            tok[i] = (llama_token) i;      // one distinct id per slot, so a row names its slot
            out[i] = (gen() % 3) != 0;
        }

        llama_context_ptr ctx = make_ctx(n_ubatch, n_seq, std::max(n_tokens, n_ubatch));
        llama_batch b = llama_batch_init(n_tokens, 0, 1);
        for (uint32_t i = 0; i < n_tokens; i++) {
            common_batch_add(b, tok[i], pos[i], { seq[i] }, out[i]);
        }
        const int rc = llama_decode(ctx.get(), b);
        llama_batch_free(b);
        if (rc != 0) { n_skip++; continue; }

        n_cases++;
        const float * li = llama_get_embeddings_layer_inp(ctx.get(), 0);

        bool wrong = false;
        for (uint32_t i = 0; i < n_tokens; i++) {
            bool eq = true;
            for (uint32_t k = 0; k < n_embd; k++) {
                if (li[(size_t) i*n_embd + k] != ref[(size_t) tok[i]*n_embd + k]) { eq = false; break; }
            }
            if (!eq && !wrong) {
                printf("PROBE FAIL n_seq=%u per_seq=%u n_ubatch=%u  row %u is not batch order\n",
                       n_seq, per_seq, n_ubatch, i);
                n_fail++;
                wrong = true;
            }
        }
    }

    printf("PROBE SUMMARY cases=%d failures=%d skipped=%d\n", n_cases, n_fail, n_skip);
    return n_fail == 0 ? 0 : 1;
}
