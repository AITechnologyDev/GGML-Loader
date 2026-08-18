// `ggml-loader run <model.gguf> [-p/--prompt "text"] [-n/--n-predict N]`: one-shot raw
// completion (no chat template), with prefill/decode timing -- the mode used to benchmark
// against llama-cli/llama-bench on the same prompt.
#include "commands.h"
#include "model.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s [model.gguf] [-p/--prompt \"text\"] [-n/--n-predict N]\n"
        "  no -p: BOS-seeded unconditional free-run\n", argv0);
}

int cmd_run(int argc, char ** argv) {
    std::string model_path = "Phi-4-mini-instruct-Q4_K_M.gguf";
    bool model_path_set = false;
    std::string prompt;
    int max_new_tokens = 32;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if ((a == "-p" || a == "--prompt") && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((a == "-n" || a == "--n-predict") && i + 1 < argc) {
            max_new_tokens = atoi(argv[++i]);
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (!model_path_set) {
            model_path = a;
            model_path_set = true;
        }
    }

    struct ggml_context * wctx = nullptr;
    struct gguf_init_params gp = { /*.no_alloc =*/ false, /*.ctx =*/ &wctx };
    struct gguf_context * gctx = gguf_init_from_file(model_path.c_str(), gp);
    if (!gctx || !wctx) {
        fprintf(stderr, "error: failed to load '%s'\n", model_path.c_str());
        return 1;
    }

    hparams hp = load_hparams(gctx);
    vocab   vc = load_vocab(gctx);
    model   m  = load_model(wctx, hp);
    const int64_t n_vocab = m.token_embd->ne[1];
    // some models occasionally emit the literal "<|endoftext|>" token distinct from hp.eos_id;
    // treat it as a stop signal too if the vocab has it (see cmd_chat.cpp for where this was found)
    auto eot_it = vc.token_to_id.find("<|endoftext|>");
    const int32_t tok_eot = eot_it != vc.token_to_id.end() ? eot_it->second : -1;

    std::vector<int32_t> prompt_tokens = { (int32_t) hp.bos_id };
    if (!prompt.empty()) {
        std::vector<int32_t> enc = vc.encode(prompt);
        prompt_tokens.insert(prompt_tokens.end(), enc.begin(), enc.end());
        fprintf(stderr, "prompt: \"%s\" -> %zu tokens (round-trip: \"%s\")\n",
                prompt.c_str(), enc.size(), vc.detok(enc).c_str());
    }

    if ((int64_t) prompt_tokens.size() + max_new_tokens > N_CTX_MAX) {
        fprintf(stderr, "error: prompt (%zu) + max_new_tokens (%d) exceeds the %lld-token KV cache\n",
                prompt_tokens.size(), max_new_tokens, (long long) N_CTX_MAX);
        return 1;
    }

    fprintf(stderr, "n_embd=%lld n_head=%lld n_head_kv=%lld n_layer=%lld n_ff=%lld n_rot=%lld n_vocab=%lld\n",
            (long long) hp.n_embd, (long long) hp.n_head, (long long) hp.n_head_kv,
            (long long) hp.n_layer, (long long) hp.n_ff, (long long) hp.n_rot, (long long) n_vocab);

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 8);

    kv_cache kv = init_kv_cache(hp, backend);
    decode_session ds = init_decode_session();
    const int64_t head_dim = hp.n_embd_head;
    const double kv_cache_mib =
        (double)(hp.n_layer * 2 * N_CTX_MAX * hp.n_head_kv * head_dim * 2) / (1024.0 * 1024.0);
    fprintf(stderr, "KV cache: %lld ctx x %lld layers, f16 -> %.0f MiB\n",
            (long long) N_CTX_MAX, (long long) hp.n_layer, kv_cache_mib);

    // prefill: process the whole prompt (BOS + encoded text) in one batch
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> logits = forward_step(m, hp, kv, ds, backend, 0, prompt_tokens, n_vocab);
    auto t1 = std::chrono::steady_clock::now();
    double prefill_s = std::chrono::duration<double>(t1 - t0).count();

    std::vector<int32_t> all_tokens = prompt_tokens;
    int64_t n_past = (int64_t) prompt_tokens.size();
    int32_t next = argmax(logits);

    fprintf(stderr, "prefill: %zu tokens in %.3fs (%.2f tok/s)\n",
            prompt_tokens.size(), prefill_s, prompt_tokens.size() / prefill_s);
    fprintf(stderr, "generating (KV-cached, up to %d tokens)...\n", max_new_tokens);

    // decode: one new token at a time, timed separately from prefill for a pp/tg-style comparison
    int n_decoded = 0;
    auto t2 = std::chrono::steady_clock::now();
    for (int step = 0; step < max_new_tokens; step++) {
        if (next == hp.eos_id || next == tok_eot) {
            fprintf(stderr, "[eos]\n");
            break;
        }
        all_tokens.push_back(next);
        n_decoded++;

        std::string piece = vc.detok({ next });
        fputs(piece.c_str(), stdout);
        fflush(stdout);

        std::vector<float> step_logits = forward_step(m, hp, kv, ds, backend, n_past, { next }, n_vocab);
        n_past += 1;
        next = argmax(step_logits);
    }
    auto t3 = std::chrono::steady_clock::now();
    double decode_s = std::chrono::duration<double>(t3 - t2).count();
    fputc('\n', stdout);

    fprintf(stderr, "decode: %d tokens in %.3fs (%.2f tok/s)\n", n_decoded, decode_s,
            n_decoded / decode_s);
    fprintf(stderr, "\nfull output:\n%s\n", vc.detok(all_tokens).c_str());

    free_decode_session(ds);
    ggml_backend_buffer_free(kv.buffer);
    ggml_free(kv.ctx);
    ggml_backend_free(backend);
    gguf_free(gctx);
    return 0;
}
