// `ggml-loader run <model.gguf> [-p/--prompt "text"] [-n/--n-predict N] [--backend cpu|vulkan]`:
// one-shot raw completion (no chat template), with prefill/decode timing -- the mode used to
// benchmark against llama-cli/llama-bench on the same prompt, and cpu vs vulkan against each other.
#include "commands.h"
#include "model.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s [model.gguf] [-p/--prompt \"text\"] [-n/--n-predict N] [--backend cpu|vulkan] [--flash-attn]\n"
        "           [--temp F] [--top-k N] [--top-p F] [--repeat-penalty F] [--repeat-last-n N] [--seed N]\n"
        "  no -p: BOS-seeded unconditional free-run\n"
        "  temp<=0 (default): greedy/deterministic, same as before these flags existed\n", argv0);
}

int cmd_run(int argc, char ** argv) {
    std::string model_path = "Phi-4-mini-instruct-Q4_K_M.gguf";
    bool model_path_set = false;
    std::string prompt;
    std::string backend_name = "cpu";
    int max_new_tokens = 32;
    bool flash_attn = false;
    sampler_params sp;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if ((a == "-p" || a == "--prompt") && i + 1 < argc) {
            prompt = argv[++i];
        } else if ((a == "-n" || a == "--n-predict") && i + 1 < argc) {
            max_new_tokens = atoi(argv[++i]);
        } else if (a == "--backend" && i + 1 < argc) {
            backend_name = argv[++i];
        } else if (a == "--flash-attn") {
            flash_attn = true;
        } else if (a == "--temp" && i + 1 < argc) {
            sp.temp = (float) atof(argv[++i]);
        } else if (a == "--top-k" && i + 1 < argc) {
            sp.top_k = atoi(argv[++i]);
        } else if (a == "--top-p" && i + 1 < argc) {
            sp.top_p = (float) atof(argv[++i]);
        } else if (a == "--repeat-penalty" && i + 1 < argc) {
            sp.repeat_penalty = (float) atof(argv[++i]);
        } else if (a == "--repeat-last-n" && i + 1 < argc) {
            sp.repeat_last_n = atoi(argv[++i]);
        } else if (a == "--seed" && i + 1 < argc) {
            sp.seed = (uint32_t) strtoul(argv[++i], nullptr, 10);
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (!model_path_set) {
            model_path = a;
            model_path_set = true;
        }
    }
    sampler smp = init_sampler(sp);

    ggml_backend_t backend = init_backend(backend_name, 8);

    struct gguf_context * gctx = nullptr;
    weights_store ws = load_weights(model_path.c_str(), backend, &gctx);

    hparams hp = load_hparams(gctx);
    vocab   vc = load_vocab(gctx);
    model   m  = load_model(ws.ctx, hp);
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

    fprintf(stderr, "backend=%s n_embd=%lld n_head=%lld n_head_kv=%lld n_layer=%lld n_ff=%lld n_rot=%lld n_vocab=%lld\n",
            backend_name.c_str(), (long long) hp.n_embd, (long long) hp.n_head, (long long) hp.n_head_kv,
            (long long) hp.n_layer, (long long) hp.n_ff, (long long) hp.n_rot, (long long) n_vocab);

    kv_cache kv = init_kv_cache(hp, backend);
    ssm_state ss = init_ssm_state(hp, backend); // no-op struct for every arch except nemotron_h
    decode_session ds = init_decode_session(backend, flash_attn);
    const int64_t head_dim = hp.n_embd_head;
    const double kv_cache_mib =
        (double)(hp.n_layer * 2 * N_CTX_MAX * hp.n_head_kv * head_dim * 2) / (1024.0 * 1024.0);
    fprintf(stderr, "KV cache: %lld ctx x %lld layers, f16 -> %.0f MiB\n",
            (long long) N_CTX_MAX, (long long) hp.n_layer, kv_cache_mib);

    // prefill: process the whole prompt (BOS + encoded text) in one batch
    auto t0 = std::chrono::steady_clock::now();
    std::vector<float> logits = forward_step(m, hp, kv, ds, backend, 0, prompt_tokens, n_vocab, &ss);
    auto t1 = std::chrono::steady_clock::now();
    double prefill_s = std::chrono::duration<double>(t1 - t0).count();

    std::vector<int32_t> all_tokens = prompt_tokens;
    int64_t n_past = (int64_t) prompt_tokens.size();
    int32_t next = sample(smp, logits, all_tokens);

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

        std::vector<float> step_logits = forward_step(m, hp, kv, ds, backend, n_past, { next }, n_vocab, &ss);
        n_past += 1;
        next = sample(smp, step_logits, all_tokens);
    }
    auto t3 = std::chrono::steady_clock::now();
    double decode_s = std::chrono::duration<double>(t3 - t2).count();
    fputc('\n', stdout);

    fprintf(stderr, "decode: %d tokens in %.3fs (%.2f tok/s)\n", n_decoded, decode_s,
            n_decoded / decode_s);
    fprintf(stderr, "\nfull output:\n%s\n", vc.detok(all_tokens).c_str());

    free_decode_session(ds);
    free_ssm_state(ss);
    ggml_backend_buffer_free(kv.buffer);
    ggml_free(kv.ctx);
    ggml_backend_buffer_free(ws.buffer);
    ggml_free(ws.ctx);
    ggml_backend_free(backend);
    gguf_free(gctx);
    return 0;
}
