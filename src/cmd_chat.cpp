// `ggml-loader chat <model.gguf> [--system "text"]`: interactive multi-turn chat.
// Uses the phi3/Phi-4 instruct chat format directly (hardcoded, not a Jinja engine):
//   <|system|>{text}<|end|>  <|user|>{text}<|end|>  <|assistant|>{text}<|end|>  ...
// which is exactly what this model's tokenizer.ggml.chat_template renders for plain
// (non-tool-calling) messages. Multi-turn history lives entirely in the persistent KV cache --
// each turn only encodes/prefills the newest user message, not the whole conversation.
#include "commands.h"
#include "model.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

static void print_usage(const char * argv0) {
    fprintf(stderr, "usage: %s [model.gguf] [--system \"text\"]\n", argv0);
}

static int32_t special_token(const vocab & vc, const char * tag) {
    auto it = vc.token_to_id.find(tag);
    if (it == vc.token_to_id.end()) {
        fprintf(stderr, "error: model vocab has no special token '%s' -- chat template unsupported\n", tag);
        exit(1);
    }
    return it->second;
}

int cmd_chat(int argc, char ** argv) {
    std::string model_path = "Phi-4-mini-instruct-Q4_K_M.gguf";
    bool model_path_set = false;
    std::string system_prompt;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--system" && i + 1 < argc) {
            system_prompt = argv[++i];
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

    const int32_t tok_system    = special_token(vc, "<|system|>");
    const int32_t tok_user      = special_token(vc, "<|user|>");
    const int32_t tok_assistant = special_token(vc, "<|assistant|>");
    const int32_t tok_end       = special_token(vc, "<|end|>");

    ggml_backend_t backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 8);
    kv_cache kv = init_kv_cache(hp, backend);
    decode_session ds = init_decode_session();

    fprintf(stderr, "%s loaded (%lld layers). Type a message and press Enter "
                     "(Ctrl-D or 'exit' to quit).\n",
            model_path.c_str(), (long long) hp.n_layer);

    std::vector<int32_t> pending = { (int32_t) hp.bos_id };
    if (!system_prompt.empty()) {
        pending.push_back(tok_system);
        std::vector<int32_t> enc = vc.encode(system_prompt);
        pending.insert(pending.end(), enc.begin(), enc.end());
        pending.push_back(tok_end);
    }

    int64_t n_past = 0;
    const int max_reply_tokens = 512;

    while (true) {
        printf("> ");
        fflush(stdout);
        std::string line;
        if (!std::getline(std::cin, line)) break; // EOF (Ctrl-D)
        if (line == "exit" || line == "quit") break;
        if (line.empty()) continue;

        pending.push_back(tok_user);
        std::vector<int32_t> enc = vc.encode(line);
        pending.insert(pending.end(), enc.begin(), enc.end());
        pending.push_back(tok_end);
        pending.push_back(tok_assistant);

        if (n_past + (int64_t) pending.size() + max_reply_tokens > N_CTX_MAX) {
            fprintf(stderr, "\n[context window full (%lld tokens) -- restart to continue]\n",
                    (long long) N_CTX_MAX);
            break;
        }

        auto t0 = std::chrono::steady_clock::now();
        std::vector<float> logits = forward_step(m, hp, kv, ds, backend, n_past, pending, n_vocab);
        n_past += (int64_t) pending.size();
        auto t1 = std::chrono::steady_clock::now();
        double prefill_s = std::chrono::duration<double>(t1 - t0).count();
        double prefill_tps = pending.size() / prefill_s;
        size_t n_prefill = pending.size();
        pending.clear(); // this turn's tokens are now committed to the KV cache via n_past

        int32_t next = argmax(logits);
        int n_reply = 0;
        auto t2 = std::chrono::steady_clock::now();
        while (n_reply < max_reply_tokens && next != tok_end && next != hp.eos_id) {
            std::string piece = vc.detok({ next });
            fputs(piece.c_str(), stdout);
            fflush(stdout);

            std::vector<float> step_logits = forward_step(m, hp, kv, ds, backend, n_past, { next }, n_vocab);
            n_past += 1;
            n_reply++;
            next = argmax(step_logits);
        }
        // commit the terminator itself to the cache so history matches the chat template
        // exactly ("...<|assistant|>{reply}<|end|>") for the next turn's context
        forward_step(m, hp, kv, ds, backend, n_past, { next }, n_vocab);
        n_past += 1;
        auto t3 = std::chrono::steady_clock::now();
        double decode_s = std::chrono::duration<double>(t3 - t2).count();

        fprintf(stderr, "\n[prefill %zu tok in %.2fs (%.1f tok/s), reply %d tok in %.2fs (%.1f tok/s)]\n",
                n_prefill, prefill_s, prefill_tps, n_reply, decode_s, n_reply / decode_s);
    }

    free_decode_session(ds);
    ggml_backend_buffer_free(kv.buffer);
    ggml_free(kv.ctx);
    ggml_backend_free(backend);
    gguf_free(gctx);
    return 0;
}
