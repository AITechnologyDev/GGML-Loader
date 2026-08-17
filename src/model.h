// Shared phi3-architecture model/tokenizer/inference machinery used by all ggml-loader
// subcommands (inspect, run, chat). See model.cpp for the implementation and the design notes
// on scope (no persistent-context prefill beyond N_CTX_MAX, simplified GPT-2-style
// pretokenizer, etc).
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// KV cache size: n_ctx_max positions, F16, across all layers. 4096 matches phi3's
// original_context_length (the range the "short" rope factors are valid for, which is all this
// loader implements so far). At that size the cache is ~512 MiB for Phi-4-mini's shape.
constexpr int64_t N_CTX_MAX = 4096;

struct hparams {
    int64_t n_embd;
    int64_t n_head;
    int64_t n_head_kv;
    int64_t n_layer;
    int64_t n_ff;        // half of ffn_up's output (gate+up are concatenated)
    int64_t n_rot;       // rope.dimension_count (partial rotary)
    int64_t n_ctx_orig;  // rope.scaling.original_context_length
    float   rms_eps;
    float   rope_freq_base;
    float   rope_attn_factor;
    int64_t bos_id;
    int64_t eos_id;
};

hparams load_hparams(const gguf_context * ctx);

struct bpe_ranks {
    std::unordered_map<std::string, int> rank; // "left\x01right" -> merge priority (lower = earlier)
};

// GPT-2-style byte-level BPE tokenizer (encode) + detokenizer (decode). The pretokenizer
// implements the classic GPT-2 splitting rules (contractions/letter-run/digit-run/other-run/
// whitespace-run), not the more elaborate "gpt-4o" pretokenizer this model's tokenizer.ggml.pre
// actually names. Byte-level BPE is lossless regardless of where the pretokenizer draws chunk
// boundaries, so this still encodes to valid vocab ids and round-trips correctly -- it just may
// not match the official tokenizer's exact token boundaries/count in edge cases.
struct vocab {
    std::vector<std::string> tokens;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::array<std::string, 256> byte_encoder;
    std::unordered_map<uint32_t, uint8_t> byte_decoder;
    bpe_ranks bpe;

    std::string detok(const std::vector<int32_t> & ids) const;
    std::vector<int32_t> encode(const std::string & text) const;
};

vocab load_vocab(const gguf_context * ctx);

struct layer_weights {
    struct ggml_tensor * attn_norm;
    struct ggml_tensor * attn_qkv;
    struct ggml_tensor * attn_output;
    struct ggml_tensor * ffn_norm;
    struct ggml_tensor * ffn_up;
    struct ggml_tensor * ffn_down;
};

struct model {
    struct ggml_tensor * token_embd;
    struct ggml_tensor * output_norm;
    struct ggml_tensor * rope_factors_short;
    std::vector<layer_weights> layers;
};

model load_model(struct ggml_context * wctx, const hparams & hp);

// persistent KV cache: one F16 [head_dim, n_head_kv, N_CTX_MAX] tensor per layer, per K/V.
struct kv_cache {
    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    std::vector<struct ggml_tensor *> k;
    std::vector<struct ggml_tensor *> v;
};

kv_cache init_kv_cache(const hparams & hp, ggml_backend_t backend);

// Reusable scratch state for forward_step: a persistent graph-building buffer and a persistent
// graph allocator, so a generation loop doesn't malloc/free them (and re-plan the activation
// buffer layout) on every single token -- ggml_gallocr_alloc_graph is designed to be called
// repeatedly with differently-shaped graphs (our KV range grows every decode step) and replans
// as needed, so one decode_session can be reused for an entire run/chat session.
struct decode_session {
    std::vector<uint8_t> graph_buf;
    ggml_gallocr_t galloc = nullptr;
};

decode_session init_decode_session();
void free_decode_session(decode_session & ds);

// Runs one forward pass for `new_tokens`, appending them to the KV cache at [n_past, n_past +
// new_tokens.size()) and attending over the full [0, n_past + new_tokens.size()) cached range.
// Returns the logits (size n_vocab) for the *last* new token only.
std::vector<float> forward_step(const model & m, const hparams & hp, kv_cache & kv,
                                 decode_session & ds, ggml_backend_t backend, int64_t n_past,
                                 const std::vector<int32_t> & new_tokens, int64_t n_vocab);

int32_t argmax(const std::vector<float> & logits);
