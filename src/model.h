// Shared model/tokenizer/inference machinery used by all ggml-loader subcommands (inspect, run,
// chat). See model.cpp for the implementation and the design notes on scope.
//
// Multi-architecture support: hparams/layer_weights carry a handful of per-architecture flags
// (qkv_fused, has_qkv_bias, has_qk_norm, ffn_fused, has_rope_scaling) set once at load time from
// `general.architecture`; build_graph()/load_model() branch on those flags at a few well-defined
// points (QKV projection, FFN, RoPE params) rather than having one code path per architecture.
// Whether the output projection is tied to token_embd is NOT one of these flags -- it's decided
// by load_model() from whether an `output.weight` tensor actually exists in the file, since that
// varies by model *size* within an architecture (e.g. small Qwen models tie, larger ones don't),
// not by architecture alone.
// Currently supported: phi3 (fused QKV, fused gate+up FFN, LongRoPE partial rotary), qwen2
// (separate QKV with bias, separate gate/up FFN, plain full RoPE), qwen3 (like qwen2 but no QKV
// bias, adds per-head QK-Norm before RoPE, and attention width can differ from n_embd/n_head --
// see n_embd_head), llama (like qwen3's flags minus QK-Norm: separate QKV, no bias, separate
// gate/up FFN, plain full RoPE). Each of the 4 has its own chat template (see append_chat_turn's
// switch); llama's is "<|start_header_id|>role<|end_header_id|>\n\ntext<|eot_id|>", distinct from
// both phi3's and ChatML. Adding another architecture means extending load_hparams/load_model
// and, if its chat template differs from all existing ones, append_chat_turn()/friends.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <array>
#include <cstdint>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// KV cache size: n_ctx_max positions, F16, across all layers. 4096 keeps the cache a bounded,
// known size (~a few hundred MiB to ~1 GiB depending on the model's head/layer count) and is
// within the range phi3's "short" LongRoPE factors are valid for (the only range this loader
// implements).
constexpr int64_t N_CTX_MAX = 4096;

enum class arch_t {
    PHI3,
    QWEN2,
    QWEN3,
    LLAMA,
};

struct hparams {
    arch_t      arch;
    std::string arch_name; // gguf key prefix, e.g. "phi3", "qwen2"

    int64_t n_embd;
    int64_t n_head;
    int64_t n_head_kv;
    int64_t n_embd_head;    // per-head Q/K/V width. Usually n_embd/n_head, but NOT always: qwen3
                             // decouples attention width from n_embd (attention.key_length is
                             // explicit and can differ), so this is read from that key when
                             // present rather than always derived.
    int64_t n_layer;
    int64_t n_ff;
    int64_t n_rot;          // rope dimensions to rotate (== n_embd_head for full rotary)
    bool    rope_neox;      // true: NeoX-style (split-half) rotation, used by phi3/qwen2/qwen3.
                             // false: "normal" (interleaved-pairs) rotation, used by llama --
                             // this is a real per-architecture distinction (confirmed against
                             // llama.cpp's llama_model_rope_type()), not a universal default;
                             // getting it wrong doesn't crash, it just silently scrambles
                             // position information (observed: fluent words, broken grammar).
    float   rms_eps;
    float   rope_freq_base;

    bool    has_rope_scaling; // phi3 LongRoPE; if false, plain RoPE (ext_factor=0)
    int64_t n_ctx_orig;       // only meaningful if has_rope_scaling
    float   rope_attn_factor; // only meaningful if has_rope_scaling

    bool qkv_fused;    // phi3: one attn_qkv weight. qwen2/qwen3: separate attn_q/k/v.
    bool has_qkv_bias; // qwen2: attn_q/k/v.bias present. phi3/qwen3: no bias.
    bool has_qk_norm;  // qwen3: attn_q_norm/attn_k_norm (RMSNorm per-head, before RoPE).
    bool ffn_fused;    // phi3: one ffn_up weight holding [gate;up]. qwen2/qwen3: separate ffn_gate/up.

    int64_t bos_id;
    int64_t eos_id;
};

hparams load_hparams(const gguf_context * ctx);

struct bpe_ranks {
    std::unordered_map<std::string, int> rank; // "left\x01right" -> merge priority (lower = earlier)
};

// GPT-2-style byte-level BPE tokenizer (encode) + detokenizer (decode). The pretokenizer
// implements the classic GPT-2 splitting rules (contractions/letter-run/digit-run/other-run/
// whitespace-run), not the more elaborate model-specific pretokenizers named by tokenizer.ggml.pre
// ("gpt-4o", "qwen2", ...). Byte-level BPE is lossless regardless of where the pretokenizer draws
// chunk boundaries, so this still encodes to valid vocab ids and round-trips correctly -- it just
// may not match the official tokenizer's exact token boundaries/count in edge cases.
struct vocab {
    std::vector<std::string> tokens;
    std::unordered_map<std::string, int32_t> token_to_id;
    std::array<std::string, 256> byte_encoder;
    std::unordered_map<uint32_t, uint8_t> byte_decoder;
    bpe_ranks bpe;

    std::string detok(const std::vector<int32_t> & ids) const;
    std::vector<int32_t> encode(const std::string & text) const;

    // exact-match lookup for a control/special token (e.g. "<|im_start|>"); exits with an error
    // if the model's vocab doesn't have it, rather than mis-tokenizing it through BPE
    int32_t special(const char * tag) const;
};

vocab load_vocab(const gguf_context * ctx);

// Appends one chat-template turn for `role` ("system"/"user"/"assistant") to `out`, in this
// model's own instruct format (phi3: "<|role|>{text}<|end|>"; qwen2/ChatML:
// "<|im_start|>role\n{text}<|im_end|>\n").
void append_chat_turn(const hparams & hp, const vocab & vc, std::vector<int32_t> & out,
                       const std::string & role, const std::string & text);

// Appends the tokens that prompt the model to start its own turn (phi3: "<|assistant|>";
// qwen2: "<|im_start|>assistant\n").
void append_generation_prompt(const hparams & hp, const vocab & vc, std::vector<int32_t> & out);

// The token id that ends a turn in this model's chat template (phi3: "<|end|>"; qwen2: "<|im_end|>").
int32_t turn_end_token(const hparams & hp, const vocab & vc);

struct layer_weights {
    struct ggml_tensor * attn_norm;

    // fused (phi3) vs separate (qwen2) QKV -- only the relevant fields are set, per hparams::qkv_fused
    struct ggml_tensor * attn_qkv     = nullptr;
    struct ggml_tensor * attn_q       = nullptr;
    struct ggml_tensor * attn_k       = nullptr;
    struct ggml_tensor * attn_v       = nullptr;
    struct ggml_tensor * attn_q_bias  = nullptr;
    struct ggml_tensor * attn_k_bias  = nullptr;
    struct ggml_tensor * attn_v_bias  = nullptr;

    // per-head RMSNorm applied to Q/K before RoPE -- per hparams::has_qk_norm (qwen3)
    struct ggml_tensor * attn_q_norm = nullptr;
    struct ggml_tensor * attn_k_norm = nullptr;

    struct ggml_tensor * attn_output;
    struct ggml_tensor * ffn_norm;

    // fused (phi3) vs separate (qwen2) FFN gate/up -- per hparams::ffn_fused
    struct ggml_tensor * ffn_up_gate = nullptr; // packed [gate;up], phi3
    struct ggml_tensor * ffn_gate    = nullptr; // qwen2
    struct ggml_tensor * ffn_up      = nullptr; // qwen2

    struct ggml_tensor * ffn_down;
};

struct model {
    struct ggml_tensor * token_embd;
    struct ggml_tensor * output;             // == token_embd if no separate output.weight tensor exists
    struct ggml_tensor * output_norm;
    struct ggml_tensor * rope_factors = nullptr; // phi3 LongRoPE only; nullptr => plain RoPE
    std::vector<layer_weights> layers;
};

model load_model(struct ggml_context * wctx, const hparams & hp);

// Loads every tensor from a GGUF file into memory owned by `backend` -- works for any backend,
// including GPU ones like Vulkan, unlike gguf_init_from_file(no_alloc=false) which only ever
// gives plain host (malloc) memory. Reads each tensor's raw bytes directly from the file at its
// recorded offset and uploads via ggml_backend_tensor_set(), rather than relying on gguf.c's own
// (CPU-only) bulk loading path.
struct weights_store {
    struct ggml_context * ctx = nullptr;    // tensor metadata + backend-owned data; pass to load_model()
    ggml_backend_buffer_t buffer = nullptr; // owns the tensor data; free with ggml_backend_buffer_free()
};
weights_store load_weights(const char * path, ggml_backend_t backend, struct gguf_context ** out_gguf);

// Same as load_weights(), but only allocates/reads tensors whose name starts with one of
// `prefixes` -- everything else in the file is skipped entirely (never allocated in the backend
// buffer, never read off disk). Lets a process that only needs e.g. the SDXL VAE decoder avoid
// paying for CLIP+UNet's memory too, so a staged encoder/UNet/decoder pipeline run as separate
// process invocations actually lowers peak memory instead of each stage loading the whole
// checkpoint anyway. An empty `prefixes` matches everything (same behavior as load_weights()).
weights_store load_weights_filtered(const char * path, ggml_backend_t backend,
                                     const std::vector<std::string> & prefixes,
                                     struct gguf_context ** out_gguf);

// Selects a backend by name ("cpu" or "vulkan"). Exits with an error on an unknown name or if
// vulkan is requested but this build/device doesn't have it. n_threads only affects "cpu".
ggml_backend_t init_backend(const std::string & name, int n_threads);

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
    bool use_flash_attn = false; // ggml_flash_attn_ext instead of the manual mul_mat/softmax/
                                  // mul_mat sequence -- opt-in (default off, matching llama.cpp's
                                  // own -fa flag) so it can be A/B benchmarked before trusting it
                                  // not to regress this project's existing lead over llama.cpp.
};

decode_session init_decode_session(ggml_backend_t backend, bool use_flash_attn = false);
void free_decode_session(decode_session & ds);

// Runs one forward pass for `new_tokens`, appending them to the KV cache at [n_past, n_past +
// new_tokens.size()) and attending over the full [0, n_past + new_tokens.size()) cached range.
// Returns the logits (size n_vocab) for the *last* new token only.
std::vector<float> forward_step(const model & m, const hparams & hp, kv_cache & kv,
                                 decode_session & ds, ggml_backend_t backend, int64_t n_past,
                                 const std::vector<int32_t> & new_tokens, int64_t n_vocab);

int32_t argmax(const std::vector<float> & logits);

// Sampling: temp<=0 is greedy (== argmax, deterministic) -- this is the default everywhere
// (run/chat/bench) so nothing changes unless a caller explicitly opts in via CLI flags. Above
// that: temperature scaling, then top-k, then top-p (nucleus) narrow the candidate set before a
// weighted random draw. repeat_penalty (1.0 = disabled) downweights tokens seen in the last
// repeat_last_n tokens of `recent_tokens` -- applied even at temp<=0, since it doesn't need
// randomness to stop greedy decoding's "Paris is the capital of France. Paris is..." loops.
struct sampler_params {
    float    temp            = 0.0f;
    int      top_k           = 40;
    float    top_p           = 0.95f;
    float    repeat_penalty  = 1.0f;
    int      repeat_last_n   = 64;
    uint32_t seed            = 0; // 0 = seed from std::random_device
};

struct sampler {
    sampler_params params;
    std::mt19937 rng;
};

sampler init_sampler(const sampler_params & params);

// mutates `logits` in place (repetition penalty); returns the sampled token id
int32_t sample(sampler & smp, std::vector<float> & logits, const std::vector<int32_t> & recent_tokens);
