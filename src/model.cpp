#include "model.h"

#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "ggml-vulkan.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static int64_t kv_find(const gguf_context * ctx, const char * key) {
    int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        fprintf(stderr, "error: missing gguf key '%s'\n", key);
        exit(1);
    }
    return id;
}

static long long kv_int(const gguf_context * ctx, const char * key) {
    int64_t id = kv_find(ctx, key);
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8:  return gguf_get_val_u8(ctx, id);
        case GGUF_TYPE_INT8:   return gguf_get_val_i8(ctx, id);
        case GGUF_TYPE_UINT16: return gguf_get_val_u16(ctx, id);
        case GGUF_TYPE_INT16:  return gguf_get_val_i16(ctx, id);
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT64: return (long long) gguf_get_val_u64(ctx, id);
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(ctx, id);
        default:
            fprintf(stderr, "error: key '%s' is not an integer type\n", key);
            exit(1);
    }
}

static double kv_float(const gguf_context * ctx, const char * key) {
    int64_t id = kv_find(ctx, key);
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(ctx, id);
        case GGUF_TYPE_FLOAT64: return gguf_get_val_f64(ctx, id);
        default:
            fprintf(stderr, "error: key '%s' is not a float type\n", key);
            exit(1);
    }
}

static std::string kv_string(const gguf_context * ctx, const char * key) {
    int64_t id = kv_find(ctx, key);
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        fprintf(stderr, "error: key '%s' is not a string type\n", key);
        exit(1);
    }
    return gguf_get_val_str(ctx, id);
}

hparams load_hparams(const gguf_context * ctx) {
    hparams h{};
    h.arch_name = kv_string(ctx, "general.architecture");
    if      (h.arch_name == "phi3")  h.arch = arch_t::PHI3;
    else if (h.arch_name == "qwen2") h.arch = arch_t::QWEN2;
    else if (h.arch_name == "qwen3") h.arch = arch_t::QWEN3;
    else if (h.arch_name == "llama") h.arch = arch_t::LLAMA;
    else {
        fprintf(stderr, "error: unsupported architecture '%s' (supported: phi3, qwen2, qwen3, llama)\n",
                h.arch_name.c_str());
        exit(1);
    }

    auto pfx = [&](const char * suffix) { return h.arch_name + "." + suffix; };

    h.n_embd    = kv_int(ctx, pfx("embedding_length").c_str());
    h.n_head    = kv_int(ctx, pfx("attention.head_count").c_str());
    h.n_head_kv = kv_int(ctx, pfx("attention.head_count_kv").c_str());
    h.n_layer   = kv_int(ctx, pfx("block_count").c_str());
    h.n_ff      = kv_int(ctx, pfx("feed_forward_length").c_str());
    h.rms_eps   = (float) kv_float(ctx, pfx("attention.layer_norm_rms_epsilon").c_str());
    h.rope_freq_base = (float) kv_float(ctx, pfx("rope.freq_base").c_str());

    // attention width isn't always n_embd/n_head (qwen3 decouples them); prefer the explicit
    // key_length key when the model provides one
    std::string key_len_key = pfx("attention.key_length");
    h.n_embd_head = gguf_find_key(ctx, key_len_key.c_str()) >= 0
                        ? kv_int(ctx, key_len_key.c_str())
                        : h.n_embd / h.n_head;

    std::string rot_key = pfx("rope.dimension_count");
    h.n_rot = gguf_find_key(ctx, rot_key.c_str()) >= 0 ? kv_int(ctx, rot_key.c_str()) : h.n_embd_head;

    std::string ctx_orig_key = pfx("rope.scaling.original_context_length");
    h.has_rope_scaling = gguf_find_key(ctx, ctx_orig_key.c_str()) >= 0;
    if (h.has_rope_scaling) {
        h.n_ctx_orig       = kv_int(ctx, ctx_orig_key.c_str());
        h.rope_attn_factor = (float) kv_float(ctx, pfx("rope.scaling.attn_factor").c_str());
    } else {
        h.n_ctx_orig       = kv_int(ctx, pfx("context_length").c_str()); // unused when ext_factor=0
        h.rope_attn_factor = 1.0f;
    }

    h.qkv_fused    = (h.arch == arch_t::PHI3);
    h.has_qkv_bias = (h.arch == arch_t::QWEN2);
    h.has_qk_norm  = (h.arch == arch_t::QWEN3);
    h.ffn_fused    = (h.arch == arch_t::PHI3);
    h.rope_neox    = (h.arch != arch_t::LLAMA); // llama uses "normal" interleaved-pairs rotation

    h.bos_id = kv_int(ctx, "tokenizer.ggml.bos_token_id");
    h.eos_id = kv_int(ctx, "tokenizer.ggml.eos_token_id");
    return h;
}

// ---- GPT-2 byte-level BPE: shared byte<->unicode table, then decode (detokenize) and
// encode (pretokenize + BPE merge) built on top of it ----

// bytes 33-126, 161-172, 174-255 map to themselves; every other byte (control chars, space,
// 127-160, 173) maps to a synthetic codepoint starting at 256, in byte order. This is the
// standard GPT-2 bytes_to_unicode() table, needed both directions: decoder for detok, encoder
// for tokenize.
static void build_byte_tables(std::array<std::string, 256> & encoder,
                               std::unordered_map<uint32_t, uint8_t> & decoder) {
    std::vector<int> bs;
    for (int b = 33;  b <= 126; b++) bs.push_back(b);
    for (int b = 161; b <= 172; b++) bs.push_back(b);
    for (int b = 174; b <= 255; b++) bs.push_back(b);

    std::vector<bool> in_bs(256, false);
    for (int b : bs) in_bs[b] = true;

    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (!in_bs[b]) {
            bs.push_back(b);
            cs.push_back(256 + n);
            n++;
        }
    }

    for (size_t i = 0; i < bs.size(); i++) {
        uint32_t cp = (uint32_t) cs[i];
        decoder[cp] = (uint8_t) bs[i];

        std::string enc;
        if (cp < 0x80) {
            enc.push_back((char) cp);
        } else if (cp < 0x800) {
            enc.push_back((char)(0xC0 | (cp >> 6)));
            enc.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
            enc.push_back((char)(0xE0 | (cp >> 12)));
            enc.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            enc.push_back((char)(0x80 | (cp & 0x3F)));
        }
        encoder[(uint8_t) bs[i]] = enc;
    }
}

static std::vector<uint32_t> utf8_decode(const std::string & s) {
    std::vector<uint32_t> out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = (unsigned char) s[i];
        uint32_t cp;
        int len;
        if      ((c & 0x80) == 0x00) { cp = c;        len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
        else                         { cp = c;        len = 1; }
        for (int k = 1; k < len && i + k < s.size(); k++) {
            cp = (cp << 6) | ((unsigned char) s[i + k] & 0x3F);
        }
        out.push_back(cp);
        i += len;
    }
    return out;
}

// GPT-2 style pretokenizer: contractions, then a maximal run of letters/digits/other-symbols
// (each optionally prefixed by one leading space), then whitespace runs.
static bool is_letter(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80; }
static bool is_digit_(unsigned char c) { return c >= '0' && c <= '9'; }
static bool is_space_(unsigned char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

static std::vector<std::string> pretokenize(const std::string & text) {
    static const char * contractions[] = { "'s", "'t", "'re", "'ve", "'m", "'ll", "'d" };
    std::vector<std::string> out;
    size_t i = 0, n = text.size();
    while (i < n) {
        bool matched = false;
        for (const char * c : contractions) {
            size_t len = strlen(c);
            if (i + len <= n && text.compare(i, len, c) == 0) {
                out.push_back(text.substr(i, len));
                i += len;
                matched = true;
                break;
            }
        }
        if (matched) continue;

        size_t start = i;
        if (text[i] == ' ' && i + 1 < n && text[i + 1] != ' ') {
            i++; // a single leading space attaches to the run that follows
        }

        unsigned char c = (unsigned char) text[i];
        size_t j = i;
        if (is_space_(c)) {
            while (j < n && is_space_((unsigned char) text[j])) j++;
        } else if (is_digit_(c)) {
            while (j < n && is_digit_((unsigned char) text[j])) j++;
        } else if (is_letter(c)) {
            while (j < n && is_letter((unsigned char) text[j])) j++;
        } else {
            while (j < n && !is_space_((unsigned char) text[j]) && !is_digit_((unsigned char) text[j])
                          && !is_letter((unsigned char) text[j])) j++;
        }
        out.push_back(text.substr(start, j - start));
        i = j;
    }
    return out;
}

static bpe_ranks load_bpe(const gguf_context * ctx) {
    bpe_ranks b;
    int64_t id = kv_find(ctx, "tokenizer.ggml.merges");
    size_t n = gguf_get_arr_n(ctx, id);
    for (size_t i = 0; i < n; i++) {
        std::string merge = gguf_get_arr_str(ctx, id, i);
        size_t sp = merge.find(' ');
        if (sp == std::string::npos) continue;
        std::string key = merge.substr(0, sp) + "\x01" + merge.substr(sp + 1);
        b.rank[key] = (int) i;
    }
    return b;
}

// merges byte-encoded symbols for one pretoken chunk until no adjacent pair has a known rank
static std::vector<std::string> bpe_merge(const std::vector<std::string> & symbols_in,
                                           const bpe_ranks & bpe) {
    std::vector<std::string> symbols = symbols_in;
    while (symbols.size() > 1) {
        int best_rank = -1;
        size_t best_i = 0;
        for (size_t i = 0; i + 1 < symbols.size(); i++) {
            auto it = bpe.rank.find(symbols[i] + "\x01" + symbols[i + 1]);
            if (it != bpe.rank.end() && (best_rank < 0 || it->second < best_rank)) {
                best_rank = it->second;
                best_i = i;
            }
        }
        if (best_rank < 0) break;
        symbols[best_i] += symbols[best_i + 1];
        symbols.erase(symbols.begin() + best_i + 1);
    }
    return symbols;
}

std::string vocab::detok(const std::vector<int32_t> & ids) const {
    std::string raw;
    for (int32_t id : ids) {
        if (id < 0 || (size_t) id >= tokens.size()) continue;
        for (uint32_t cp : utf8_decode(tokens[id])) {
            auto it = byte_decoder.find(cp);
            if (it != byte_decoder.end()) {
                raw.push_back((char) it->second);
            }
        }
    }
    return raw;
}

std::vector<int32_t> vocab::encode(const std::string & text) const {
    std::vector<int32_t> ids;
    for (const std::string & chunk : pretokenize(text)) {
        std::vector<std::string> symbols;
        symbols.reserve(chunk.size());
        for (unsigned char b : chunk) {
            symbols.push_back(byte_encoder[b]);
        }
        for (const std::string & piece : bpe_merge(symbols, bpe)) {
            auto it = token_to_id.find(piece);
            if (it != token_to_id.end()) {
                ids.push_back(it->second);
            } else {
                // shouldn't happen for a well-formed vocab, but fail soft rather than crash
                fprintf(stderr, "warning: no vocab id for BPE piece '%s'\n", piece.c_str());
            }
        }
    }
    return ids;
}

vocab load_vocab(const gguf_context * ctx) {
    vocab v;
    build_byte_tables(v.byte_encoder, v.byte_decoder);
    v.bpe = load_bpe(ctx);
    int64_t id = kv_find(ctx, "tokenizer.ggml.tokens");
    size_t n = gguf_get_arr_n(ctx, id);
    v.tokens.reserve(n);
    for (size_t i = 0; i < n; i++) {
        v.tokens.emplace_back(gguf_get_arr_str(ctx, id, i));
        v.token_to_id[v.tokens.back()] = (int32_t) i;
    }
    return v;
}

int32_t vocab::special(const char * tag) const {
    auto it = token_to_id.find(tag);
    if (it == token_to_id.end()) {
        fprintf(stderr, "error: model vocab has no special token '%s'\n", tag);
        exit(1);
    }
    return it->second;
}

// ---- chat templates (one hardcoded format per architecture, not a Jinja engine -- see model.h) ----

static void append_encoded(const vocab & vc, std::vector<int32_t> & out, const std::string & text) {
    std::vector<int32_t> enc = vc.encode(text);
    out.insert(out.end(), enc.begin(), enc.end());
}

void append_chat_turn(const hparams & hp, const vocab & vc, std::vector<int32_t> & out,
                       const std::string & role, const std::string & text) {
    switch (hp.arch) {
        case arch_t::PHI3:
            out.push_back(vc.special(("<|" + role + "|>").c_str()));
            append_encoded(vc, out, text);
            out.push_back(vc.special("<|end|>"));
            break;
        case arch_t::LLAMA: // "<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>"
            out.push_back(vc.special("<|start_header_id|>"));
            append_encoded(vc, out, role);
            out.push_back(vc.special("<|end_header_id|>"));
            append_encoded(vc, out, "\n\n");
            append_encoded(vc, out, text);
            out.push_back(vc.special("<|eot_id|>"));
            break;
        case arch_t::QWEN2: // ChatML: "<|im_start|>{role}\n{text}<|im_end|>\n"
        case arch_t::QWEN3:
        default:
            out.push_back(vc.special("<|im_start|>"));
            append_encoded(vc, out, role + "\n");
            append_encoded(vc, out, text);
            out.push_back(vc.special("<|im_end|>"));
            append_encoded(vc, out, "\n");
            break;
    }
}

void append_generation_prompt(const hparams & hp, const vocab & vc, std::vector<int32_t> & out) {
    switch (hp.arch) {
        case arch_t::PHI3:
            out.push_back(vc.special("<|assistant|>"));
            break;
        case arch_t::LLAMA:
            out.push_back(vc.special("<|start_header_id|>"));
            append_encoded(vc, out, "assistant");
            out.push_back(vc.special("<|end_header_id|>"));
            append_encoded(vc, out, "\n\n");
            break;
        case arch_t::QWEN2:
        case arch_t::QWEN3:
        default:
            out.push_back(vc.special("<|im_start|>"));
            append_encoded(vc, out, "assistant\n");
            break;
    }
}

int32_t turn_end_token(const hparams & hp, const vocab & vc) {
    switch (hp.arch) {
        case arch_t::PHI3:  return vc.special("<|end|>");
        case arch_t::LLAMA: return vc.special("<|eot_id|>");
        case arch_t::QWEN2:
        case arch_t::QWEN3:
        default:             return vc.special("<|im_end|>");
    }
}

// ---- model weights ----

static struct ggml_tensor * must_get(struct ggml_context * ctx, const std::string & name) {
    struct ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t) {
        fprintf(stderr, "error: missing tensor '%s'\n", name.c_str());
        exit(1);
    }
    return t;
}

model load_model(struct ggml_context * wctx, const hparams & hp) {
    model m;
    m.token_embd  = must_get(wctx, "token_embd.weight");
    m.output_norm = must_get(wctx, "output_norm.weight");
    // tied vs untied output varies by model *size* within an architecture (small variants often
    // tie, larger ones often don't), so this is decided by tensor presence, not an hparams flag
    struct ggml_tensor * output_w = ggml_get_tensor(wctx, "output.weight");
    m.output = output_w ? output_w : m.token_embd;

    if (hp.has_rope_scaling) {
        m.rope_factors = must_get(wctx, "rope_factors_short.weight"); // phi3 LongRoPE
    }

    m.layers.resize(hp.n_layer);
    for (int64_t il = 0; il < hp.n_layer; il++) {
        std::string p = "blk." + std::to_string(il) + ".";
        layer_weights & l = m.layers[il];
        l.attn_norm   = must_get(wctx, p + "attn_norm.weight");
        l.attn_output = must_get(wctx, p + "attn_output.weight");
        l.ffn_norm    = must_get(wctx, p + "ffn_norm.weight");
        l.ffn_down    = must_get(wctx, p + "ffn_down.weight");

        if (hp.qkv_fused) {
            l.attn_qkv = must_get(wctx, p + "attn_qkv.weight");
        } else {
            l.attn_q = must_get(wctx, p + "attn_q.weight");
            l.attn_k = must_get(wctx, p + "attn_k.weight");
            l.attn_v = must_get(wctx, p + "attn_v.weight");
            if (hp.has_qkv_bias) {
                l.attn_q_bias = must_get(wctx, p + "attn_q.bias");
                l.attn_k_bias = must_get(wctx, p + "attn_k.bias");
                l.attn_v_bias = must_get(wctx, p + "attn_v.bias");
            }
            if (hp.has_qk_norm) {
                l.attn_q_norm = must_get(wctx, p + "attn_q_norm.weight");
                l.attn_k_norm = must_get(wctx, p + "attn_k_norm.weight");
            }
        }

        if (hp.ffn_fused) {
            l.ffn_up_gate = must_get(wctx, p + "ffn_up.weight");
        } else {
            l.ffn_gate = must_get(wctx, p + "ffn_gate.weight");
            l.ffn_up   = must_get(wctx, p + "ffn_up.weight");
        }
    }
    return m;
}

weights_store load_weights(const char * path, ggml_backend_t backend, struct gguf_context ** out_gguf) {
    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ &meta_ctx };
    struct gguf_context * gctx = gguf_init_from_file(path, gp);
    if (!gctx || !meta_ctx) {
        fprintf(stderr, "error: failed to load '%s'\n", path);
        exit(1);
    }

    weights_store ws;
    ws.ctx = meta_ctx;
    ws.buffer = ggml_backend_alloc_ctx_tensors(meta_ctx, backend);
    if (!ws.buffer) {
        fprintf(stderr, "error: failed to allocate backend memory for weights from '%s'\n", path);
        exit(1);
    }

    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "error: failed to reopen '%s' to read tensor data\n", path);
        exit(1);
    }
    size_t data_base = gguf_get_data_offset(gctx);
    int64_t n_tensors = gguf_get_n_tensors(gctx);
    std::vector<uint8_t> buf;
    for (int64_t i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gctx, i);
        struct ggml_tensor * t = ggml_get_tensor(meta_ctx, name);
        size_t size = ggml_nbytes(t);
        size_t offset = data_base + gguf_get_tensor_offset(gctx, i);
        buf.resize(size);
        if (fseek(f, (long) offset, SEEK_SET) != 0 || fread(buf.data(), 1, size, f) != size) {
            fprintf(stderr, "error: failed to read tensor '%s' from '%s'\n", name, path);
            exit(1);
        }
        ggml_backend_tensor_set(t, buf.data(), 0, size);
    }
    fclose(f);

    *out_gguf = gctx;
    return ws;
}

ggml_backend_t init_backend(const std::string & name, int n_threads) {
    if (name == "cpu") {
        ggml_backend_t backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backend, n_threads);
        return backend;
    }
    if (name == "vulkan") {
        if (ggml_backend_vk_get_device_count() == 0) {
            fprintf(stderr, "error: backend 'vulkan' requested but no Vulkan device is available\n");
            exit(1);
        }
        char desc[256];
        ggml_backend_vk_get_device_description(0, desc, sizeof(desc));
        fprintf(stderr, "vulkan device 0: %s\n", desc);
        return ggml_backend_vk_init(0);
    }
    fprintf(stderr, "error: unknown backend '%s' (expected cpu or vulkan)\n", name.c_str());
    exit(1);
}

// ---- persistent KV cache ----

kv_cache init_kv_cache(const hparams & hp, ggml_backend_t backend) {
    kv_cache kv;
    const int64_t head_dim = hp.n_embd_head;

    size_t buf_size = ggml_tensor_overhead() * (size_t)(hp.n_layer * 2 + 8);
    struct ggml_init_params ip = { buf_size, nullptr, /*.no_alloc =*/ true };
    kv.ctx = ggml_init(ip);

    kv.k.resize(hp.n_layer);
    kv.v.resize(hp.n_layer);
    for (int64_t il = 0; il < hp.n_layer; il++) {
        kv.k[il] = ggml_new_tensor_3d(kv.ctx, GGML_TYPE_F16, head_dim, hp.n_head_kv, N_CTX_MAX);
        kv.v[il] = ggml_new_tensor_3d(kv.ctx, GGML_TYPE_F16, head_dim, hp.n_head_kv, N_CTX_MAX);
    }

    kv.buffer = ggml_backend_alloc_ctx_tensors(kv.ctx, backend);
    if (!kv.buffer) {
        fprintf(stderr, "error: failed to allocate KV cache (%lld positions x %lld layers)\n",
                (long long) N_CTX_MAX, (long long) hp.n_layer);
        exit(1);
    }
    return kv;
}

// ---- graph construction: processes only the newest tokens, attending over the KV cache ----

struct built_graph {
    struct ggml_cgraph * gf;
    struct ggml_tensor * tokens;    // I32 [n_new] input
    struct ggml_tensor * positions; // I32 [n_new] input (absolute positions: n_past..n_past+n_new-1)
    struct ggml_tensor * mask;      // F32 [n_kv, n_new] input
    struct ggml_tensor * logits;    // F32 [n_vocab, n_new] output
};

static struct ggml_tensor * rms_norm_mul(struct ggml_context * ctx, struct ggml_tensor * x,
                                          struct ggml_tensor * w, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), w);
}

// Builds Q/K/V for one layer from the post-attn-norm input `x`, as [head_dim, n_head(_kv), n_new]
// tensors ready for RoPE. Branches on hparams::qkv_fused: phi3 has one fused attn_qkv weight
// (split via strided views, since mul_mat's single output can't be reshaped directly), qwen2 has
// separate attn_q/k/v (+ bias) weights (each a fresh contiguous mul_mat output, so reshape works).
static void build_qkv(struct ggml_context * ctx, const hparams & hp, const layer_weights & l,
                       struct ggml_tensor * x, int64_t n_new, int64_t head_dim,
                       struct ggml_tensor ** out_Q, struct ggml_tensor ** out_K,
                       struct ggml_tensor ** out_V) {
    const size_t es = sizeof(float);
    if (hp.qkv_fused) {
        struct ggml_tensor * qkv = ggml_mul_mat(ctx, l.attn_qkv, x); // [n_embd + 2*n_embd_kv, n_new]
        *out_Q = ggml_view_3d(ctx, qkv, head_dim, hp.n_head, n_new,
                               head_dim * es, qkv->nb[1], 0);
        *out_K = ggml_view_3d(ctx, qkv, head_dim, hp.n_head_kv, n_new,
                               head_dim * es, qkv->nb[1], hp.n_head * head_dim * es);
        *out_V = ggml_view_3d(ctx, qkv, head_dim, hp.n_head_kv, n_new,
                               head_dim * es, qkv->nb[1], (hp.n_head + hp.n_head_kv) * head_dim * es);
    } else {
        struct ggml_tensor * q = ggml_mul_mat(ctx, l.attn_q, x); // [n_head*head_dim, n_new]
        struct ggml_tensor * k = ggml_mul_mat(ctx, l.attn_k, x); // [n_head_kv*head_dim, n_new]
        struct ggml_tensor * v = ggml_mul_mat(ctx, l.attn_v, x);
        if (hp.has_qkv_bias) {
            q = ggml_add(ctx, q, l.attn_q_bias);
            k = ggml_add(ctx, k, l.attn_k_bias);
            v = ggml_add(ctx, v, l.attn_v_bias);
        }
        *out_Q = ggml_reshape_3d(ctx, q, head_dim, hp.n_head, n_new);
        *out_K = ggml_reshape_3d(ctx, k, head_dim, hp.n_head_kv, n_new);
        *out_V = ggml_reshape_3d(ctx, v, head_dim, hp.n_head_kv, n_new);
    }
}

// SwiGLU FFN activation (silu(gate) * up), before the down projection. Branches on
// hparams::ffn_fused: phi3 packs gate+up into one ffn_up weight (split via a strided view),
// qwen2 has separate ffn_gate/ffn_up weights.
static struct ggml_tensor * build_ffn_act(struct ggml_context * ctx, const hparams & hp,
                                           const layer_weights & l, struct ggml_tensor * y,
                                           int64_t n_new) {
    struct ggml_tensor * gate;
    struct ggml_tensor * up;
    if (hp.ffn_fused) {
        struct ggml_tensor * gate_up = ggml_mul_mat(ctx, l.ffn_up_gate, y); // [2*n_ff, n_new]
        gate = ggml_view_2d(ctx, gate_up, hp.n_ff, n_new, gate_up->nb[1], 0);
        up   = ggml_view_2d(ctx, gate_up, hp.n_ff, n_new, gate_up->nb[1], hp.n_ff * sizeof(float));
    } else {
        gate = ggml_mul_mat(ctx, l.ffn_gate, y);
        up   = ggml_mul_mat(ctx, l.ffn_up, y);
    }
    return ggml_mul(ctx, ggml_silu(ctx, gate), up);
}

static built_graph build_graph(struct ggml_context * ctx, const model & m, const hparams & hp,
                                kv_cache & kv, int64_t n_past, int64_t n_new) {
    built_graph bg{};
    bg.gf = ggml_new_graph(ctx);

    const int64_t head_dim = hp.n_embd_head;
    const int64_t n_kv     = n_past + n_new;
    const float   kq_scale = 1.0f / sqrtf((float) head_dim);

    bg.tokens    = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_new);
    bg.positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_new);
    bg.mask      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_new);

    struct ggml_tensor * cur = ggml_get_rows(ctx, m.token_embd, bg.tokens); // [n_embd, n_new]

    // ext_factor=1 (LongRoPE/YaRN blending) only for phi3; plain RoPE otherwise (ext_factor=0
    // disables the blend entirely, per ggml_rope_ext's documented defaults)
    const float rope_ext_factor  = hp.has_rope_scaling ? 1.0f : 0.0f;
    const float rope_beta_fast   = hp.has_rope_scaling ? 32.0f : 0.0f;
    const float rope_beta_slow   = hp.has_rope_scaling ? 1.0f : 0.0f;

    for (int64_t il = 0; il < hp.n_layer; il++) {
        const layer_weights & l = m.layers[il];
        struct ggml_tensor * residual = cur;

        struct ggml_tensor * x = rms_norm_mul(ctx, cur, l.attn_norm, hp.rms_eps);

        struct ggml_tensor * Qcur;
        struct ggml_tensor * Kcur;
        struct ggml_tensor * Vcur;
        build_qkv(ctx, hp, l, x, n_new, head_dim, &Qcur, &Kcur, &Vcur);

        if (hp.has_qk_norm) {
            // per-head RMSNorm: rms_norm operates over ne0, which is head_dim here, so this
            // normalizes each (head, token) vector independently -- exactly qwen3's QK-Norm
            Qcur = rms_norm_mul(ctx, Qcur, l.attn_q_norm, hp.rms_eps);
            Kcur = rms_norm_mul(ctx, Kcur, l.attn_k_norm, hp.rms_eps);
        }

        const int rope_mode = hp.rope_neox ? GGML_ROPE_TYPE_NEOX : GGML_ROPE_TYPE_NORMAL;
        Qcur = ggml_rope_ext(ctx, Qcur, bg.positions, m.rope_factors,
                              (int) hp.n_rot, rope_mode, (int) hp.n_ctx_orig,
                              hp.rope_freq_base, 1.0f, rope_ext_factor, hp.rope_attn_factor,
                              rope_beta_fast, rope_beta_slow);
        Kcur = ggml_rope_ext(ctx, Kcur, bg.positions, m.rope_factors,
                              (int) hp.n_rot, rope_mode, (int) hp.n_ctx_orig,
                              hp.rope_freq_base, 1.0f, rope_ext_factor, hp.rope_attn_factor,
                              rope_beta_fast, rope_beta_slow);

        // write the newest K/V (post-RoPE for K) into the persistent cache at [n_past, n_past+n_new)
        struct ggml_tensor * k_dst = ggml_view_3d(ctx, kv.k[il], head_dim, hp.n_head_kv, n_new,
                                                   kv.k[il]->nb[1], kv.k[il]->nb[2],
                                                   n_past * kv.k[il]->nb[2]);
        struct ggml_tensor * v_dst = ggml_view_3d(ctx, kv.v[il], head_dim, hp.n_head_kv, n_new,
                                                   kv.v[il]->nb[1], kv.v[il]->nb[2],
                                                   n_past * kv.v[il]->nb[2]);
        ggml_build_forward_expand(bg.gf, ggml_cpy(ctx, Kcur, k_dst));
        ggml_build_forward_expand(bg.gf, ggml_cpy(ctx, Vcur, v_dst));

        struct ggml_tensor * K_full = ggml_view_3d(ctx, kv.k[il], head_dim, hp.n_head_kv, n_kv,
                                                     kv.k[il]->nb[1], kv.k[il]->nb[2], 0);
        struct ggml_tensor * V_full = ggml_view_3d(ctx, kv.v[il], head_dim, hp.n_head_kv, n_kv,
                                                     kv.v[il]->nb[1], kv.v[il]->nb[2], 0);

        struct ggml_tensor * Q = ggml_permute(ctx, Qcur, 0, 2, 1, 3);   // [head_dim, n_new, n_head]
        struct ggml_tensor * K = ggml_permute(ctx, K_full, 0, 2, 1, 3); // [head_dim, n_kv, n_head_kv]

        struct ggml_tensor * KQ = ggml_mul_mat(ctx, K, Q); // [n_kv, n_new, n_head] (GQA-broadcast)
        KQ = ggml_soft_max_ext(ctx, KQ, bg.mask, kq_scale, 0.0f);

        struct ggml_tensor * V_trans = ggml_cont_3d(ctx,
            ggml_permute(ctx, V_full, 1, 2, 0, 3), n_kv, head_dim, hp.n_head_kv); // [n_kv, head_dim, n_head_kv]

        struct ggml_tensor * KQV = ggml_mul_mat(ctx, V_trans, KQ); // [head_dim, n_new, n_head]
        struct ggml_tensor * merged = ggml_permute(ctx, KQV, 0, 2, 1, 3); // [head_dim, n_head, n_new]
        // n_head*head_dim is the attn_output projection's *input* width, which isn't always
        // n_embd (qwen3 decouples them -- see hparams::n_embd_head)
        cur = ggml_cont_2d(ctx, merged, hp.n_head * head_dim, n_new);

        cur = ggml_mul_mat(ctx, l.attn_output, cur);
        cur = ggml_add(ctx, cur, residual);

        residual = cur;
        struct ggml_tensor * y = rms_norm_mul(ctx, cur, l.ffn_norm, hp.rms_eps);
        struct ggml_tensor * ffn_act = build_ffn_act(ctx, hp, l, y, n_new);

        cur = ggml_mul_mat(ctx, l.ffn_down, ffn_act);
        cur = ggml_add(ctx, cur, residual);
    }

    cur = rms_norm_mul(ctx, cur, m.output_norm, hp.rms_eps);
    bg.logits = ggml_mul_mat(ctx, m.output, cur); // [n_vocab, n_new]

    ggml_build_forward_expand(bg.gf, bg.logits);
    return bg;
}

decode_session init_decode_session(ggml_backend_t backend) {
    decode_session ds;
    size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    ds.graph_buf.resize(buf_size);
    ds.galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    return ds;
}

void free_decode_session(decode_session & ds) {
    ggml_gallocr_free(ds.galloc);
    ds.galloc = nullptr;
}

std::vector<float> forward_step(const model & m, const hparams & hp, kv_cache & kv,
                                 decode_session & ds, ggml_backend_t backend, int64_t n_past,
                                 const std::vector<int32_t> & new_tokens, int64_t n_vocab) {
    int64_t n_new = (int64_t) new_tokens.size();
    int64_t n_kv  = n_past + n_new;

    struct ggml_init_params gip = {
        /*.mem_size   =*/ ds.graph_buf.size(),
        /*.mem_buffer =*/ ds.graph_buf.data(),
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * gctx_build = ggml_init(gip);

    built_graph bg = build_graph(gctx_build, m, hp, kv, n_past, n_new);

    // reused across calls: replans the (possibly differently-shaped) activation layout each
    // time rather than malloc/free-ing a fresh allocator every step
    if (!ggml_gallocr_alloc_graph(ds.galloc, bg.gf)) {
        fprintf(stderr, "error: graph allocation failed\n");
        exit(1);
    }

    ggml_backend_tensor_set(bg.tokens, new_tokens.data(), 0, n_new * sizeof(int32_t));

    std::vector<int32_t> positions(n_new);
    for (int64_t i = 0; i < n_new; i++) positions[i] = (int32_t)(n_past + i);
    ggml_backend_tensor_set(bg.positions, positions.data(), 0, n_new * sizeof(int32_t));

    std::vector<float> mask((size_t)(n_kv * n_new));
    for (int64_t q = 0; q < n_new; q++) {
        int64_t abs_q = n_past + q;
        for (int64_t k = 0; k < n_kv; k++) {
            mask[q * n_kv + k] = (k <= abs_q) ? 0.0f : -INFINITY;
        }
    }
    ggml_backend_tensor_set(bg.mask, mask.data(), 0, mask.size() * sizeof(float));

    ggml_backend_graph_compute(backend, bg.gf);

    std::vector<float> last_logits(n_vocab);
    size_t last_col_offset = (size_t)(n_new - 1) * n_vocab * sizeof(float);
    ggml_backend_tensor_get(bg.logits, last_logits.data(), last_col_offset, n_vocab * sizeof(float));

    ggml_free(gctx_build); // frees only the ggml_context bookkeeping; ds.graph_buf is ours to reuse

    return last_logits;
}

int32_t argmax(const std::vector<float> & logits) {
    int32_t best = 0;
    float best_v = logits[0];
    for (size_t i = 1; i < logits.size(); i++) {
        if (logits[i] > best_v) { best_v = logits[i]; best = (int32_t) i; }
    }
    return best;
}

sampler init_sampler(const sampler_params & params) {
    sampler smp;
    smp.params = params;
    smp.rng.seed(params.seed != 0 ? params.seed : std::random_device{}());
    return smp;
}

int32_t sample(sampler & smp, std::vector<float> & logits, const std::vector<int32_t> & recent_tokens) {
    const sampler_params & p = smp.params;

    if (p.repeat_penalty != 1.0f && !recent_tokens.empty()) {
        size_t start = recent_tokens.size() > (size_t) p.repeat_last_n
                            ? recent_tokens.size() - (size_t) p.repeat_last_n : 0;
        for (size_t i = start; i < recent_tokens.size(); i++) {
            int32_t id = recent_tokens[i];
            if (id < 0 || (size_t) id >= logits.size()) continue;
            float & l = logits[id];
            l = l > 0.0f ? l / p.repeat_penalty : l * p.repeat_penalty;
        }
    }

    if (p.temp <= 0.0f) {
        return argmax(logits);
    }

    std::vector<int32_t> ids(logits.size());
    for (size_t i = 0; i < logits.size(); i++) ids[i] = (int32_t) i;

    size_t k = (p.top_k > 0 && (size_t) p.top_k < ids.size()) ? (size_t) p.top_k : ids.size();
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
                       [&](int32_t a, int32_t b) { return logits[a] > logits[b]; });
    ids.resize(k);

    float max_logit = logits[ids[0]];
    std::vector<float> probs(ids.size());
    float sum = 0.0f;
    for (size_t i = 0; i < ids.size(); i++) {
        probs[i] = expf((logits[ids[i]] - max_logit) / p.temp);
        sum += probs[i];
    }
    for (float & pr : probs) pr /= sum;

    // ids/probs are sorted descending (from the partial_sort above), so a running cumulative
    // sum directly gives the nucleus cutoff
    if (p.top_p < 1.0f) {
        float cum = 0.0f;
        size_t cutoff = probs.size();
        for (size_t i = 0; i < probs.size(); i++) {
            cum += probs[i];
            if (cum >= p.top_p) { cutoff = i + 1; break; }
        }
        ids.resize(cutoff);
        probs.resize(cutoff);
        float s = 0.0f;
        for (float pr : probs) s += pr;
        for (float & pr : probs) pr /= s;
    }

    std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
    return ids[dist(smp.rng)];
}
