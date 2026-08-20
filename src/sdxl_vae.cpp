#include "sdxl_vae.h"
#include "sdxl_common.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

// standard SD1.x/SDXL VAE config -- see file header in sdxl_vae.h
static constexpr int64_t VAE_CH             = 128;
static constexpr int64_t VAE_Z_CHANNELS     = 4;
static constexpr int64_t VAE_NUM_RES_BLOCKS = 2; // decoder uses NUM_RES_BLOCKS+1 per level
static const int64_t VAE_CH_MULT[4] = { 1, 2, 4, 4 };

static vae_resnet_block load_resnet(struct ggml_context * ctx, const std::string & p,
                                     bool has_shortcut) {
    vae_resnet_block b;
    b.norm1_w = sdxl_must_get(ctx, p + "norm1.weight");
    b.norm1_b = sdxl_must_get(ctx, p + "norm1.bias");
    b.conv1_w = sdxl_must_get(ctx, p + "conv1.weight");
    b.conv1_b = sdxl_must_get(ctx, p + "conv1.bias");
    b.norm2_w = sdxl_must_get(ctx, p + "norm2.weight");
    b.norm2_b = sdxl_must_get(ctx, p + "norm2.bias");
    b.conv2_w = sdxl_must_get(ctx, p + "conv2.weight");
    b.conv2_b = sdxl_must_get(ctx, p + "conv2.bias");
    if (has_shortcut) {
        b.nin_shortcut_w = sdxl_must_get(ctx, p + "nin_shortcut.weight");
        b.nin_shortcut_b = sdxl_must_get(ctx, p + "nin_shortcut.bias");
    }
    return b;
}

static vae_attn_block load_attn(struct ggml_context * ctx, const std::string & p) {
    vae_attn_block b;
    b.norm_w     = sdxl_must_get(ctx, p + "norm.weight");
    b.norm_b     = sdxl_must_get(ctx, p + "norm.bias");
    b.q_w        = sdxl_must_get(ctx, p + "q.weight");
    b.q_b        = sdxl_must_get(ctx, p + "q.bias");
    b.k_w        = sdxl_must_get(ctx, p + "k.weight");
    b.k_b        = sdxl_must_get(ctx, p + "k.bias");
    b.v_w        = sdxl_must_get(ctx, p + "v.weight");
    b.v_b        = sdxl_must_get(ctx, p + "v.bias");
    b.proj_out_w = sdxl_must_get(ctx, p + "proj_out.weight");
    b.proj_out_b = sdxl_must_get(ctx, p + "proj_out.bias");
    return b;
}

vae_decoder load_vae_decoder(struct ggml_context * wctx) {
    vae_decoder vae;
    const std::string base = "first_stage_model.decoder.";

    vae.conv_in_w = sdxl_must_get(wctx, base + "conv_in.weight");
    vae.conv_in_b = sdxl_must_get(wctx, base + "conv_in.bias");

    vae.mid_block_1 = load_resnet(wctx, base + "mid.block_1.", /*has_shortcut=*/false);
    vae.mid_attn_1  = load_attn(wctx, base + "mid.attn_1.");
    vae.mid_block_2 = load_resnet(wctx, base + "mid.block_2.", /*has_shortcut=*/false);

    vae.up.resize(4);
    int64_t block_in = VAE_CH * VAE_CH_MULT[3]; // matches mid block_in (512)
    for (int i = 3; i >= 0; i--) {
        int64_t block_out = VAE_CH * VAE_CH_MULT[i];
        vae_up_level & lvl = vae.up[i];
        for (int j = 0; j < VAE_NUM_RES_BLOCKS + 1; j++) {
            std::string p = base + "up." + std::to_string(i) + ".block." + std::to_string(j) + ".";
            lvl.blocks.push_back(load_resnet(wctx, p, /*has_shortcut=*/block_in != block_out));
            block_in = block_out;
        }
        if (i != 0) {
            std::string p = base + "up." + std::to_string(i) + ".upsample.";
            lvl.upsample_conv_w = sdxl_must_get(wctx, p + "conv.weight");
            lvl.upsample_conv_b = sdxl_must_get(wctx, p + "conv.bias");
        }
    }

    vae.norm_out_w = sdxl_must_get(wctx, base + "norm_out.weight");
    vae.norm_out_b = sdxl_must_get(wctx, base + "norm_out.bias");
    vae.conv_out_w = sdxl_must_get(wctx, base + "conv_out.weight");
    vae.conv_out_b = sdxl_must_get(wctx, base + "conv_out.bias");

    return vae;
}

// ---- graph building (GroupNorm/Conv2d helpers now shared via sdxl_common.h) ----

static struct ggml_tensor * resnet_fwd(struct ggml_context * ctx, const vae_resnet_block & rb,
                                        struct ggml_tensor * x) {
    struct ggml_tensor * h = sdxl_group_norm(ctx, x, rb.norm1_w, rb.norm1_b);
    h = ggml_silu(ctx, h);
    h = sdxl_conv2d(ctx, rb.conv1_w, rb.conv1_b, h, 1, 1, 1, 1, 1, 1);

    h = sdxl_group_norm(ctx, h, rb.norm2_w, rb.norm2_b);
    h = ggml_silu(ctx, h);
    h = sdxl_conv2d(ctx, rb.conv2_w, rb.conv2_b, h, 1, 1, 1, 1, 1, 1);

    struct ggml_tensor * residual = x;
    if (rb.nin_shortcut_w) {
        residual = sdxl_conv2d(ctx, rb.nin_shortcut_w, rb.nin_shortcut_b, x, 1, 1, 0, 0, 1, 1);
    }
    return ggml_add(ctx, h, residual);
}

// single-head spatial self-attention over the h*w "pixels" -- see sdxl_vae.h / auto_encoder_kl.hpp
static struct ggml_tensor * attn_fwd(struct ggml_context * ctx, const vae_attn_block & ab,
                                      struct ggml_tensor * x) {
    struct ggml_tensor * h = sdxl_group_norm(ctx, x, ab.norm_w, ab.norm_b);
    const int64_t w = h->ne[0], hh = h->ne[1], c = h->ne[2], n = h->ne[3];

    auto to_seq = [&](struct ggml_tensor * t) {
        t = ggml_cont(ctx, ggml_permute(ctx, t, 1, 2, 0, 3)); // [w,h,c,n] -> [c,w,h,n]
        return ggml_reshape_3d(ctx, t, c, w * hh, n);          // [c, seq, n]
    };

    struct ggml_tensor * q = to_seq(sdxl_conv2d(ctx, ab.q_w, ab.q_b, h, 1, 1, 0, 0, 1, 1));
    struct ggml_tensor * k = to_seq(sdxl_conv2d(ctx, ab.k_w, ab.k_b, h, 1, 1, 0, 0, 1, 1));
    struct ggml_tensor * v = to_seq(sdxl_conv2d(ctx, ab.v_w, ab.v_b, h, 1, 1, 0, 0, 1, 1));

    float scale = 1.0f / sqrtf((float) c);
    struct ggml_tensor * kq = ggml_mul_mat(ctx, k, q); // [seq_k, seq_q, n]
    kq = ggml_soft_max_ext(ctx, kq, nullptr, scale, 0.0f);

    struct ggml_tensor * v_trans = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3)); // [seq, c, n]
    struct ggml_tensor * kqv = ggml_mul_mat(ctx, v_trans, kq); // [c, seq_q, n]

    struct ggml_tensor * out = ggml_cont(ctx, ggml_permute(ctx, kqv, 1, 0, 2, 3)); // [seq, c, n]
    out = ggml_reshape_4d(ctx, out, w, hh, c, n); // [w, h, c, n]

    out = sdxl_conv2d(ctx, ab.proj_out_w, ab.proj_out_b, out, 1, 1, 0, 0, 1, 1);
    return ggml_add(ctx, out, x);
}

static struct ggml_tensor * upsample_fwd(struct ggml_context * ctx, struct ggml_tensor * w,
                                          struct ggml_tensor * b, struct ggml_tensor * x) {
    x = ggml_upscale(ctx, x, 2, GGML_SCALE_MODE_NEAREST);
    return sdxl_conv2d(ctx, w, b, x, 1, 1, 1, 1, 1, 1);
}

static struct ggml_tensor * vae_decode_graph(struct ggml_context * ctx, const vae_decoder & vae,
                                              struct ggml_tensor * z,
                                              std::vector<std::pair<std::string, struct ggml_tensor *>> * checkpoints = nullptr) {
    auto mark = [&](const char * name, struct ggml_tensor * t) {
        if (checkpoints) {
            ggml_set_output(t); // protect from ggml_gallocr's buffer-reuse planning -- without this
                                 // a checkpoint's memory can legitimately get overwritten by a later
                                 // tensor before we read it back post-compute (bit us once already)
            checkpoints->push_back({name, t});
        }
    };

    struct ggml_tensor * h = sdxl_conv2d(ctx, vae.conv_in_w, vae.conv_in_b, z, 1, 1, 1, 1, 1, 1);
    mark("conv_in", h);

    h = resnet_fwd(ctx, vae.mid_block_1, h);
    mark("mid_block_1", h);
    h = attn_fwd(ctx, vae.mid_attn_1, h);
    mark("mid_attn_1", h);
    h = resnet_fwd(ctx, vae.mid_block_2, h);
    mark("mid_block_2", h);

    for (int i = 3; i >= 0; i--) {
        const vae_up_level & lvl = vae.up[i];
        int j = 0;
        for (const vae_resnet_block & rb : lvl.blocks) {
            h = resnet_fwd(ctx, rb, h);
            mark(("up_" + std::to_string(i) + "_block_" + std::to_string(j++)).c_str(), h);
        }
        if (lvl.upsample_conv_w) {
            h = upsample_fwd(ctx, lvl.upsample_conv_w, lvl.upsample_conv_b, h);
            mark(("up_" + std::to_string(i) + "_upsample").c_str(), h);
        }
    }

    h = sdxl_group_norm(ctx, h, vae.norm_out_w, vae.norm_out_b);
    mark("norm_out", h);
    h = ggml_silu(ctx, h);
    h = sdxl_conv2d(ctx, vae.conv_out_w, vae.conv_out_b, h, 1, 1, 1, 1, 1, 1);
    return h; // [w*8, h*8, 3, n]
}

std::vector<float> vae_decode(const vae_decoder & vae, ggml_backend_t backend,
                               const std::vector<float> & latent_data,
                               int64_t latent_w, int64_t latent_h) {
    // generous fixed budget: this graph has on the order of a few hundred nodes (~12 resnets *
    // ~9 ops + 1 attn block + upsamples), nowhere near GGML_DEFAULT_GRAPH_SIZE
    size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    std::vector<uint8_t> graph_buf(buf_size);
    struct ggml_init_params gip = { graph_buf.size(), graph_buf.data(), /*.no_alloc=*/true };
    struct ggml_context * ctx = ggml_init(gip);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    struct ggml_tensor * z = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, latent_w, latent_h,
                                                 VAE_Z_CHANNELS, 1);
    bool debug = getenv("SDXL_VAE_DEBUG") != nullptr;
    std::vector<std::pair<std::string, struct ggml_tensor *>> checkpoints;
    struct ggml_tensor * out = vae_decode_graph(ctx, vae, z, debug ? &checkpoints : nullptr);
    ggml_build_forward_expand(gf, out);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "error: VAE graph allocation failed\n");
        exit(1);
    }

    ggml_backend_tensor_set(z, latent_data.data(), 0, latent_data.size() * sizeof(float));
    ggml_backend_graph_compute(backend, gf);

    if (debug) {
        for (auto & [name, t] : checkpoints) {
            std::vector<float> v(ggml_nelements(t));
            ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
            float vmin = v[0], vmax = v[0], sum = 0.0f;
            int n_nan = 0;
            for (float x : v) {
                if (std::isnan(x)) { n_nan++; continue; }
                vmin = std::min(vmin, x);
                vmax = std::max(vmax, x);
                sum += x;
            }
            fprintf(stderr, "  [vae] %-16s n=%-8zu mean=%.4f min=%.4f max=%.4f nan=%d\n",
                    name.c_str(), v.size(), sum / v.size(), vmin, vmax, n_nan);
        }
    }

    std::vector<float> result(ggml_nelements(out));
    ggml_backend_tensor_get(out, result.data(), 0, result.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return result;
}
