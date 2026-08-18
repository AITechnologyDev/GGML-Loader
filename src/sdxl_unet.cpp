#include "sdxl_unet.h"
#include "sdxl_common.h"

#include "ggml-alloc.h"

#include <cstdio>
#include <cstdlib>
#include <string>

// SDXL-fixed config -- see file header in sdxl_unet.h
static constexpr int64_t UNET_MODEL_CHANNELS = 320;
static constexpr int64_t UNET_CONTEXT_DIM    = 2048;
static constexpr int64_t UNET_HEAD_DIM       = 64;
static const int64_t UNET_CHANNEL_MULT[3]      = {1, 2, 4};
static const int64_t UNET_TRANSFORMER_DEPTH[3] = {1, 2, 10}; // unused at level 0 (no attention there)
static const int UNET_ATTN_RESOLUTIONS[2]      = {4, 2};     // ds values that get a SpatialTransformer
static constexpr int UNET_NUM_RES_BLOCKS = 2;

static bool has_attn_at(int ds) {
    for (int a : UNET_ATTN_RESOLUTIONS) if (a == ds) return true;
    return false;
}

static unet_resblock load_resblock(struct ggml_context * ctx, const std::string & p, bool has_shortcut) {
    unet_resblock r;
    r.in_norm_w = sdxl_must_get(ctx, p + "in_layers.0.weight");
    r.in_norm_b = sdxl_must_get(ctx, p + "in_layers.0.bias");
    r.in_conv_w = sdxl_must_get(ctx, p + "in_layers.2.weight");
    r.in_conv_b = sdxl_must_get(ctx, p + "in_layers.2.bias");
    r.emb_w     = sdxl_must_get(ctx, p + "emb_layers.1.weight");
    r.emb_b     = sdxl_must_get(ctx, p + "emb_layers.1.bias");
    r.out_norm_w = sdxl_must_get(ctx, p + "out_layers.0.weight");
    r.out_norm_b = sdxl_must_get(ctx, p + "out_layers.0.bias");
    r.out_conv_w = sdxl_must_get(ctx, p + "out_layers.3.weight");
    r.out_conv_b = sdxl_must_get(ctx, p + "out_layers.3.bias");
    if (has_shortcut) {
        r.skip_w = sdxl_must_get(ctx, p + "skip_connection.weight");
        r.skip_b = sdxl_must_get(ctx, p + "skip_connection.bias");
    }
    return r;
}

static unet_xattn_layer load_xattn_layer(struct ggml_context * ctx, const std::string & p) {
    unet_xattn_layer l;
    l.norm1_w = sdxl_must_get(ctx, p + "norm1.weight");
    l.norm1_b = sdxl_must_get(ctx, p + "norm1.bias");
    l.attn1_q_w = sdxl_must_get(ctx, p + "attn1.to_q.weight");
    l.attn1_k_w = sdxl_must_get(ctx, p + "attn1.to_k.weight");
    l.attn1_v_w = sdxl_must_get(ctx, p + "attn1.to_v.weight");
    l.attn1_out_w = sdxl_must_get(ctx, p + "attn1.to_out.0.weight");
    l.attn1_out_b = sdxl_must_get(ctx, p + "attn1.to_out.0.bias");
    l.norm2_w = sdxl_must_get(ctx, p + "norm2.weight");
    l.norm2_b = sdxl_must_get(ctx, p + "norm2.bias");
    l.attn2_q_w = sdxl_must_get(ctx, p + "attn2.to_q.weight");
    l.attn2_k_w = sdxl_must_get(ctx, p + "attn2.to_k.weight");
    l.attn2_v_w = sdxl_must_get(ctx, p + "attn2.to_v.weight");
    l.attn2_out_w = sdxl_must_get(ctx, p + "attn2.to_out.0.weight");
    l.attn2_out_b = sdxl_must_get(ctx, p + "attn2.to_out.0.bias");
    l.norm3_w = sdxl_must_get(ctx, p + "norm3.weight");
    l.norm3_b = sdxl_must_get(ctx, p + "norm3.bias");
    l.ff_proj_w = sdxl_must_get(ctx, p + "ff.net.0.proj.weight");
    l.ff_proj_b = sdxl_must_get(ctx, p + "ff.net.0.proj.bias");
    l.ff_out_w = sdxl_must_get(ctx, p + "ff.net.2.weight");
    l.ff_out_b = sdxl_must_get(ctx, p + "ff.net.2.bias");
    return l;
}

static unet_spatial_transformer load_spatial_transformer(struct ggml_context * ctx,
                                                           const std::string & p, int depth, int n_head) {
    unet_spatial_transformer t;
    t.n_head = n_head;
    t.norm_w = sdxl_must_get(ctx, p + "norm.weight");
    t.norm_b = sdxl_must_get(ctx, p + "norm.bias");
    t.proj_in_w = sdxl_must_get(ctx, p + "proj_in.weight");
    t.proj_in_b = sdxl_must_get(ctx, p + "proj_in.bias");
    t.proj_out_w = sdxl_must_get(ctx, p + "proj_out.weight");
    t.proj_out_b = sdxl_must_get(ctx, p + "proj_out.bias");
    t.layers.reserve(depth);
    for (int i = 0; i < depth; i++) {
        t.layers.push_back(load_xattn_layer(ctx, p + "transformer_blocks." + std::to_string(i) + "."));
    }
    return t;
}

sdxl_unet load_sdxl_unet(struct ggml_context * wctx) {
    sdxl_unet u;
    const std::string base = "model.diffusion_model.";

    u.time_embed_0_w = sdxl_must_get(wctx, base + "time_embed.0.weight");
    u.time_embed_0_b = sdxl_must_get(wctx, base + "time_embed.0.bias");
    u.time_embed_2_w = sdxl_must_get(wctx, base + "time_embed.2.weight");
    u.time_embed_2_b = sdxl_must_get(wctx, base + "time_embed.2.bias");
    u.label_emb_0_w = sdxl_must_get(wctx, base + "label_emb.0.0.weight");
    u.label_emb_0_b = sdxl_must_get(wctx, base + "label_emb.0.0.bias");
    u.label_emb_2_w = sdxl_must_get(wctx, base + "label_emb.0.2.weight");
    u.label_emb_2_b = sdxl_must_get(wctx, base + "label_emb.0.2.bias");
    u.conv_in_w = sdxl_must_get(wctx, base + "input_blocks.0.0.weight");
    u.conv_in_b = sdxl_must_get(wctx, base + "input_blocks.0.0.bias");

    // Mirrors UnetModelBlock's own index bookkeeping (input_block_idx, ds, channel stack) directly,
    // rather than hand-deriving a fixed 9-slot layout, so this can't silently drift from how the
    // checkpoint's blocks are actually numbered/ordered.
    std::vector<int64_t> chan_stack;
    chan_stack.push_back(UNET_MODEL_CHANNELS);
    int64_t ch = UNET_MODEL_CHANNELS;
    int input_block_idx = 0;
    int ds = 1;
    for (int i = 0; i < 3; i++) {
        int64_t mult = UNET_CHANNEL_MULT[i];
        for (int j = 0; j < UNET_NUM_RES_BLOCKS; j++) {
            input_block_idx++;
            std::string p = base + "input_blocks." + std::to_string(input_block_idx) + ".0.";
            unet_down_slot slot;
            slot.res = load_resblock(wctx, p, /*has_shortcut=*/ch != mult * UNET_MODEL_CHANNELS);
            ch = mult * UNET_MODEL_CHANNELS;
            if (has_attn_at(ds)) {
                slot.has_attn = true;
                int64_t n_head = ch / UNET_HEAD_DIM;
                slot.attn = load_spatial_transformer(wctx,
                    base + "input_blocks." + std::to_string(input_block_idx) + ".1.",
                    (int) UNET_TRANSFORMER_DEPTH[i], (int) n_head);
            }
            u.down.push_back(std::move(slot));
            chan_stack.push_back(ch);
        }
        if (i != 2) {
            input_block_idx++;
            unet_down_slot slot;
            slot.is_downsample = true;
            std::string p = base + "input_blocks." + std::to_string(input_block_idx) + ".0.op.";
            slot.down_w = sdxl_must_get(wctx, p + "weight");
            slot.down_b = sdxl_must_get(wctx, p + "bias");
            u.down.push_back(std::move(slot));
            chan_stack.push_back(ch);
            ds *= 2;
        }
    }

    int64_t mid_n_head = ch / UNET_HEAD_DIM;
    u.mid_res1 = load_resblock(wctx, base + "middle_block.0.", /*has_shortcut=*/false);
    u.mid_attn = load_spatial_transformer(wctx, base + "middle_block.1.",
                                           (int) UNET_TRANSFORMER_DEPTH[2], (int) mid_n_head);
    u.mid_res2 = load_resblock(wctx, base + "middle_block.2.", /*has_shortcut=*/false);

    int output_block_idx = 0;
    for (int i = 2; i >= 0; i--) {
        int64_t mult = UNET_CHANNEL_MULT[i];
        for (int j = 0; j < UNET_NUM_RES_BLOCKS + 1; j++) {
            int64_t ich = chan_stack.back();
            chan_stack.pop_back();

            unet_up_slot slot;
            std::string p = base + "output_blocks." + std::to_string(output_block_idx) + ".0.";
            slot.res = load_resblock(wctx, p, /*has_shortcut=*/(ch + ich) != mult * UNET_MODEL_CHANNELS);
            ch = mult * UNET_MODEL_CHANNELS;

            int up_sample_idx = 1;
            if (has_attn_at(ds)) {
                slot.has_attn = true;
                int64_t n_head = ch / UNET_HEAD_DIM;
                slot.attn = load_spatial_transformer(wctx,
                    base + "output_blocks." + std::to_string(output_block_idx) + ".1.",
                    (int) UNET_TRANSFORMER_DEPTH[i], (int) n_head);
                up_sample_idx = 2;
            }
            if (i > 0 && j == UNET_NUM_RES_BLOCKS) {
                slot.has_upsample = true;
                std::string up = base + "output_blocks." + std::to_string(output_block_idx) + "." +
                                  std::to_string(up_sample_idx) + ".conv.";
                slot.up_w = sdxl_must_get(wctx, up + "weight");
                slot.up_b = sdxl_must_get(wctx, up + "bias");
                ds /= 2;
            }
            u.up.push_back(std::move(slot));
            output_block_idx++;
        }
    }

    u.out_norm_w = sdxl_must_get(wctx, base + "out.0.weight");
    u.out_norm_b = sdxl_must_get(wctx, base + "out.0.bias");
    u.out_conv_w = sdxl_must_get(wctx, base + "out.2.weight");
    u.out_conv_b = sdxl_must_get(wctx, base + "out.2.bias");

    return u;
}

// ---- graph building ----

// proj_in/proj_out are stored either as a genuine 2D Linear [in,out] or, as in this checkpoint, a
// 1x1 Conv2d [1,1,in,out] -- mathematically identical since a 1x1 kernel does no spatial mixing.
// ggml_n_dims drops trailing size-1 dims, so it's 2 for a real Linear and 4 for the conv-shaped one;
// reshape the latter down to [in,out] for a uniform sdxl_linear() call either way.
static struct ggml_tensor * as_linear_2d(struct ggml_context * ctx, struct ggml_tensor * w) {
    return ggml_n_dims(w) <= 2 ? w : ggml_reshape_2d(ctx, w, w->ne[2], w->ne[3]);
}

static struct ggml_tensor * resblock_fwd(struct ggml_context * ctx, const unet_resblock & r,
                                          struct ggml_tensor * x, struct ggml_tensor * emb) {
    struct ggml_tensor * h = sdxl_group_norm(ctx, x, r.in_norm_w, r.in_norm_b);
    h = ggml_silu(ctx, h);
    h = sdxl_conv2d(ctx, r.in_conv_w, r.in_conv_b, h, 1, 1, 1, 1, 1, 1);

    struct ggml_tensor * emb_out = ggml_silu(ctx, emb);
    emb_out = sdxl_linear(ctx, r.emb_w, r.emb_b, emb_out); // [out_ch, N]
    emb_out = ggml_reshape_4d(ctx, emb_out, 1, 1, emb_out->ne[0], emb_out->ne[1]); // [1,1,out_ch,N]
    h = ggml_add(ctx, h, emb_out);

    h = sdxl_group_norm(ctx, h, r.out_norm_w, r.out_norm_b);
    h = ggml_silu(ctx, h);
    h = sdxl_conv2d(ctx, r.out_conv_w, r.out_conv_b, h, 1, 1, 1, 1, 1, 1);

    struct ggml_tensor * residual = x;
    if (r.skip_w) {
        residual = sdxl_conv2d(ctx, r.skip_w, r.skip_b, x, 1, 1, 0, 0, 1, 1);
    }
    return ggml_add(ctx, h, residual);
}

// x, context: [C, seq, N]. GEGLU feed-forward: proj to 2*inner_dim, split, gate one half through
// GELU, multiply, project back down -- see FeedForward/GEGLU in block.hpp.
static struct ggml_tensor * xattn_layer_fwd(struct ggml_context * ctx, const unet_xattn_layer & l,
                                             struct ggml_tensor * x, struct ggml_tensor * context,
                                             int n_head) {
    struct ggml_tensor * r = x;
    struct ggml_tensor * h = sdxl_layer_norm(ctx, x, l.norm1_w, l.norm1_b);
    struct ggml_tensor * q = sdxl_linear(ctx, l.attn1_q_w, nullptr, h);
    struct ggml_tensor * k = sdxl_linear(ctx, l.attn1_k_w, nullptr, h);
    struct ggml_tensor * v = sdxl_linear(ctx, l.attn1_v_w, nullptr, h);
    h = sdxl_attention(ctx, q, k, v, n_head, nullptr); // self-attention, unmasked
    h = sdxl_linear(ctx, l.attn1_out_w, l.attn1_out_b, h);
    x = ggml_add(ctx, r, h);

    r = x;
    h = sdxl_layer_norm(ctx, x, l.norm2_w, l.norm2_b);
    q = sdxl_linear(ctx, l.attn2_q_w, nullptr, h);
    k = sdxl_linear(ctx, l.attn2_k_w, nullptr, context); // cross-attention: k/v from the CLIP context
    v = sdxl_linear(ctx, l.attn2_v_w, nullptr, context);
    h = sdxl_attention(ctx, q, k, v, n_head, nullptr);
    h = sdxl_linear(ctx, l.attn2_out_w, l.attn2_out_b, h);
    x = ggml_add(ctx, r, h);

    r = x;
    h = sdxl_layer_norm(ctx, x, l.norm3_w, l.norm3_b);
    h = sdxl_linear(ctx, l.ff_proj_w, l.ff_proj_b, h); // [2*inner_dim, seq, N]
    int64_t inner = h->ne[0] / 2;
    struct ggml_tensor * h_part = ggml_cont(ctx, ggml_view_3d(ctx, h, inner, h->ne[1], h->ne[2],
                                                               h->nb[1], h->nb[2], 0));
    struct ggml_tensor * gate = ggml_cont(ctx, ggml_view_3d(ctx, h, inner, h->ne[1], h->ne[2],
                                                             h->nb[1], h->nb[2], inner * ggml_element_size(h)));
    gate = ggml_gelu(ctx, gate);
    h = ggml_mul(ctx, h_part, gate);
    h = sdxl_linear(ctx, l.ff_out_w, l.ff_out_b, h);
    x = ggml_add(ctx, r, h);
    return x;
}

// x: [W,H,C,N]. context: [context_dim, n_ctx, N].
static struct ggml_tensor * spatial_transformer_fwd(struct ggml_context * ctx,
                                                      const unet_spatial_transformer & t,
                                                      struct ggml_tensor * x, struct ggml_tensor * context) {
    struct ggml_tensor * x_in = x;
    const int64_t w = x->ne[0], hh = x->ne[1], c = x->ne[2], n = x->ne[3];

    struct ggml_tensor * h = sdxl_group_norm(ctx, x, t.norm_w, t.norm_b);
    h = ggml_cont(ctx, ggml_permute(ctx, h, 1, 2, 0, 3)); // [w,h,c,n] -> [c,w,h,n]
    h = ggml_reshape_3d(ctx, h, c, w * hh, n);              // [c, seq, n]
    h = sdxl_linear(ctx, as_linear_2d(ctx, t.proj_in_w), t.proj_in_b, h);

    for (const unet_xattn_layer & layer : t.layers) {
        h = xattn_layer_fwd(ctx, layer, h, context, t.n_head);
    }

    h = sdxl_linear(ctx, as_linear_2d(ctx, t.proj_out_w), t.proj_out_b, h);
    h = ggml_cont(ctx, ggml_permute(ctx, h, 1, 0, 2, 3)); // [c,seq,n] -> [seq,c,n]
    h = ggml_reshape_4d(ctx, h, w, hh, c, n);              // [w,h,c,n]

    return ggml_add(ctx, h, x_in);
}

std::vector<float> unet_forward(const sdxl_unet & u, ggml_backend_t backend,
                                 const std::vector<float> & latent, int64_t latent_w, int64_t latent_h,
                                 float timestep, const std::vector<float> & context, int context_len,
                                 const std::vector<float> & y) {
    // Down (9) + mid (3) + up (9) blocks, several with up to 10 BasicTransformerBlocks each (level-2
    // attention resolution) -- comfortably exceeds ggml_new_graph()'s default 2048-node size (the
    // CLIP encoder hit this same assert during testing), so size it explicitly and generously.
    const size_t n_graph_nodes = 65536;
    size_t buf_size = ggml_tensor_overhead() * n_graph_nodes + ggml_graph_overhead_custom(n_graph_nodes, false);
    std::vector<uint8_t> graph_buf(buf_size);
    struct ggml_init_params gip = { graph_buf.size(), graph_buf.data(), /*.no_alloc=*/true };
    struct ggml_context * ctx = ggml_init(gip);
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, n_graph_nodes, false);

    struct ggml_tensor * z        = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, latent_w, latent_h, 4, 1);
    struct ggml_tensor * t_scalar = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * context_t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, UNET_CONTEXT_DIM, context_len, 1);
    struct ggml_tensor * y_t      = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) y.size(), 1);

    struct ggml_tensor * t_emb = ggml_timestep_embedding(ctx, t_scalar, UNET_MODEL_CHANNELS, 10000);
    struct ggml_tensor * emb = sdxl_linear(ctx, u.time_embed_0_w, u.time_embed_0_b, t_emb);
    emb = ggml_silu(ctx, emb);
    emb = sdxl_linear(ctx, u.time_embed_2_w, u.time_embed_2_b, emb);

    struct ggml_tensor * label = sdxl_linear(ctx, u.label_emb_0_w, u.label_emb_0_b, y_t);
    label = ggml_silu(ctx, label);
    label = sdxl_linear(ctx, u.label_emb_2_w, u.label_emb_2_b, label);
    emb = ggml_add(ctx, emb, label);

    struct ggml_tensor * h = sdxl_conv2d(ctx, u.conv_in_w, u.conv_in_b, z, 1, 1, 1, 1, 1, 1);
    std::vector<struct ggml_tensor *> hs;
    hs.push_back(h);
    for (const unet_down_slot & slot : u.down) {
        if (slot.is_downsample) {
            h = sdxl_conv2d(ctx, slot.down_w, slot.down_b, h, 2, 2, 1, 1, 1, 1);
        } else {
            h = resblock_fwd(ctx, slot.res, h, emb);
            if (slot.has_attn) h = spatial_transformer_fwd(ctx, slot.attn, h, context_t);
        }
        hs.push_back(h);
    }

    h = resblock_fwd(ctx, u.mid_res1, h, emb);
    h = spatial_transformer_fwd(ctx, u.mid_attn, h, context_t);
    h = resblock_fwd(ctx, u.mid_res2, h, emb);

    for (const unet_up_slot & slot : u.up) {
        struct ggml_tensor * h_skip = hs.back();
        hs.pop_back();
        h = ggml_concat(ctx, h, h_skip, 2); // channel dim
        h = resblock_fwd(ctx, slot.res, h, emb);
        if (slot.has_attn) h = spatial_transformer_fwd(ctx, slot.attn, h, context_t);
        if (slot.has_upsample) {
            h = ggml_upscale(ctx, h, 2, GGML_SCALE_MODE_NEAREST);
            h = sdxl_conv2d(ctx, slot.up_w, slot.up_b, h, 1, 1, 1, 1, 1, 1);
        }
    }

    h = sdxl_group_norm(ctx, h, u.out_norm_w, u.out_norm_b);
    h = ggml_silu(ctx, h);
    h = sdxl_conv2d(ctx, u.out_conv_w, u.out_conv_b, h, 1, 1, 1, 1, 1, 1); // [w,h,4,1]

    ggml_build_forward_expand(gf, h);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "error: U-Net graph allocation failed\n");
        exit(1);
    }

    ggml_backend_tensor_set(z, latent.data(), 0, latent.size() * sizeof(float));
    ggml_backend_tensor_set(t_scalar, &timestep, 0, sizeof(float));
    ggml_backend_tensor_set(context_t, context.data(), 0, context.size() * sizeof(float));
    ggml_backend_tensor_set(y_t, y.data(), 0, y.size() * sizeof(float));

    ggml_backend_graph_compute(backend, gf);

    std::vector<float> result(ggml_nelements(h));
    ggml_backend_tensor_get(h, result.data(), 0, result.size() * sizeof(float));

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    return result;
}
