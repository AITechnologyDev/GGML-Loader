// SDXL's dual CLIP text encoder: CLIP-L (ViT-L/14, 768d/12L/12H) and OpenCLIP-bigG (1280d/32L/20H).
// Ported from stable-diffusion.cpp's CLIPTextModel (see reference/stable-diffusion.cpp/src/model/
// te/clip.hpp), cross-checked against our own gguf's actual tensor names via `inspect`. Two things
// that are easy to get subtly wrong and were verified against the reference source directly rather
// than assumed:
//   - Self-attention uses a CAUSAL mask (upper-triangular -inf), same convention as an LLM decoder,
//     even though this isn't autoregressive generation -- that's genuinely how CLIP's text encoder
//     was trained (CLIPTextModelRunner::build_graph in clip.hpp).
//   - The "context" fed to the U-Net's cross-attention stops 2 layers early (clip_skip=2, SDXL's
//     fixed default) and is NOT layer-normed; the "pooled" embedding (bigG only, at the EOS token
//     position, then projected via text_projection) runs the FULL depth plus final_layer_norm,
//     regardless of clip_skip -- these are two different forward passes through the same encoder,
//     not one pass reused for both (get_learned_condition_common in conditioner.hpp).
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <string>
#include <vector>

struct clip_layer {
    struct ggml_tensor * ln1_w = nullptr, * ln1_b = nullptr;
    struct ggml_tensor * ln2_w = nullptr, * ln2_b = nullptr;
    struct ggml_tensor * q_w = nullptr, * q_b = nullptr;
    struct ggml_tensor * k_w = nullptr, * k_b = nullptr;
    struct ggml_tensor * v_w = nullptr, * v_b = nullptr;
    struct ggml_tensor * out_w = nullptr, * out_b = nullptr;
    struct ggml_tensor * fc1_w = nullptr, * fc1_b = nullptr;
    struct ggml_tensor * fc2_w = nullptr, * fc2_b = nullptr;
};

struct clip_text_model {
    struct ggml_tensor * token_embedding    = nullptr; // [hidden, 49408]
    struct ggml_tensor * position_embedding = nullptr; // [hidden, 77]
    std::vector<clip_layer> layers;
    struct ggml_tensor * final_ln_w = nullptr, * final_ln_b = nullptr;
    struct ggml_tensor * text_projection = nullptr; // bigG only: [1280,1280], no bias
    int n_head  = 0;
    bool gelu_tanh = false; // false: gelu_quick (CLIP-L, d_model==768); true: tanh-approx gelu (bigG)
};

// prefix e.g. "cond_stage_model.transformer.text_model." (CLIP-L) or
// "cond_stage_model.1.transformer.text_model." (bigG).
clip_text_model load_clip_text(struct ggml_context * wctx, const std::string & prefix,
                                int n_layer, int hidden, int n_head, bool has_text_projection,
                                bool gelu_tanh);

// One prompt's SDXL text conditioning: the U-Net's cross-attention context [2048,77] (CLIP-L's and
// bigG's clip_skip=2 hidden states, concatenated along the channel dim) and bigG's pooled+projected
// embedding [1280] (see file header for why these are two separate forward passes). `ids` is the
// BOS+content+EOS+EOS-padded 77-token sequence from clip_vocab::tokenize(); this function derives
// bigG's own zero-past-EOS-padded copy internally (a real, deliberate quirk of how this checkpoint
// was converted -- CLIP-L keeps EOS padding, bigG's embedding lookup does not, see clip_text.cpp).
struct sdxl_text_condition {
    std::vector<float> context; // [2048 * 77], ggml-native (2048 fastest)
    std::vector<float> pooled;  // [1280]
};

sdxl_text_condition clip_encode(const clip_text_model & clip_l, const clip_text_model & clip_g,
                                 ggml_backend_t backend, const std::vector<int32_t> & ids);
