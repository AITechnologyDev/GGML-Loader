// Shared ggml graph-building helpers for the SDXL VAE decoder, CLIP text encoders, and U-Net --
// factored out because all three must apply the exact same GroupNorm/LayerNorm/Linear/Conv2d
// conventions (eps, affine application order, the Q8_0-cast workaround) against the same gguf's
// weights, and letting them drift apart between files would be a correctness risk, not just
// duplication.
#pragma once

#include "ggml.h"

#include <string>
#include <vector>

// Looks up a tensor by name, exiting with an error naming it if missing -- shared across the VAE,
// CLIP, and U-Net loaders so a missing/renamed tensor always fails the same clear way.
struct ggml_tensor * sdxl_must_get(struct ggml_context * ctx, const std::string & name);

// stable-diffusion.cpp's gguf quantizes even tiny 1D norm/bias/vector tensors (unlike llama.cpp
// style gguf, which keeps norms in F32) -- ggml_mul/ggml_add can't mix F32 with a quantized operand
// on the CPU backend, so anything used in an elementwise op needs this first. Cheap no-op when
// already F32.
struct ggml_tensor * sdxl_to_f32(struct ggml_context * ctx, struct ggml_tensor * t);

// Standard nn.Linear: x [in_features, ..., N] -> [out_features, ..., N]. w: [in,out] (ggml's own
// ggml_mul_mat(w,x) convention, matching this gguf's stored Linear weight layout directly). b may
// be nullptr (bias-free layers, e.g. CLIP/U-Net attention's to_q/to_k/to_v).
struct ggml_tensor * sdxl_linear(struct ggml_context * ctx, struct ggml_tensor * w,
                                  struct ggml_tensor * b, struct ggml_tensor * x);

// x: [C, seq, N] (or any shape with C as ne0). Standard pre-affine LayerNorm over ne0, eps=1e-5
// (CLIP/U-Net-transformer convention throughout this checkpoint).
struct ggml_tensor * sdxl_layer_norm(struct ggml_context * ctx, struct ggml_tensor * x,
                                      struct ggml_tensor * w, struct ggml_tensor * b,
                                      float eps = 1e-5f);

// x: [W, H, C, N]. 32-group GroupNorm (this checkpoint's fixed convention throughout: VAE, U-Net
// ResBlocks/SpatialTransformer norms), eps=1e-6, with affine w/b of shape [C].
struct ggml_tensor * sdxl_group_norm(struct ggml_context * ctx, struct ggml_tensor * x,
                                      struct ggml_tensor * w, struct ggml_tensor * b,
                                      int n_groups = 32, float eps = 1e-6f);

// s0/s1 stride, p0/p1 padding, d0/d1 dilation -- matches ggml_conv_2d's own parameter order.
struct ggml_tensor * sdxl_conv2d(struct ggml_context * ctx, struct ggml_tensor * w,
                                  struct ggml_tensor * b, struct ggml_tensor * x,
                                  int s0, int s1, int p0, int p1, int d0, int d1);

// Standard multi-head attention, shared by CLIP's self-attention and the U-Net's self- and
// cross-attention (SpatialTransformer's attn1/attn2): q [n_head*d_head, n_q, N], k/v
// [n_head*d_head, n_kv, N] -- n_kv may differ from n_q for cross-attention (k/v come from the CLIP
// text context, not x). mask is optional: additive, [n_kv, n_q], broadcasts over heads/batch (used
// for CLIP's causal mask; U-Net attention is unmasked, pass nullptr). Returns [n_head*d_head, n_q, N].
struct ggml_tensor * sdxl_attention(struct ggml_context * ctx, struct ggml_tensor * q,
                                     struct ggml_tensor * k, struct ggml_tensor * v,
                                     int n_head, struct ggml_tensor * mask);

// pixels: ggml-native [w,h,3,1] (w fastest, c slowest), raw VAE output range (~[-1,1]). Rescales to
// [0,255] and transposes to stb_image_write's row-major interleaved HWC before writing. Returns
// false (and prints an error) on write failure.
bool sdxl_write_png(const std::string & path, const std::vector<float> & pixels, int width, int height);

// sigma at t=999 under the standard SD/SDXL noise schedule (beta_start=0.00085, beta_end=0.012,
// scaled-linear, 1000 steps) -- see cmd_txt2img.cpp's fuller derivation comment. Shared so a
// staged unet-denoise process uses exactly the same constant as the monolithic txt2img path.
float sdxl_sigma_max();

// SDXL's 2816-dim micro-conditioning vector: pooled text embedding [1280] + sinusoidal embeddings
// of (original_size, crop_top_left, target_size). We don't support cropping/aspect tricks, so crop
// is always (0,0) and original_size==target_size==the actual output size (see
// cmd_txt2img.cpp's fuller comment).
std::vector<float> sdxl_micro_conditioning(const std::vector<float> & pooled, int height, int width);

// n/mean/min/max/nan diagnostic print, shared across every SDXL subcommand's debug/verification output.
void sdxl_print_stats(const char * name, const std::vector<float> & v);

// Serializes text conditioning (context[2048*77] + pooled[1280]) to a small self-describing binary
// file (two int64 sizes, then the raw floats), so `clip-encode` can run as its own process and hand
// off to a separate `unet-denoise` process without both needing CLIP+UNet loaded simultaneously at
// once (see load_weights_filtered() in model.h). Returns false (and prints an error) on failure.
bool sdxl_save_condition(const std::string & path, const std::vector<float> & context,
                          const std::vector<float> & pooled);
bool sdxl_load_condition(const std::string & path, std::vector<float> & context,
                          std::vector<float> & pooled);
