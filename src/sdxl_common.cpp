#include "sdxl_common.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/ggml/examples/stb_image_write.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

struct ggml_tensor * sdxl_must_get(struct ggml_context * ctx, const std::string & name) {
    struct ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t) {
        fprintf(stderr, "error: missing SDXL tensor '%s'\n", name.c_str());
        exit(1);
    }
    return t;
}

struct ggml_tensor * sdxl_to_f32(struct ggml_context * ctx, struct ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 ? t : ggml_cast(ctx, t, GGML_TYPE_F32);
}

struct ggml_tensor * sdxl_linear(struct ggml_context * ctx, struct ggml_tensor * w,
                                  struct ggml_tensor * b, struct ggml_tensor * x) {
    struct ggml_tensor * out = ggml_mul_mat(ctx, w, x); // [out_features, ..., N]
    if (b != nullptr) {
        out = ggml_add(ctx, out, sdxl_to_f32(ctx, b)); // b: [out_features], broadcasts over ne1..3
    }
    return out;
}

struct ggml_tensor * sdxl_layer_norm(struct ggml_context * ctx, struct ggml_tensor * x,
                                      struct ggml_tensor * w, struct ggml_tensor * b, float eps) {
    x = ggml_norm(ctx, x, eps);
    x = ggml_mul(ctx, x, sdxl_to_f32(ctx, w)); // w/b: [C] == ne0, broadcasts over ne1..3
    x = ggml_add(ctx, x, sdxl_to_f32(ctx, b));
    return x;
}

struct ggml_tensor * sdxl_group_norm(struct ggml_context * ctx, struct ggml_tensor * x,
                                      struct ggml_tensor * w, struct ggml_tensor * b,
                                      int n_groups, float eps) {
    x = ggml_group_norm(ctx, x, n_groups, eps);
    struct ggml_tensor * w4 = ggml_reshape_4d(ctx, sdxl_to_f32(ctx, w), 1, 1, w->ne[0], 1);
    struct ggml_tensor * b4 = ggml_reshape_4d(ctx, sdxl_to_f32(ctx, b), 1, 1, b->ne[0], 1);
    x = ggml_mul(ctx, x, w4);
    x = ggml_add(ctx, x, b4);
    return x;
}

struct ggml_tensor * sdxl_conv2d(struct ggml_context * ctx, struct ggml_tensor * w,
                                  struct ggml_tensor * b, struct ggml_tensor * x,
                                  int s0, int s1, int p0, int p1, int d0, int d1) {
    x = ggml_conv_2d(ctx, w, x, s0, s1, p0, p1, d0, d1);
    struct ggml_tensor * b4 = ggml_reshape_4d(ctx, sdxl_to_f32(ctx, b), 1, 1, b->ne[0], 1);
    x = ggml_add(ctx, x, b4);
    return x;
}

struct ggml_tensor * sdxl_attention(struct ggml_context * ctx, struct ggml_tensor * q,
                                     struct ggml_tensor * k, struct ggml_tensor * v,
                                     int n_head, struct ggml_tensor * mask) {
    const int64_t inner_dim = q->ne[0];
    const int64_t d_head    = inner_dim / n_head;
    const int64_t n_q       = q->ne[1];
    const int64_t n_kv      = k->ne[1];
    const int64_t N         = q->ne[2];

    struct ggml_tensor * q4 = ggml_reshape_4d(ctx, q, d_head, n_head, n_q, N);
    q4 = ggml_cont(ctx, ggml_permute(ctx, q4, 0, 2, 1, 3)); // [d_head, n_q, n_head, N]

    struct ggml_tensor * k4 = ggml_reshape_4d(ctx, k, d_head, n_head, n_kv, N);
    k4 = ggml_cont(ctx, ggml_permute(ctx, k4, 0, 2, 1, 3)); // [d_head, n_kv, n_head, N]

    struct ggml_tensor * v4 = ggml_reshape_4d(ctx, v, d_head, n_head, n_kv, N);
    v4 = ggml_cont(ctx, ggml_permute(ctx, v4, 0, 2, 1, 3)); // [d_head, n_kv, n_head, N]

    float scale = 1.0f / sqrtf((float) d_head);
    struct ggml_tensor * kq = ggml_mul_mat(ctx, k4, q4); // [n_kv, n_q, n_head, N]
    kq = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);

    struct ggml_tensor * v4t = ggml_cont(ctx, ggml_permute(ctx, v4, 1, 0, 2, 3)); // [n_kv, d_head, n_head, N]
    struct ggml_tensor * kqv = ggml_mul_mat(ctx, v4t, kq); // [d_head, n_q, n_head, N]

    struct ggml_tensor * out = ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)); // [d_head, n_head, n_q, N]
    return ggml_reshape_3d(ctx, out, inner_dim, n_q, N);
}

bool sdxl_write_png(const std::string & path, const std::vector<float> & pixels, int width, int height) {
    std::vector<uint8_t> rgb((size_t) width * height * 3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            for (int c = 0; c < 3; c++) {
                size_t src = (size_t) x + (size_t) y * width + (size_t) c * width * height;
                float v = (pixels[src] * 0.5f + 0.5f) * 255.0f;
                v = std::min(255.0f, std::max(0.0f, v));
                rgb[((size_t) y * width + x) * 3 + c] = (uint8_t) v;
            }
        }
    }
    if (!stbi_write_png(path.c_str(), width, height, 3, rgb.data(), width * 3)) {
        fprintf(stderr, "error: failed to write '%s'\n", path.c_str());
        return false;
    }
    return true;
}
