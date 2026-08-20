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
    // NOT ggml_conv_2d(): its own im2col step always downcasts the *activations* (not just the
    // weight) to F16 unless the weight happens to be BF16 -- none of ours are, they're F16. The
    // SDXL VAE decoder's intermediate activations routinely exceed F16's ~65504 max (confirmed via
    // debug tracing on a real generated latent: max grew past 100000 by the last upsample stage),
    // silently overflowing to inf then NaN. This is the well-known "SDXL VAE needs fp32" issue from
    // the wider SD community, just hit inside our own conv rather than a framework flag to flip --
    // fixed by reimplementing ggml_conv_2d's own body with the im2col destination forced to F32.
    // ggml_mul_mat(ctx, a, b) requires b (src1) to already be F32 -- a (src0) is the operand allowed
    // to be quantized/F16 (same convention sdxl_linear already relies on: weight first, F32
    // activation second). So the weight goes first here, im2col (forced F32 above) second --
    // opposite of ggml_conv_2d's own internal order, which puts im2col first because ITS im2col
    // stays F16 there.
    struct ggml_tensor * im2col = ggml_im2col(ctx, w, x, s0, s1, p0, p1, d0, d1, true, GGML_TYPE_F32);
    struct ggml_tensor * result = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, w, w->ne[0] * w->ne[1] * w->ne[2], w->ne[3]),
        ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[3] * im2col->ne[2] * im2col->ne[1]));
    // result: [OC, OW*OH*N] -> [OC,OW,OH,N] -> permute -> [OW,OH,OC,N]
    result = ggml_reshape_4d(ctx, result, w->ne[3], im2col->ne[1], im2col->ne[2], im2col->ne[3]);
    result = ggml_cont(ctx, ggml_permute(ctx, result, 2, 0, 1, 3));

    struct ggml_tensor * b4 = ggml_reshape_4d(ctx, sdxl_to_f32(ctx, b), 1, 1, b->ne[0], 1);
    result = ggml_add(ctx, result, b4);
    return result;
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

float sdxl_sigma_max() {
    const double beta_start = 0.00085, beta_end = 0.012;
    const int timesteps = 1000;
    double alphas_cumprod = 1.0;
    for (int i = 0; i < timesteps; i++) {
        double sb = sqrt(beta_start) + (sqrt(beta_end) - sqrt(beta_start)) * ((double) i / (timesteps - 1));
        double beta = sb * sb;
        alphas_cumprod *= (1.0 - beta);
    }
    return (float) sqrt((1.0 - alphas_cumprod) / alphas_cumprod);
}

// matches ggml_timestep_embedding's own CPU kernel exactly (cos first, sin second, half=dim/2) --
// computed on the host since it's only ever applied to the 6 scalar size/crop/target values, not
// worth a graph round-trip for.
static std::vector<float> sinusoidal_embed(const std::vector<float> & timesteps, int dim, int max_period = 10000) {
    int half = dim / 2;
    std::vector<float> out(timesteps.size() * dim);
    for (size_t i = 0; i < timesteps.size(); i++) {
        for (int j = 0; j < half; j++) {
            float freq = expf(-logf((float) max_period) * (float) j / (float) half);
            float arg = timesteps[i] * freq;
            out[i * dim + j] = cosf(arg);
            out[i * dim + j + half] = sinf(arg);
        }
    }
    return out;
}

std::vector<float> sdxl_micro_conditioning(const std::vector<float> & pooled, int height, int width) {
    std::vector<float> vec = pooled;
    auto append = [&](float a, float b) {
        std::vector<float> e = sinusoidal_embed({a, b}, 256);
        vec.insert(vec.end(), e.begin(), e.end());
    };
    append((float) height, (float) width);
    append(0.0f, 0.0f);
    append((float) height, (float) width);
    return vec;
}

void sdxl_print_stats(const char * name, const std::vector<float> & v) {
    float vmin = v[0], vmax = v[0], sum = 0.0f;
    int n_nan = 0;
    for (float x : v) {
        if (std::isnan(x)) { n_nan++; continue; }
        vmin = std::min(vmin, x);
        vmax = std::max(vmax, x);
        sum += x;
    }
    fprintf(stderr, "%s: n=%zu mean=%.4f min=%.4f max=%.4f nan=%d\n",
            name, v.size(), sum / v.size(), vmin, vmax, n_nan);
}

bool sdxl_save_condition(const std::string & path, const std::vector<float> & context,
                          const std::vector<float> & pooled) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "error: failed to open '%s' for writing\n", path.c_str());
        return false;
    }
    int64_t n_context = (int64_t) context.size(), n_pooled = (int64_t) pooled.size();
    bool ok = fwrite(&n_context, sizeof(n_context), 1, f) == 1 &&
              fwrite(&n_pooled, sizeof(n_pooled), 1, f) == 1 &&
              fwrite(context.data(), sizeof(float), context.size(), f) == context.size() &&
              fwrite(pooled.data(), sizeof(float), pooled.size(), f) == pooled.size();
    fclose(f);
    if (!ok) {
        fprintf(stderr, "error: failed to write condition data to '%s'\n", path.c_str());
    }
    return ok;
}

bool sdxl_load_condition(const std::string & path, std::vector<float> & context,
                          std::vector<float> & pooled) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "error: failed to open '%s' for reading\n", path.c_str());
        return false;
    }
    int64_t n_context = 0, n_pooled = 0;
    bool ok = fread(&n_context, sizeof(n_context), 1, f) == 1 &&
              fread(&n_pooled, sizeof(n_pooled), 1, f) == 1;
    if (ok) {
        context.resize((size_t) n_context);
        pooled.resize((size_t) n_pooled);
        ok = fread(context.data(), sizeof(float), context.size(), f) == context.size() &&
             fread(pooled.data(), sizeof(float), pooled.size(), f) == pooled.size();
    }
    fclose(f);
    if (!ok) {
        fprintf(stderr, "error: failed to read condition data from '%s'\n", path.c_str());
    }
    return ok;
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
