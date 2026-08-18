// `ggml-loader clip-encode <model.gguf> "prompt" [--backend cpu|vulkan]`: tokenizes a prompt,
// runs it through SDXL's dual CLIP text encoder, and prints stats (mean/min/max, NaN/Inf check) for
// the resulting cross-attention context [2048,77] and bigG's pooled+projected embedding [1280].
// Verification-only stage before wiring CLIP into the U-Net -- a real encoder should produce
// finite, plausibly-scaled (not all-zero, not exploding) values; it can't be checked against a
// reference image without the U-Net+sampler existing yet, so this is the strongest test available
// at this stage: no crash, no NaN, reasonable magnitude.
#include "commands.h"
#include "clip_tokenizer.h"
#include "clip_text.h"
#include "model.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void print_usage(const char * argv0) {
    fprintf(stderr, "usage: %s <model.gguf> \"prompt\" [--backend cpu|vulkan]\n", argv0);
}

static void print_stats(const char * name, const std::vector<float> & v) {
    float vmin = v[0], vmax = v[0], sum = 0.0f;
    int n_nan = 0, n_inf = 0;
    for (float x : v) {
        if (std::isnan(x)) { n_nan++; continue; }
        if (std::isinf(x)) { n_inf++; continue; }
        vmin = std::min(vmin, x);
        vmax = std::max(vmax, x);
        sum += x;
    }
    fprintf(stderr, "%s: n=%zu mean=%.4f min=%.4f max=%.4f nan=%d inf=%d\n",
            name, v.size(), sum / v.size(), vmin, vmax, n_nan, n_inf);
}

int cmd_clip_encode(int argc, char ** argv) {
    std::string model_path, prompt;
    std::string backend_name = "cpu";

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--backend" && i + 1 < argc) {
            backend_name = argv[++i];
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (model_path.empty()) {
            model_path = a;
        } else if (prompt.empty()) {
            prompt = a;
        }
    }
    if (model_path.empty() || prompt.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    ggml_backend_t backend = init_backend(backend_name, 8);

    struct gguf_context * gctx = nullptr;
    weights_store ws = load_weights(model_path.c_str(), backend, &gctx);

    clip_text_model clip_l = load_clip_text(ws.ctx, "cond_stage_model.transformer.text_model.",
                                             /*n_layer=*/12, /*hidden=*/768, /*n_head=*/12,
                                             /*has_text_projection=*/false, /*gelu_tanh=*/false);
    clip_text_model clip_g = load_clip_text(ws.ctx, "cond_stage_model.1.transformer.text_model.",
                                             /*n_layer=*/32, /*hidden=*/1280, /*n_head=*/20,
                                             /*has_text_projection=*/true, /*gelu_tanh=*/true);
    fprintf(stderr, "CLIP-L + CLIP-bigG loaded (backend=%s)\n", backend_name.c_str());

    clip_vocab vocab = load_clip_vocab();
    std::vector<int32_t> ids = vocab.tokenize(prompt);
    fprintf(stderr, "tokenized '%s' -> %zu ids (first 12: ", prompt.c_str(), ids.size());
    for (int i = 0; i < 12 && i < (int) ids.size(); i++) fprintf(stderr, "%d ", ids[i]);
    fprintf(stderr, ")\n");

    sdxl_text_condition cond = clip_encode(clip_l, clip_g, backend, ids);
    print_stats("context[2048,77]", cond.context);
    print_stats("pooled[1280]", cond.pooled);

    ggml_backend_buffer_free(ws.buffer);
    ggml_free(ws.ctx);
    ggml_backend_free(backend);
    gguf_free(gctx);
    return 0;
}
