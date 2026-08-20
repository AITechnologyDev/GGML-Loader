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
#include "sdxl_common.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s <model.gguf> \"prompt\" [--backend cpu|vulkan] [--dump-context path]\n"
        "  --dump-context writes context[2048,77]+pooled[1280] to a file for a separate\n"
        "  `unet-denoise` process to consume (see sdxl_save_condition) -- lets CLIP's weights be\n"
        "  freed before the U-Net's are loaded, part of the staged encoder/UNet/decoder pipeline.\n",
        argv0);
}

int cmd_clip_encode(int argc, char ** argv) {
    std::string model_path, prompt;
    std::string backend_name = "cpu";
    std::string dump_context_path;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if ((a == "-p" || a == "--prompt") && i + 1 < argc) {
            prompt = argv[++i];
        } else if (a == "--backend" && i + 1 < argc) {
            backend_name = argv[++i];
        } else if (a == "--dump-context" && i + 1 < argc) {
            dump_context_path = argv[++i];
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (!a.empty() && a[0] == '-') {
            fprintf(stderr, "error: unrecognized flag '%s'\n", a.c_str());
            print_usage(argv[0]);
            return 1;
        } else if (model_path.empty()) {
            model_path = a;
        } else if (prompt.empty()) {
            prompt = a;
        } else {
            fprintf(stderr, "error: unexpected extra argument '%s'\n", a.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }
    if (model_path.empty() || prompt.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    ggml_backend_t backend = init_backend(backend_name, 8);

    struct gguf_context * gctx = nullptr;
    // filtered to just the two CLIP text-tower's tensors ("cond_stage_model." and
    // "cond_stage_model.1.") -- part of the staged pipeline, see --dump-context above.
    weights_store ws = load_weights_filtered(model_path.c_str(), backend, {"cond_stage_model."}, &gctx);

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
    sdxl_print_stats("context[2048,77]", cond.context);
    sdxl_print_stats("pooled[1280]", cond.pooled);

    if (!dump_context_path.empty()) {
        if (!sdxl_save_condition(dump_context_path, cond.context, cond.pooled)) {
            return 1;
        }
        fprintf(stderr, "dumped text conditioning to %s\n", dump_context_path.c_str());
    }

    ggml_backend_buffer_free(ws.buffer);
    ggml_free(ws.ctx);
    ggml_backend_free(backend);
    gguf_free(gctx);
    return 0;
}
