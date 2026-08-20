// `ggml-loader unet-denoise <model.gguf> --load-context path [--width W] [--height H] [--seed N]
//   [-o out.bin] [--backend cpu|vulkan]`: the middle stage of the staged encoder/UNet/decoder
// pipeline -- reads a `clip-encode --dump-context` file, runs the single-step SDXL Turbo Euler
// sampler (same math as cmd_txt2img.cpp), and writes a VAE-scaled latent consumable directly by
// `vae-decode --load-latent`. Loads only the U-Net's own tensors (see load_weights_filtered), so
// this process's peak memory never includes CLIP's or the VAE's weights -- the whole point of
// running each stage as its own process: same total FLOPs as the monolithic txt2img, but a much
// lower peak, which is what actually avoids OOM at 512x512 on this device.
#include "commands.h"
#include "model.h"
#include "sdxl_unet.h"
#include "sdxl_common.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s <model.gguf> --load-context path [--width W] [--height H] [--seed N]\n"
        "                 [-o out.bin] [--backend cpu|vulkan]\n"
        "  --load-context is required: a file from `clip-encode --dump-context`.\n"
        "  W/H must be multiples of 8 (default 512x512). Writes a raw float32 VAE-scaled latent to\n"
        "  -o (default unet_latent.bin), directly loadable by `vae-decode --load-latent`.\n",
        argv0);
}

int cmd_unet_denoise(int argc, char ** argv) {
    std::string model_path, load_context_path, out_path = "unet_latent.bin", backend_name = "cpu";
    int width = 512, height = 512;
    uint32_t seed = 0;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--load-context" && i + 1 < argc) {
            load_context_path = argv[++i];
        } else if (a == "--width" && i + 1 < argc) {
            width = atoi(argv[++i]);
        } else if (a == "--height" && i + 1 < argc) {
            height = atoi(argv[++i]);
        } else if ((a == "-o" || a == "--output") && i + 1 < argc) {
            out_path = argv[++i];
        } else if (a == "--seed" && i + 1 < argc) {
            seed = (uint32_t) strtoul(argv[++i], nullptr, 10);
        } else if (a == "--backend" && i + 1 < argc) {
            backend_name = argv[++i];
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (!a.empty() && a[0] == '-') {
            fprintf(stderr, "error: unrecognized flag '%s'\n", a.c_str());
            print_usage(argv[0]);
            return 1;
        } else if (model_path.empty()) {
            model_path = a;
        } else {
            fprintf(stderr, "error: unexpected extra argument '%s'\n", a.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }
    if (model_path.empty() || load_context_path.empty() || width % 8 != 0 || height % 8 != 0) {
        print_usage(argv[0]);
        return 1;
    }

    std::vector<float> context, pooled;
    if (!sdxl_load_condition(load_context_path, context, pooled)) {
        return 1;
    }
    int context_len = (int) (context.size() / 2048);

    ggml_backend_t backend = init_backend(backend_name, 8);

    struct gguf_context * gctx = nullptr;
    // filtered to just the U-Net's tensors -- see file header comment.
    weights_store ws = load_weights_filtered(model_path.c_str(), backend, {"model.diffusion_model."}, &gctx);
    sdxl_unet unet = load_sdxl_unet(ws.ctx);
    fprintf(stderr, "U-Net loaded (backend=%s), context_len=%d\n", backend_name.c_str(), context_len);

    std::vector<float> y = sdxl_micro_conditioning(pooled, height, width);

    int64_t latent_w = width / 8, latent_h = height / 8;
    size_t n_latent = (size_t)(latent_w * latent_h * 4);
    std::mt19937 rng(seed != 0 ? seed : std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> noise(n_latent);
    for (float & v : noise) v = dist(rng);

    float sigma_max = sdxl_sigma_max();
    float c_in = 1.0f / sqrtf(sigma_max * sigma_max + 1.0f); // sigma_data=1

    std::vector<float> x_noisy(n_latent), model_input(n_latent);
    for (size_t i = 0; i < n_latent; i++) {
        x_noisy[i] = noise[i] * sigma_max; // CompVisDenoiser::noise_scaling, from a zero (txt2img) latent
        model_input[i] = x_noisy[i] * c_in;
    }

    sdxl_print_stats("x_noisy", x_noisy);
    sdxl_print_stats("model_input", model_input);
    fprintf(stderr, "running U-Net (sigma_max=%.3f, c_in=%.5f)...\n", sigma_max, c_in);
    std::vector<float> eps_pred = unet_forward(unet, backend, model_input, latent_w, latent_h,
                                                999.0f, context, context_len, y);
    sdxl_print_stats("eps_pred", eps_pred);

    std::vector<float> denoised(n_latent);
    for (size_t i = 0; i < n_latent; i++) {
        denoised[i] = x_noisy[i] - sigma_max * eps_pred[i]; // c_skip=1, c_out=-sigma_max
    }
    sdxl_print_stats("denoised", denoised);

    const float SDXL_VAE_SCALING_FACTOR = 0.13025f; // see cmd_txt2img.cpp's fuller comment
    std::vector<float> latent_for_vae(n_latent);
    for (size_t i = 0; i < n_latent; i++) {
        latent_for_vae[i] = denoised[i] / SDXL_VAE_SCALING_FACTOR;
    }

    FILE * f = fopen(out_path.c_str(), "wb");
    if (!f || fwrite(latent_for_vae.data(), sizeof(float), latent_for_vae.size(), f) != latent_for_vae.size()) {
        fprintf(stderr, "error: failed to write '%s'\n", out_path.c_str());
        return 1;
    }
    fclose(f);
    fprintf(stderr, "wrote VAE-scaled latent (%zu floats, %lldx%lld) to %s\n",
            latent_for_vae.size(), (long long) latent_w, (long long) latent_h, out_path.c_str());

    ggml_backend_buffer_free(ws.buffer);
    ggml_free(ws.ctx);
    ggml_backend_free(backend);
    gguf_free(gctx);
    return 0;
}
