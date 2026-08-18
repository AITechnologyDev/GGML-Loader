// `ggml-loader vae-decode <model.gguf> [--width W] [--height H] [-o out.png] [--seed N]
//   [--backend cpu|vulkan]`: decodes a synthetic (random-noise) latent through the SDXL/SD VAE
// decoder and writes a PNG. This is a smoke test, not real image generation -- there's no
// encoder, U-Net, or diffusion sampler yet (see sdxl_vae.h and the README), so the output is not
// a meaningful picture. It exists to validate the new conv/GroupNorm/spatial-attention graph
// machinery end to end (loads real SDXL weights, runs without crashing, produces plausible pixel
// values) before any of that other, much larger work starts.
#include "commands.h"
#include "model.h"
#include "sdxl_vae.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../third_party/ggml/examples/stb_image_write.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s <model.gguf> [--width W] [--height H] [-o out.png] [--seed N] [--backend cpu|vulkan]\n"
        "  W/H must be multiples of 8 (default 512x512). Decodes random noise -- not a real image;\n"
        "  see the file header comment for what this does and doesn't prove.\n", argv0);
}

int cmd_vae_decode(int argc, char ** argv) {
    std::string model_path;
    std::string out_path = "vae_test.png";
    std::string backend_name = "cpu";
    int width = 512, height = 512;
    uint32_t seed = 0;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--width" && i + 1 < argc) {
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
        } else if (model_path.empty()) {
            model_path = a;
        }
    }

    if (model_path.empty() || width % 8 != 0 || height % 8 != 0) {
        print_usage(argv[0]);
        return 1;
    }

    ggml_backend_t backend = init_backend(backend_name, 8);

    struct gguf_context * gctx = nullptr;
    weights_store ws = load_weights(model_path.c_str(), backend, &gctx);
    vae_decoder vae = load_vae_decoder(ws.ctx);
    fprintf(stderr, "VAE decoder loaded (backend=%s)\n", backend_name.c_str());

    int64_t latent_w = width / 8, latent_h = height / 8;
    std::vector<float> latent((size_t)(latent_w * latent_h * 4));
    std::mt19937 rng(seed != 0 ? seed : std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (float & v : latent) v = dist(rng);

    fprintf(stderr, "decoding %lldx%lld latent -> %dx%d image...\n",
            (long long) latent_w, (long long) latent_h, width, height);
    std::vector<float> pixels = vae_decode(vae, backend, latent, latent_w, latent_h);
    fprintf(stderr, "decode done (%zu floats)\n", pixels.size());

    // ggml-native [w,h,3,1] (w fastest, c slowest) -> stb's row-major interleaved HWC uint8,
    // rescaling the model's raw ~[-1,1] output to [0,255]
    std::vector<uint8_t> rgb((size_t)(width * height * 3));
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

    if (!stbi_write_png(out_path.c_str(), width, height, 3, rgb.data(), width * 3)) {
        fprintf(stderr, "error: failed to write '%s'\n", out_path.c_str());
        return 1;
    }
    fprintf(stderr, "wrote %s\n", out_path.c_str());

    ggml_backend_buffer_free(ws.buffer);
    ggml_free(ws.ctx);
    ggml_backend_free(backend);
    gguf_free(gctx);
    return 0;
}
