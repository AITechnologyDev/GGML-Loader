// `ggml-loader txt2img <model.gguf> "prompt" [--width W] [--height H] [-o out.png] [--seed N]
//   [--backend cpu|vulkan]`: SDXL Turbo text-to-image, CPU-first. Single-step (SDXL Turbo's native
// mode -- see the sigma_max/no-CFG math below), 512x512 by default. Wires together everything built
// this session: clip_tokenizer -> clip_text (dual CLIP encoder) -> sdxl_unet (single denoising
// step) -> sdxl_vae (decoder, already existed).
//
// One command, but internally staged like the standalone clip-encode/unet-denoise/vae-decode
// subcommands: each of the three loads its own tensors via load_weights_filtered() and frees them
// (ggml_backend_buffer_free + ggml_free(ctx)) before the next stage loads, instead of one
// load_weights() call holding CLIP+UNet+VAE resident for the whole run. Freed backend memory is
// reusable by the next stage's allocation within the same process (glibc/bionic malloc keeps freed
// heap chunks around rather than always returning them to the OS), so this gets most of the same
// peak-memory win as running three separate processes -- without making the user drive three
// commands and shuttle intermediate files by hand. Verified this keeps 512x512 from OOMing on this
// device where the old all-at-once load routinely did.
//
// The 1-step sampler math (verified against reference/stable-diffusion.cpp's runtime/denoiser.hpp,
// not guessed): SDXL uses eps-prediction (CompVisDenoiser) with the standard scaled-linear beta
// schedule (beta 0.00085->0.012 over 1000 steps). A single Euler step from sigma_max (t=999, pure
// noise) to sigma=0 reduces to a closed form -- see build_sigma_max()'s comment -- so there's no
// need for a general multi-step ODE solver here; that's future work once step count > 1 is wanted.
#include "commands.h"
#include "model.h"
#include "clip_tokenizer.h"
#include "clip_text.h"
#include "sdxl_unet.h"
#include "sdxl_vae.h"
#include "sdxl_common.h"

#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

static void print_usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s <model.gguf> \"prompt\" [--width W] [--height H] [-o out.png] [--seed N]\n"
        "                 [--backend cpu|vulkan]\n"
        "  \"prompt\" may also be given as -p/--prompt \"text\".\n"
        "  W/H must be multiples of 8 (default 512x512). Single-step SDXL Turbo, no negative prompt.\n",
        argv0);
}

// sigma_max/micro-conditioning/print_stats now live in sdxl_common.h/.cpp, shared with the staged
// `unet-denoise` subcommand (cmd_unet.cpp) so both paths use exactly the same sampler math. See
// sdxl_common.cpp for the sigma_max derivation and micro-conditioning layout comments -- the 1-step
// Euler sampler itself (predict eps once at sigma_max, denoised = x_noisy - sigma_max*eps_pred, no
// iterative solver needed since sigma_next==0 for n_steps=1) is still specific to this file.

int cmd_txt2img(int argc, char ** argv) {
    std::string model_path, prompt, out_path = "txt2img.png", backend_name = "cpu";
    std::string dump_latent_path;
    int width = 512, height = 512;
    uint32_t seed = 0;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if ((a == "-p" || a == "--prompt") && i + 1 < argc) {
            prompt = argv[++i];
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
        } else if (a == "--dump-latent" && i + 1 < argc) {
            dump_latent_path = argv[++i];
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (!a.empty() && a[0] == '-') {
            // reject rather than silently swallow -- an unrecognized flag falling through to the
            // positional branches below would silently become the prompt (this is exactly how a
            // real bug was found: `-p "text"` with -p unsupported here made "-p" itself the prompt
            // and dropped "text" entirely, with no error to signal it)
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
    if (model_path.empty() || prompt.empty() || width % 8 != 0 || height % 8 != 0) {
        print_usage(argv[0]);
        return 1;
    }

    ggml_backend_t backend = init_backend(backend_name, 8);
    int64_t latent_w = width / 8, latent_h = height / 8;
    size_t n_latent = (size_t)(latent_w * latent_h * 4);

    // -- stage 1: CLIP text encoder --------------------------------------------------------
    sdxl_text_condition cond;
    int n_tok;
    {
        struct gguf_context * gctx = nullptr;
        weights_store ws = load_weights_filtered(model_path.c_str(), backend, {"cond_stage_model."}, &gctx);
        clip_text_model clip_l = load_clip_text(ws.ctx, "cond_stage_model.transformer.text_model.",
                                                 12, 768, 12, false, false);
        clip_text_model clip_g = load_clip_text(ws.ctx, "cond_stage_model.1.transformer.text_model.",
                                                 32, 1280, 20, true, true);
        fprintf(stderr, "CLIP loaded (backend=%s)\n", backend_name.c_str());

        clip_vocab vocab = load_clip_vocab();
        std::vector<int32_t> ids = vocab.tokenize(prompt);
        n_tok = (int) ids.size();
        cond = clip_encode(clip_l, clip_g, backend, ids);
        fprintf(stderr, "text conditioning ready (context %zu floats)\n", cond.context.size());

        ggml_backend_buffer_free(ws.buffer);
        ggml_free(ws.ctx);
        gguf_free(gctx);
    }
    std::vector<float> y = sdxl_micro_conditioning(cond.pooled, height, width);

    // -- stage 2: U-Net denoiser (single Euler step) ----------------------------------------
    std::vector<float> latent_for_vae(n_latent);
    {
        struct gguf_context * gctx = nullptr;
        weights_store ws = load_weights_filtered(model_path.c_str(), backend, {"model.diffusion_model."}, &gctx);
        sdxl_unet unet = load_sdxl_unet(ws.ctx);
        fprintf(stderr, "U-Net loaded (backend=%s)\n", backend_name.c_str());

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
                                                    999.0f, cond.context, n_tok, y);
        sdxl_print_stats("eps_pred", eps_pred);

        std::vector<float> denoised(n_latent);
        for (size_t i = 0; i < n_latent; i++) {
            denoised[i] = x_noisy[i] - sigma_max * eps_pred[i]; // c_skip=1, c_out=-sigma_max
        }
        sdxl_print_stats("denoised", denoised);

        // SDXL's VAE was trained/calibrated on latents ~1/0.13025 larger than the diffusion U-Net's
        // own latent scale (diffusers: `latents = latents / vae.config.scaling_factor` before decode)
        // -- skipping this produced solid-black/saturated output during testing.
        const float SDXL_VAE_SCALING_FACTOR = 0.13025f;
        for (size_t i = 0; i < n_latent; i++) {
            latent_for_vae[i] = denoised[i] / SDXL_VAE_SCALING_FACTOR;
        }

        // Dumped AFTER VAE-scaling, not before: `vae-decode --load-latent` feeds a loaded file
        // straight into vae_decode() with no further scaling -- dumping the pre-scale `denoised`
        // here would silently hand the decoder a latent ~7.68x too large (an out-of-distribution
        // input, not a real VAE bug, even though it once also exposed a real separate
        // F16-im2col-overflow issue in sdxl_conv2d).
        if (!dump_latent_path.empty()) {
            FILE * f = fopen(dump_latent_path.c_str(), "wb");
            fwrite(latent_for_vae.data(), sizeof(float), latent_for_vae.size(), f);
            fclose(f);
            fprintf(stderr, "dumped VAE-scaled latent (%zu floats, %lldx%lld) to %s\n",
                    latent_for_vae.size(), (long long) latent_w, (long long) latent_h, dump_latent_path.c_str());
        }

        ggml_backend_buffer_free(ws.buffer);
        ggml_free(ws.ctx);
        gguf_free(gctx);
    }

    // -- stage 3: VAE decoder ---------------------------------------------------------------
    std::vector<float> pixels;
    {
        struct gguf_context * gctx = nullptr;
        weights_store ws = load_weights_filtered(model_path.c_str(), backend, {"first_stage_model.decoder."}, &gctx);
        vae_decoder vae = load_vae_decoder(ws.ctx);
        fprintf(stderr, "VAE loaded (backend=%s)\n", backend_name.c_str());

        fprintf(stderr, "decoding VAE...\n");
        pixels = vae_decode(vae, backend, latent_for_vae, latent_w, latent_h);
        sdxl_print_stats("pixels", pixels);

        ggml_backend_buffer_free(ws.buffer);
        ggml_free(ws.ctx);
        gguf_free(gctx);
    }

    if (!sdxl_write_png(out_path, pixels, width, height)) {
        return 1;
    }
    fprintf(stderr, "wrote %s\n", out_path.c_str());

    ggml_backend_free(backend);
    return 0;
}
