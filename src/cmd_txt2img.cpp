// `ggml-loader txt2img <model.gguf> "prompt" [--width W] [--height H] [-o out.png] [--seed N]
//   [--backend cpu|vulkan]`: SDXL Turbo text-to-image, CPU-first. Single-step (SDXL Turbo's native
// mode -- see the sigma_max/no-CFG math below), 512x512 by default. Wires together everything built
// this session: clip_tokenizer -> clip_text (dual CLIP encoder) -> sdxl_unet (single denoising
// step) -> sdxl_vae (decoder, already existed).
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
        "  W/H must be multiples of 8 (default 512x512). Single-step SDXL Turbo, no negative prompt.\n",
        argv0);
}

// sin-then-... matches ggml_timestep_embedding's own CPU kernel exactly (cos first, sin second,
// half=dim/2) -- computed on the host here since it's only ever applied to the 6 scalar
// size/crop/target values, not worth a graph round-trip for.
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

// SDXL's 2816-dim micro-conditioning vector: pooled text embedding + sinusoidal embeddings of
// (original_size, crop_top_left, target_size) -- we don't support cropping/aspect tricks, so
// crop is always (0,0) and original_size==target_size==the actual output size, matching a plain
// "generate at this resolution" request (see conditioner.hpp's get_learned_condition_common).
static std::vector<float> build_micro_conditioning(const std::vector<float> & pooled, int height, int width) {
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

// sigma at t=999 under the standard SD/SDXL noise schedule (beta_start=0.00085, beta_end=0.012,
// scaled-linear, 1000 steps) -- see CompVisDenoiser/DiscreteScheduler in denoiser.hpp. For a 1-step
// Euler run, the schedule's only other sigma is sigma_next=0 (t_max -> 0 in a single hop), and the
// Euler update x + (x-denoised)/sigma*(sigma_next-sigma) reduces to exactly `denoised` when
// sigma_next==0 -- so the whole sampler is: predict eps once at sigma_max, then
// denoised = x_noisy - sigma_max*eps_pred. No iterative solver needed for n_steps=1.
static float compute_sigma_max() {
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

static void print_stats(const char * name, const std::vector<float> & v) {
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

int cmd_txt2img(int argc, char ** argv) {
    std::string model_path, prompt, out_path = "txt2img.png", backend_name = "cpu";
    std::string dump_latent_path;
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
        } else if (a == "--dump-latent" && i + 1 < argc) {
            dump_latent_path = argv[++i];
        } else if (a == "-h" || a == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (model_path.empty()) {
            model_path = a;
        } else if (prompt.empty()) {
            prompt = a;
        }
    }
    if (model_path.empty() || prompt.empty() || width % 8 != 0 || height % 8 != 0) {
        print_usage(argv[0]);
        return 1;
    }

    ggml_backend_t backend = init_backend(backend_name, 8);

    struct gguf_context * gctx = nullptr;
    weights_store ws = load_weights(model_path.c_str(), backend, &gctx);

    clip_text_model clip_l = load_clip_text(ws.ctx, "cond_stage_model.transformer.text_model.",
                                             12, 768, 12, false, false);
    clip_text_model clip_g = load_clip_text(ws.ctx, "cond_stage_model.1.transformer.text_model.",
                                             32, 1280, 20, true, true);
    sdxl_unet unet = load_sdxl_unet(ws.ctx);
    vae_decoder vae = load_vae_decoder(ws.ctx);
    fprintf(stderr, "CLIP + U-Net + VAE loaded (backend=%s)\n", backend_name.c_str());

    clip_vocab vocab = load_clip_vocab();
    std::vector<int32_t> ids = vocab.tokenize(prompt);
    sdxl_text_condition cond = clip_encode(clip_l, clip_g, backend, ids);
    std::vector<float> y = build_micro_conditioning(cond.pooled, height, width);
    fprintf(stderr, "text conditioning ready (context %zu floats, y %zu floats)\n",
            cond.context.size(), y.size());

    int64_t latent_w = width / 8, latent_h = height / 8;
    size_t n_latent = (size_t)(latent_w * latent_h * 4);
    std::mt19937 rng(seed != 0 ? seed : std::random_device{}());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> noise(n_latent);
    for (float & v : noise) v = dist(rng);

    float sigma_max = compute_sigma_max();
    float c_in = 1.0f / sqrtf(sigma_max * sigma_max + 1.0f); // sigma_data=1

    std::vector<float> x_noisy(n_latent), model_input(n_latent);
    for (size_t i = 0; i < n_latent; i++) {
        x_noisy[i] = noise[i] * sigma_max; // CompVisDenoiser::noise_scaling, from a zero (txt2img) latent
        model_input[i] = x_noisy[i] * c_in;
    }

    print_stats("x_noisy", x_noisy);
    print_stats("model_input", model_input);
    fprintf(stderr, "running U-Net (sigma_max=%.3f, c_in=%.5f)...\n", sigma_max, c_in);
    std::vector<float> eps_pred = unet_forward(unet, backend, model_input, latent_w, latent_h,
                                                999.0f, cond.context, (int) ids.size(), y);
    print_stats("eps_pred", eps_pred);

    std::vector<float> denoised(n_latent);
    for (size_t i = 0; i < n_latent; i++) {
        denoised[i] = x_noisy[i] - sigma_max * eps_pred[i]; // c_skip=1, c_out=-sigma_max
    }
    print_stats("denoised", denoised);

    if (!dump_latent_path.empty()) {
        FILE * f = fopen(dump_latent_path.c_str(), "wb");
        fwrite(denoised.data(), sizeof(float), denoised.size(), f);
        fclose(f);
        fprintf(stderr, "dumped denoised latent (%zu floats, %lldx%lld) to %s\n",
                denoised.size(), (long long) latent_w, (long long) latent_h, dump_latent_path.c_str());
    }

    // SDXL's VAE was trained/calibrated on latents ~1/0.13025 larger than the diffusion U-Net's own
    // latent scale (diffusers: `latents = latents / vae.config.scaling_factor` before decode) --
    // skipping this produced solid-black/saturated output during testing (confirmed directly: the
    // VAE decoder graph itself is fine on correctly-scaled input, either random noise or a real
    // denoised latent; it's specifically this missing rescale that broke it).
    const float SDXL_VAE_SCALING_FACTOR = 0.13025f;
    std::vector<float> latent_for_vae(n_latent);
    for (size_t i = 0; i < n_latent; i++) {
        latent_for_vae[i] = denoised[i] / SDXL_VAE_SCALING_FACTOR;
    }

    fprintf(stderr, "decoding VAE...\n");
    std::vector<float> pixels = vae_decode(vae, backend, latent_for_vae, latent_w, latent_h);
    print_stats("pixels", pixels);

    if (!sdxl_write_png(out_path, pixels, width, height)) {
        return 1;
    }
    fprintf(stderr, "wrote %s\n", out_path.c_str());

    ggml_backend_buffer_free(ws.buffer);
    ggml_free(ws.ctx);
    ggml_backend_free(backend);
    gguf_free(gctx);
    return 0;
}
