# GGML-Loader

A model loader/inference engine aimed at mobile ARM devices with **Mali GPUs**, built on top of
[ggml](https://github.com/ggml-org/ggml). The goal is text (GGUF/LLM) *and* SDXL (image
diffusion) support with a Mali-tuned GPU backend -- llama.cpp alone only covers the text side,
and (see "Findings" below) its generic Vulkan path leaves real performance on the table on Mali
specifically.

**Status: early, but the CPU/Vulkan text-inference path is real now, not just a stub. The
Mali-specific GPU backend -- this project's actual reason to exist -- still hasn't been started.**

## What's actually here right now

- A from-scratch forward pass for four gguf architectures -- **phi3, qwen2, qwen3, llama** --
  written directly against ggml's tensor ops, not a wrapper around llama.cpp. Per-architecture
  differences (fused vs separate QKV, QKV bias, per-head QK-Norm, fused vs separate FFN gate/up,
  tied vs untied output, RoPE variant and scaling) are handled through a small set of flags read
  from each model's own gguf metadata, not a separate code path per model.
- A persistent F16 KV cache, sized per-model from its actual head/layer shape.
- A real GPT-2-style byte-level BPE tokenizer: encoder (pretokenize + merge-rank BPE from the
  model's own `tokenizer.ggml.merges`) and decoder, not just a decoder.
- **Both CPU and Vulkan execution**, selectable per run, sharing the identical graph-building
  code -- weights load straight into whichever backend's memory (`ggml_backend_tensor_set`, not
  the CPU-only bulk gguf loading path), so there's no separate "GPU version" to keep in sync.
- A single `ggml-loader` CLI with four subcommands:
  - `ggml-loader inspect <model.gguf>` -- dump GGUF metadata and tensor list.
  - `ggml-loader run <model.gguf> -p "prompt" -n N [--backend cpu|vulkan]` -- one-shot raw
    completion (no chat template) with separate prefill/decode timing.
  - `ggml-loader chat <model.gguf> [--system "text"] [--backend cpu|vulkan]` -- interactive
    multi-turn chat using the model's actual instruct format (phi3-style, ChatML, or
    llama3's header-id format, picked automatically), conversation history threaded through the
    KV cache -- each turn only encodes+prefills the newest message, not the whole conversation.
  - `ggml-loader bench <model.gguf> [--backend cpu|vulkan] [-p n][-n n][-r reps]` -- averaged
    pp/tg throughput (one untimed warmup cycle, then N timed repetitions on synthetic tokens),
    directly comparable to `llama-bench`'s own `-p`/`-n`/`-r`.

Verified against Phi-4-mini-instruct, Qwen2.5-Coder-7B, Qwen3-0.6B, and Llama-3.1-8B-Instruct,
all Q4_K_M/Q8_0 quantized.

## What's NOT here yet

- **A Mali-specific GPU backend.** This is the actual point of the project and hasn't been
  started. See "Findings" for why it's needed rather than just using ggml's existing backends.
- SDXL / image diffusion support.
- Other architectures the user has test models for but that turned out to be much bigger
  undertakings than "add a few flags": **qwen35** (Qwen3.5+, a hybrid Gated DeltaNet + MoE +
  vision-language architecture -- new op types we don't have at all), **nemotron_h** (NVIDIA's
  hybrid Mamba2+Transformer -- mixed layer types per block, no RoPE), and **dflash** (turned out
  to not be a standalone architecture at all -- it's a block-diffusion *draft model* for
  speculative decoding, needs a whole target+draft pipeline, not just a graph-building function).
- The exact pretokenizer regex each model's `tokenizer.ggml.pre` names (e.g. "gpt-4o", "qwen2",
  "llama-bpe"). The encoder here uses the classic GPT-2 splitting rules instead
  (contractions/letter-run/digit-run/other-run/whitespace-run). Byte-level BPE is lossless
  regardless of where the pretokenizer draws chunk boundaries, so this still produces valid,
  round-trippable tokenization -- it just may not match the official tokenizer's exact token
  boundaries/count in edge cases.
- A prompt encoder path beyond plain text (no multimodal, no tool-calling).

## Findings that shaped the plan

Benchmarked on the actual target hardware (MediaTek Dimensity, Mali-G615 MC6) before writing any
GPU code, rather than assuming:

- ggml/llama.cpp's OpenCL backend is Adreno-only (`GGML_OPENCL_USE_ADRENO_KERNELS`, prebuilt
  Adreno binary kernels) -- Mali isn't mentioned in it at all.
- The Vulkan backend is portable and does run on Mali, but on this GPU llama.cpp's own
  `llama-bench` shows it gives **almost no speedup on prompt processing** over CPU (pp128: 4.62
  CPU vs 4.76 Vulkan tok/s) and a real but modest ~3x on token generation. The GPU reports no
  matrix cores and no native int8 dot product, which is exactly what the generic coopmat-based
  Vulkan kernels are tuned to exploit -- so they're not getting traction here.
- Conclusion: there's a real, unfilled gap for Mali specifically (unlike Adreno, which has a
  hand-tuned path), which is what the planned custom backend is for.

## Benchmarks (informal, one device, no slogan yet)

`ggml-loader bench`, `-p 128 -n 64 -r 5`, vs `llama-bench` with matching `-p`/`-n`/`-r`, same
device, Phi-4-mini-instruct Q4_K_M:

| | pp128 (prefill) | tg64 (decode) |
| --- | --- | --- |
| **ggml-loader, cpu**    | 12.04 ± 1.18 | 3.42 ± 0.85 |
| **ggml-loader, vulkan** | 20.04 ± 0.19 | 10.99 ± 0.06 |
| llama.cpp, cpu (`-ngl 0`)   | 4.62 ± 0.05 | 2.99 ± 0.25 |
| llama.cpp, vulkan (`-ngl 99`) | 4.76 ± 0.07 | 9.80 ± 0.06 |

On this run: ~4.2x faster than llama.cpp's Vulkan on prefill, ~1.12x faster on decode. **Caveat
before reading too much into it:** this llama-bench run didn't pass `-fa` (flash attention), and
if that's off by default here, both sides are using an equivalent-effort ("unfused") attention
path -- meaning the gap is most likely explained by llama.cpp's more general-purpose overhead
(multi-sequence batching, a more flexible KV cache, speculative-decoding hooks) that this
narrower, single-purpose implementation doesn't pay, not by faster kernels -- the underlying
compute ops are the same ggml Vulkan kernels either side. Re-testing with `-fa` enabled, and on
more than one architecture/model size, is the natural next check before trusting this fully.

## Building

```sh
git submodule update --init --recursive   # pulls third_party/ggml
cmake -S . -B build
cmake --build build --target ggml-loader -j$(nproc)
```

Two small local patches are applied to the vendored `third_party/ggml` (uncommitted, in the
working tree, not upstream) to work around a Termux `shaderc` bug that otherwise fails the
Vulkan shader build (`shaderc: internal error: ... Invalid capability operand`) on
`coopmat2`/`USE_OCP_FP4` shaders -- see the comments at the patched sites in
`third_party/ggml/src/ggml-vulkan/` for details. These shader variants aren't relevant to Mali
(no matrix cores) so disabling them costs nothing on this hardware.

## Usage

```sh
./build/ggml-loader inspect model.gguf
./build/ggml-loader run model.gguf -p "The capital of France is" -n 24 --backend vulkan
./build/ggml-loader chat model.gguf --backend vulkan
./build/ggml-loader bench model.gguf -p 128 -n 64 -r 5 --backend vulkan
```

## License

MIT, see [LICENSE](LICENSE).
