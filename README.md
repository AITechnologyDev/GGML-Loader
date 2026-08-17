# GGML-Loader

A model loader/inference engine aimed at mobile ARM devices with **Mali GPUs**, built on top of
[ggml](https://github.com/ggml-org/ggml). The goal is text (GGUF/LLM) *and* SDXL (image
diffusion) support with a Mali-tuned GPU backend -- llama.cpp alone only covers the text side,
and (see "Findings" below) its generic Vulkan path leaves real performance on the table on Mali
specifically.

**Status: early. Text inference works end-to-end on CPU. The Mali-specific GPU backend --
this project's actual reason to exist -- has not been started yet.**

## What's actually here right now

- A from-scratch forward pass for the `phi3` gguf architecture (RoPE with phi3's LongRoPE
  freq-factors, GQA attention, SwiGLU FFN, tied embeddings), written directly against ggml's
  tensor ops -- not a wrapper around llama.cpp.
- A persistent F16 KV cache (4096 tokens, ~512 MiB for Phi-4-mini's shape).
- A real GPT-2-style byte-level BPE tokenizer: encoder (pretokenize + merge-rank BPE from the
  model's own `tokenizer.ggml.merges`) and decoder, not just a decoder.
- A single `ggml-loader` CLI with three subcommands:
  - `ggml-loader inspect <model.gguf>` -- dump GGUF metadata and tensor list.
  - `ggml-loader run <model.gguf> -p "prompt" -n N` -- one-shot raw completion (no chat
    template) with separate prefill/decode timing, for benchmarking against
    `llama-cli`/`llama-bench` on identical prompts.
  - `ggml-loader chat <model.gguf> [--system "text"]` -- interactive multi-turn chat using the
    model's actual instruct format, with conversation history threaded through the KV cache
    (each turn only encodes+prefills the newest message, not the whole conversation).

Tested against `Phi-4-mini-instruct` (Q4_K_M). Execution is **CPU-only** right now -- ggml's
Vulkan backend is vendored and compiled in (as a comparison baseline, see below) but nothing in
this project's own CLI code calls it yet.

## What's NOT here yet

- **A Mali-specific GPU backend.** This is the actual point of the project and hasn't been
  started. See "Findings" for why it's needed rather than just using ggml's existing backends.
- SDXL / image diffusion support.
- The exact pretokenizer regex this model's `tokenizer.ggml.pre` names ("gpt-4o"). The encoder
  here uses the classic GPT-2 splitting rules instead (contractions/letter-run/digit-run/
  other-run/whitespace-run). Byte-level BPE is lossless regardless of where the pretokenizer
  draws chunk boundaries, so this still produces valid, round-trippable tokenization -- it just
  may not match the official tokenizer's exact token boundaries/count in edge cases.
- A prompt encoder path beyond plain text (no multimodal, no tool-calling).

## Findings that shaped the plan

Benchmarked on the actual target hardware (MediaTek Dimensity, Mali-G615 MC6) before writing any
GPU code, rather than assuming:

- ggml/llama.cpp's OpenCL backend is Adreno-only (`GGML_OPENCL_USE_ADRENO_KERNELS`, prebuilt
  Adreno binary kernels) -- Mali isn't mentioned in it at all.
- The Vulkan backend is portable and does run on Mali, but on this GPU it gave **no measurable
  speedup on prompt processing** (batched matmul) over CPU, and only ~3x on token generation.
  The GPU reports no matrix cores and no native int8 dot product, which is exactly what the
  generic coopmat-based Vulkan kernels are tuned to exploit -- so they're not getting traction
  here.
- Conclusion: there's a real, unfilled gap for Mali specifically (unlike Adreno, which has a
  hand-tuned path), which is what the planned custom backend is for.

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
./build/ggml-loader run model.gguf -p "The capital of France is" -n 24
./build/ggml-loader chat model.gguf
```

## License

MIT, see [LICENSE](LICENSE).
