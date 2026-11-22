# basalt-arm-sm121

**An Arm64 / sm_121 port of the Basalt inference engine.** Runs Qwen3.8-Flash-Next and Qwen3.8-27B natively on
machines like the NVIDIA RTX Spark N1X — Arm64 CPUs plus integrated Blackwell GPUs (compute capability 12.1, the same
generation as the DGX Spark's GB10) — and builds natively on both Windows on Arm64 and Linux Arm64.

It is a port of the [Basalt](https://github.com/jesdga95/basalt) inference engine to Arm64 CPUs and compute
capability 12.1 GPUs (RTX Spark N1X, DGX Spark / GB10), building natively on Windows on Arm64 (MSVC) and on Linux Arm64.
On an RTX Spark N1X it decodes Qwen3.8-Flash-Next about 2.1 times as fast as vLLM on the same requests. **This fork
also adds the first verified DGX Spark (GB10) run of Flash-Next MoE, plus a Linux unified-memory fix**; see the DGX Spark
section below. Build guide, design notes, test status and measurements: [docs/arm64.md](docs/arm64.md). The upstream README is
kept as [README.basalt.md](README.basalt.md).

## What this is

- **Upstream**: [jesdga95/basalt](https://github.com/jesdga95/basalt) (MIT licensed), a local LLM inference engine for
  NVIDIA Blackwell GPUs, with MoE expert tiering, MTP / block-draft speculative decoding, and OpenAI- and
  Anthropic-compatible APIs. Upstream only supports x86-64 CPUs + sm_120 discrete GPUs.
- **This repository**: built on the latest upstream code (`63cb68a`, after v0.4.1) with support added for Arm64 CPUs and
  compute capability 12.1 integrated GPUs, maintained independently and not upstreamed. Upstream files receive only
  minimal mechanical changes, and new logic lives in new files where possible, to make future upstream syncs easy.
- **Test machines**: an RTX Spark N1X laptop (128 GB unified memory, 96 GB of it assigned to the GPU) running
  Windows 11 on Arm natively, plus WSL2 (Ubuntu 26.04 aarch64) on the same machine. **The DGX Spark (GB10) has also been
  verified on this fork** — see the "DGX Spark (GB10) results" section below. The original port's README listed the
  DGX Spark as "not tested"; this fork fills that gap.

## What changed vs. upstream

**Arm64 builds**
- Both Linux Arm64 (GCC) and Windows on Arm64 (native MSVC) build fully; one build targets one CPU architecture and one
  compute capability, and x86-64 + sm_120 stays unchanged.
- CPU kernels replaced with NEON versions: dot products for the various quant formats are generated from llama.cpp's
  Arm kernels (MIT); the NVFP4, multi-token Q2_0, router, and vision GEMM kernels are written by hand.
- Memory ordering: Arm64 counterparts exist for spin-waiting, load/store barriers, and prefetching; a read barrier is
  added after polling the GPU-written ring queues.

**Integrated-GPU (N1X) specifics**
- **VRAM reporting**: `cudaMemGetInfo` counts shared-memory segments toward the total and cannot see VRAM mapped through
  the virtual-memory APIs. Changed to treat the memory assigned to the GPU as the total and to compute usage from
  system-level statistics; all capacity planning uses this reading.
- **Windows commit-limit guard**: on an integrated GPU every VRAM allocation immediately counts against Windows' commit
  limit (physical memory + page file). The engine leaves headroom for the system so other programs don't get crowded
  out, and recommends setting the page file large enough (see below).
- **Late upload arrivals**: after a synchronous copy from pageable memory to the GPU, the data may not actually have
  arrived yet; on integrated GPUs one extra sync is performed after the copy.
- Skips NPUs listed by NVML, budgets pinned memory from DXGI's local segment, and keeps the machine awake at runtime on
  Modern Standby systems.

**Tooling**
- `tools/port/fetch_cuda.py`: assembles a CUDA toolkit from NVIDIA's redistributable packages (no installer; every
  component SHA-256-verified).
- `tools/port/env-win-arm64.cmd` and `build-win-arm64.cmd`: one-command setup and build for Windows on Arm64.
- `tools/serve_speed.py`: server-side speed testing (long context; sends the same prompts as basalt-bench to vLLM and
  other engines for comparison).

## Measured speed (RTX Spark N1X, Windows 11 on Arm)

Qwen3.8-Flash-Next (`NVFP4-DQ6_K` model package), the same basalt-bench prompts, vLLM = community vLLM 0.32.0
(NVIDIA's NVFP4 weights, MTP draft 2):

| | Basalt | vLLM |
| --- | ---: | ---: |
| Decode: technical explainer (conv) | 80.8 tok/s | 49.1 tok/s |
| Decode: JSON (struct) | 140.3 tok/s | 57.9 tok/s |
| Decode: prose (prose) | 67.8 tok/s | 45.6 tok/s |
| Decode: token-weighted | **~116 tok/s** | ~55 tok/s |
| Prefill 16k / 64k | 2,558 / 2,638 tok/s | 1,753 / 1,780 tok/s |

Long context (server, int8 KV, one long Chinese document, thinking disabled):

| Prompt length | Prefill | Decode |
| --- | --- | --- |
| 4,128 | 1,473 tok/s | 63.6 tok/s |
| 191,231 | 2,387 tok/s | 55.1 tok/s |
| 260,124 | 2,224 tok/s (117 s) | 54.3 tok/s |

With 4 concurrent slots: 1 / 2 / 4 simultaneous decode streams give total throughput of 63.7 / 78.4 / 80.1 tok/s.

Qwen3.8-27B (`QUASAR-NVFP4` package, dense model): decode conv 50, struct 157, prose 36 tok/s (16.5 tok/s with drafting
disabled — already near the memory-bandwidth ceiling); prefill 16k = 2,757 tok/s; a 260k long prompt prefills at
1,036 tok/s and decodes at 14.3 tok/s; 4-way concurrent total throughput 72.3 tok/s.

Full data and methodology: [docs/arm64.md](docs/arm64.md#speed-on-the-rtx-spark-n1x).

## DGX Spark (GB10) results (new in this fork)

The port author only tested the RTX Spark N1X, and the original README always listed the DGX Spark as "not tested".
This fork completes the first verified Flash-Next MoE run on a DGX Spark, and fixes a VRAM-detection issue under Linux
unified memory. The patch was submitted to the port author as
[PR #1](https://github.com/yanmindi/basalt-arm-sm121/pull/1); it touches only `src/core/device_memory.cpp`, about 40 lines.

**Environment**: DGX Spark (GB10, ARM64, sm_121, 128 GiB unified memory), Ubuntu 24.04, CUDA 13.0. Built with
`cmake -DBASALT_MODELS=qwen4exp`; ctest 118/119 (the only failure is the known `video_test`).

**Out-of-the-box failure and root cause**: starting with the usual parameters fails with
`no VRAM is left for the expert cache: needs 285 slots (751 MiB), about 32727 MiB more than this card has free`.
On the DGX Spark, NVML cannot read any dedicated VRAM (`nvidia-smi` shows N/A, `dedicated == 0`), so the engine falls
back to the raw `cudaMemGetInfo` reading; under Linux UMA that reading counts reclaimable page cache as used and
reports only 13.73 GiB free, while the system's `MemAvailable` is actually 118 GiB. From this the engine extrapolates
the 64.3 GiB of RAM experts into roughly 45 GiB of phantom VRAM usage, sizes the expert cache at 0 slots, and verify
fails as a consequence. The patch switches to `MemAvailable` for cache planning in the
"Linux + integrated GPU + no NVML carve-out" case only; other platforms are unaffected, `BASALT_UMA_CF=0` restores the
old behavior, and `BASALT_UMA_CAP_GIB=N` caps the estimate.

**Load after the patch** (launch parameters identical to the failing case):

```sh
loaded   68.4GiB 7.5s
experts live in VRAM 63.3 GiB (100%) ... file (SSD) 0.0 GiB (0%)
```

All 24,576 experts are 100% resident in VRAM with zero SSD reads, and every 1–8-token verify window passes. This is the
shape the port author had previously only seen on the N1X.

**A practical 262K configuration**:

```sh
LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
bin/basalt serve models/Qwen3.8-FN-NVFP4-DQ6_K.basalt \
  --max-context 262144 --parallel 4 --slot-full-context --kv int8 \
  --port 18190 --alias Qwen3.8-Flash-Next \
  --python /path/to/venv/bin/python
```

`--parallel` defaults to 1 (extra requests just queue; there is no true concurrency), and `--slot-full-context` is
required to give multiple streams their own full context. Mind the edge case: if all 4 slots genuinely fill 262K at
once (1,048,576 cells total), KV growth squeezes the expert cache down to ~292 slots and experts start swapping out;
short-context agents are unaffected.

Single-stream decode (server `predicted_per_second`, temp=0, 256 tokens, 2 runs per scenario):

| Scenario | 8K context | 262K context |
| --- | ---: | ---: |
| chat | 59.8 tok/s | 58.5 tok/s |
| code | 83.6 tok/s | 82.0 tok/s |
| agent (tool_calls) | 79.2 tok/s | 78.5 tok/s |

Going from 8K to 262K costs almost no speed.

**A/B against the existing vLLM setup** (same prompts, temp=0, 256 tokens, 2 runs per combination; the two engines
cannot coexist in memory, so they were tested serially; the table shows end-to-end total throughput = generated
tokens / wall-clock, including prefill — the only metric measured the same way on both sides):

| Scenario | Concurrency | This setup (Basalt 262K) | vLLM | Ratio |
| --- | ---: | ---: | ---: | ---: |
| chat | 1 | **52.4** | 19.7 | 2.66x |
| chat | 2 | **69.1** | 31.5 | 2.19x |
| chat | 4 | **74.2** | 44.7 | 1.66x |
| code | 1 | **71.5** | 33.1 | 2.16x |
| code | 2 | **81.6** | 44.3 | 1.84x |
| code | 4 | 74.7 | **114.5** | **0.65x** |
| agent | 1 | **51.7** | 25.5 | 2.03x |
| agent | 2 | **63.5** | 43.6 | 1.46x |
| agent | 4 | 54.3 | **87.0** | **0.62x** |

The verdict is mixed: at single-stream and low concurrency this setup leads by roughly 2x, but at 4-way concurrency
with short outputs vLLM pulls ahead — a single stream is already near the LPDDR5X bandwidth ceiling, extra streams
mostly split the same bandwidth, and vLLM's continuous batching merges the 4 short requests into one round. **Do not**
take the single-stream decode numbers above and compare them against anyone else's pure-decode numbers (these include
prefill), and do not read this as "Basalt is generally 2x faster".

Known blemishes: under `--parallel 4`, the agent's native tool calls intermittently returned HTTP 400 (`arguments`
emitted as a JSON object instead of a string); a follow-up run of 14 requests returned all 200s, so it's an
intermittent issue. Per-stream speeds are uneven under concurrency (steady-state server logs show ~17–21 tok/s per
stream, ~77 in total); just look at total throughput.

## Quick start: Windows on Arm64

Requires: Visual Studio 2026 or its Build Tools (with the ARM64 C++ toolset checked), Python 3.10+ (Arm64 build), and a
large-enough page file (see below).

```bat
git clone https://github.com/yanmindi/basalt-arm-sm121 && cd basalt-arm-sm121
python tools\port\fetch_cuda.py windows-arm64 C:\cuda-13.5
tools\port\build-win-arm64.cmd build C:\cuda-13.5
```

Download a model package (either one; in mainland China you can set `HF_ENDPOINT=https://hf-mirror.com` to go through
the mirror):

```bat
hf download jesdga/Qwen3.8-Flash-Next-Basalt Qwen3.8-FN-NVFP4-DQ6_K.basalt --local-dir models
hf download jesdga/Qwen3.8-27B-Basalt Qwen3.8-27B-QUASAR-NVFP4.basalt --local-dir models
```

Start the server (OpenAI / Anthropic-compatible API, default `http://127.0.0.1:8080`; the same address in a browser
shows a stats page):

```bat
set PATH=C:\cuda-13.5\bin\arm64;%PATH%
pip install -r requirements.txt
bin\basalt.exe serve models\Qwen3.8-FN-NVFP4-DQ6_K.basalt --python python.exe --max-context 262144 --kv int8
```

`--python` takes the path to the Python interpreter that has the dependencies installed. `bin\arm64` must be on PATH,
otherwise Windows pops up a "cublas64_13.dll not found" dialog.

## Quick start: Linux Arm64

```sh
python3 tools/port/fetch_cuda.py linux-sbsa ~/cuda-13.5
export PATH=~/cuda-13.5/bin:$PATH
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build
pip install -r requirements.txt
LD_LIBRARY_PATH=~/cuda-13.5/lib bin/basalt serve models/Qwen3.8-FN-NVFP4-DQ6_K.basalt --max-context 262144 --kv int8
```

Under WSL2, also add `/usr/lib/wsl/lib` to `LD_LIBRARY_PATH`.

## Usage notes

- **Page file**: on Windows, integrated-GPU memory counts against the commit limit. On a 128 GB N1X, set the page file
  to a custom size of **at least ~72 GB** (we use 81,920–98,304 MB). With Windows' auto-managed 68 GB page file, the
  engine conservatively keeps fewer experts in VRAM and each request runs about 0.9 s slower.
- Run only one large model at a time: Flash-Next takes over 70 GB once loaded — don't run it alongside vLLM or any
  other inference server.
- **Test status**: Linux Arm64 224/225 passing, Windows on Arm64 213/219 passing. Every failing test and its reason:
  [docs/arm64.md](docs/arm64.md#tests).
- **Not yet tested**: real DeepSeek-V4.1-Flash weights (the package is over 260 GB), actually entering standby, image
  and video input, installing dependencies into a fresh Arm64 Python environment, and x86-64 builds after this port.
  (The DGX Spark has been tested on this fork — see above.)

## Documentation

- [docs/arm64.md](docs/arm64.md): Arm64 build, integrated-GPU handling, test status, measured numbers (in English).
- [README.basalt.md](README.basalt.md): the original upstream README; its guidance on supported models, config files,
  the API, video, and other usage still applies.
- [docs/](docs/): the full upstream documentation (running, configuration, models, precision, model-file format).

## License and credits

- This repository inherits the upstream MIT license ([LICENSE](LICENSE)); licenses for third-party code
  (ggml / llama.cpp) are in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
- Basalt is developed by [jesdga95](https://github.com/jesdga95); model packages come from their Hugging Face page.
- The Windows on Arm64 work took inspiration from SolettaSolaris's approach in upstream PR
  [jesdga95/basalt#230](https://github.com/jesdga95/basalt/pull/230), without reusing its code.
- This repository is not affiliated with the upstream project or with NVIDIA.
