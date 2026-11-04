# vad.cpp

Voice activity detection and speaker diarization on [ggml](https://github.com/ggml-org/ggml), with stateful streaming, batched multi-stream inference, and CPU kernels tuned for small graphs.

Status: early.

| model | output | hop | lookahead | official parity on CPU |
|---|---|---|---|---|
| [Silero VAD](https://github.com/snakers4/silero-vad) v5, v6 | speech | 32 ms | 0 | ≤ 2e-6 |
| [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) Stream-VAD | speech | 10 ms | 0 | ≤ 3e-5 |
| FireRedVAD VAD (non-streaming model) | speech | 10 ms | 1.6 s | ≤ 2e-5 |
| [smart-turn-v3](https://huggingface.co/pipecat-ai/smart-turn-v3) v3.2 | turn end | 0.5 s (configurable) | 0 | ≤ 1e-6 |
| [Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) (Streaming Sortformer) | 8 speakers | 10 ms | 0.08–3.2 s (chunked) | ≤ 2e-5 |


- **Exact results.**
  - Silero follows the official streaming semantics, including the 64-sample context that whisper.cpp skips.
  - FireRedVAD's Kaldi fbank front end runs inside the graph.
  - Streaming output equals the official full-utterance results frame by frame.
- **Loads the published files.** It reads vad.cpp GGUF files, whisper.cpp's `ggml-silero-*.bin` files and NVIDIA's `Nemotron-3-Diarization.q8_0.gguf` as they are.
- **One interface for every model.** No model-specific calls: a model maps audio to frames of probabilities, and what they mean (speech, turn end or per-speaker activity), how many there are per frame, their hop, warm-up and lookahead are model properties.
- **Every model streams.**
  - Causal models stream exactly. Their recurrent state stays resident on the compute device, so a stream that keeps using the same slot never copies its state.
  - Models that need future context stream through a sliding window, with slice and lookahead as accuracy/latency/compute knobs. With the maximum lookahead the output is identical to offline.
  - Chunked models (diarization) run their own streaming algorithm (FIFO and speaker cache on the host); slice and lookahead select the chunk and its look-ahead.
- **Many streams at once.** `vad_streams_push` batches the work of N streams into shared computes.
- **Runs at the file's precision.**
  - F32, F16, BF16 or quantized GGUF files.
  - Which tensors to quantize is the user's choice (`scripts/quantize-gguf.py`).
  - Only the original precision is checked against the references.
- **One graph for every device.**
  - Each model is a single ggml graph.
  - On CPU it runs on the `vad-cpu` backend. This backend compiles the graph once into a plan:
    - fused GEMM + bias + ReLU, and a fused streaming FSMN/short-conv step,
    - packed weights,
    - batch rows flattened into one GEMM,
    - its own threading,
    - kernels dispatched by ISA (AVX2, AVX-512 and AVX-512 VNNI at runtime; NEON with an ARMv8.2 fp16/dotprod variant chosen at runtime on Linux/Android; WASM SIMD at build time).
  - Other devices (Vulkan, CUDA, Metal, …) use the stock ggml backends.

## Build

```sh
git submodule update --init
cmake -B build                      # add -DGGML_VULKAN=ON, -DGGML_CUDA=ON, ... for accelerators
cmake --build build -j
```

| option | default | |
|---|---|---|
| `VAD_CPU_KERNELS` | ON | CPU graphs run on vad.cpp's kernels. OFF uses stock ggml-cpu for everything. |
| `VAD_CPU_NATIVE` | OFF | Build the CPU kernels only for the host CPU (`-march=native`). |
| `VAD_BUILD_TESTS` / `_EXAMPLES` / `_BENCH` | ON when top level | |

ARM cross-compilation, WebAssembly (node, browsers) and WebGPU are covered in [docs/platforms.md](docs/platforms.md).

The `third_party/ggml` submodule is pinned to the ggml that [whisper.rn](https://github.com/mybigday/whisper.rn) vendors (ggml 0.26.0, `ffa4e8b8`, the version synced into its whisper.cpp), so both can share one ggml build. vad.cpp's `vad-cpu` backend uses ggml's backend interface headers, so move the pin together with whisper.rn's.

When vad.cpp is used as a subdirectory and the parent project already defines a `ggml` target (for example whisper.cpp or llama.cpp), vad.cpp reuses it.

## Models

```sh
scripts/prepare-test-data.sh   # downloads official models + whisper.cpp files, converts, makes test references
# or convert one model:
cd tools
python convert_silero_onnx.py silero_vad.onnx ../models/silero-v6.2.gguf v6.2
python convert_firered_onnx.py fireredvad_stream_vad.onnx cmvn.ark ../models/firered-stream-vad.gguf
python convert_smart_turn_onnx.py smart-turn-v3.2-gpu.onnx ../models/smart-turn-v3.2.gguf v3.2
python convert_sortformer.py Nemotron-3-Diarization/ ../models/nemotron-3-diarization.gguf   # F32, NVIDIA's GGUF schema
```

NVIDIA's own `Nemotron-3-Diarization.q8_0.gguf` loads as is; `convert_sortformer.py` writes the same schema from the Hugging Face checkpoint at the original F32 precision (the published file is Q8_0, which the tests only report).

The file format is described in [docs/gguf.md](docs/gguf.md).

## Usage

```sh
build/vad-cli -m models/silero-v6.2.gguf -f audio.wav            # speech segments (seconds)
build/vad-cli -m models/ggml-silero-v6.2.0.bin -f audio.wav --probs
build/vad-cli -m models/smart-turn-v3.2.gguf -f utterance.wav     # turn-complete probability
build/vad-cli -m Nemotron-3-Diarization.q8_0.gguf -f meeting.wav  # segments per speaker
build/vad-cli --list-devices
```

```c
#include "vad.h"

vad_model * m = vad_model_load("model.gguf", vad_model_default_params());   // any model, any format
vad_stream * s = vad_stream_new(m);
const int no = vad_model_n_outputs(m);                       // values per frame (1, or one per speaker)
float probs[256 * 8];
int n = vad_stream_push(s, samples, n_samples, probs, 256);   // frame i (probs[i * no ...]) refers to the audio
                                                             // ending at warmup + (i + 1) * vad_model_hop_samples(m)
n += vad_stream_flush(s, probs + n * no, 256 - n);            // end of audio
vad_stream_free(s);
vad_model_free(m);
```

[include/vad.h](include/vad.h) documents the rest:
- `vad_model_output` tells whether a frame holds speech, turn-end or per-speaker probabilities, `vad_model_n_outputs` how many;
- `vad_stream_new_with_params` sets slice and lookahead for models that need future context;
- `vad_streams_push` handles batched streams;
- `vad_detect` and `vad_detect_with_params` handle offline buffers (for example, the turn-end probability at the end of an utterance: `slice_samples = INT32_MAX`, then flush);
- `vad_segments_from_probs` turns speech or speaker probabilities into segments (per channel), with silero-vad's segmentation.

### Streaming models with lookahead

FireRedVAD's non-streaming model sees 1.6 s of future audio. A stream re-runs a window every `slice_samples`, keeping 1.52 s of past context so emitted frames are exact with respect to the past. `lookahead_samples` sets how much future audio each frame waits for.

Measured on the jfk sample (CPU, single thread), compared with the full offline result:

| lookahead | max deviation from offline | latency |
|---|---|---|
| 1.6 s (max) | 0 | 1.6 s + slice |
| 1.0 s | 7.8e-3 | 1.0 s + slice |
| 0.4 s | 0.22 | 0.4 s + slice |
| 0 | 0.58 | slice |

Compute per 10 ms frame (single thread; offline is 16 µs):

| slice | max lookahead | 0.5 s lookahead |
|---|---|---|
| 250 ms | 189 µs | 125 µs |
| 500 ms | 95 µs | 63 µs |
| 1000 ms | 56 µs | 39 µs |

### Speaker diarization

Nemotron-3-Diarization emits 8 speaker probabilities every 10 ms. Speakers are numbered by first appearance, and a returning speaker keeps its number through the speaker cache.
- `vad_detect` runs the model's offline mode: 27.2 s chunks with 3.2 s of look-ahead.
- Streams process one chunk per step. `slice_samples` sets the chunk (in 80 ms frames, default 9) and `lookahead_samples` its look-ahead. The defaults are the model card's presets: 9 + 4 frames gives 1.04 s of latency, 6 + 2 gives 0.64 s, and 3 + 1 gives 0.32 s.
- Every mode matches the official transformers implementation within 2e-5 on CPU (`tests/test_diarization.cpp`, 84 s and four speakers).

Speed on an 84 s recording, in µs per 10 ms frame (`bench/results/nemotron_x86.txt`):

| device | F32 default stream (1.04 s latency) | F32 offline | Q8_0 stream | Q8_0 offline |
|---|---|---|---|---|
| `cpu`, 4 threads | 1568 (real-time factor 0.16) | 82 | 1449 | 78 |
| `cpu`, 16 threads | 725 | 38 | 685 | 34 |
| ggml-cpu, 4 threads | 3927 | 215 | 4769 | 261 |
| ggml-cpu, 16 threads | 1533 | 82 | 1741 | 95 |
| Vulkan (Radeon 8060S) | 335 | 27 | 268 | 25 |

## Weight precision and quantization

vad.cpp computes at the precision stored in the model file. The converters write F32 (`--outtype f16|bf16` for half precision). `scripts/quantize-gguf.py` rewrites a file per tensor, in the same style as codec.cpp:

```sh
python scripts/quantize-gguf.py models/firered-stream-vad.gguf firered-q8.gguf --type Q8_0
python scripts/quantize-gguf.py in.gguf out.gguf --type F16 --tensor-type 'enc2=F32'   # keep one tensor F32
```

- **Default selection:** `--type` applies to every tensor that can take it. It skips 1-D tensors, tensors the architecture uses outside mul_mat (such as FireRed's FSMN taps and smart-turn's positions), and rows the block size doesn't divide.
- **Overrides:** `--tensor-type REGEX=TYPE` overrides per tensor.
- **Invalid files:** if a forced type cannot work for a tensor's op, `vad_model_load` fails and names the tensor.
- **No quality guarantee:** quantization does not guarantee quality, and choosing tensors and types is up to the application. For example, Silero v6.2's small encoder layers are sensitive to int8.

Measured on the jfk sample (CPU, `cpu` device), as reported by the tests (not checked):

| file | max deviation from the official model | size |
|---|---|---|
| silero-v6.2 F16 | 3.2e-3 | 0.62 MB |
| silero-v6.2 Q8_0 | 0.21 | 0.48 MB |
| firered-stream F16 / Q8_0 | 2.8e-3 / 2.3e-2 | 1.2 / 0.73 MB |
| smart-turn-v3.2 F16 / Q8_0 | 1.3e-4 / 2.4e-3 | 16 / 9.4 MB |

Speed on the x86 test machine (1 thread, streaming, jfk sample):
- F16 helps where weight bandwidth matters (`vad-cpu` converts with F16C/AVX-512/NEON):
  - Silero: 15.0 → 8.5 µs;
  - FireRed: 34.0 → 27.2 µs;
  - smart-turn is compute-bound: 34.7 → 34.9 ms.
- Q8_0 runs with int8 activations where the CPU has int8 dot products (below): on AVX-512 VNNI, Silero 12.9 → 6.8 µs, FireRed 28.0 → 16.0 µs. Without them, the int8 → float conversion costs about what the bandwidth saves.

### Reduced-precision arithmetic

Like ggml, vad.cpp trades exactness for speed where the hardware offers it:
- **F32 weights** are computed in F32. Results match the reference implementations, and streaming matches offline exactly.
- **F16 and Q8_0 weights:** on CPUs with ARMv8.2 fp16 and dot-product instructions (detected at runtime; e.g. Cortex-A55/A76/A78 and later):
  - F16 weights run with activations rounded to fp16. Products are summed in fp16 over short blocks, then accumulated in F32. Blocks that overflow are recomputed exactly.
  - Q8_0 weights run with activations quantized to int8 per 32-element block (as in ggml's Q8_0 × Q8_0), multiplied with `SDOT`.
  - On x86 CPUs with AVX-512 VNNI (detected at runtime; e.g. AMD Zen 4/5, Intel Ice Lake and later), Q8_0 weights run the same way with `VPDPBUSD`.
  - Elsewhere they run with F32 activations.

Effect against the same model's F32 file (67.8 s and 11 s samples; ARMv8.2 on a Cortex-A78, VNNI on Zen 5):

| file | max / mean deviation, F32 activations | max / mean deviation, ARMv8.2 path | max / mean deviation, x86 VNNI | frames flipping at 0.5 (F32 act. / ARMv8.2 / VNNI) |
|---|---|---|---|---|
| silero-v6.2 F16 | 4.9e-3 / 6.7e-5 | 1.7e-2 / 2.5e-4 | – | 0 / 0 |
| firered-stream F16 | 3.0e-3 / 8.2e-5 | 7.6e-3 / 1.6e-4 | – | 0 / 0 |
| firered-vad F16 | 1.8e-3 / 2.7e-5 | 3.3e-3 / 7.8e-5 | – | 1 / 1 of 6778 |
| smart-turn-v3.2 F16 | 2.1e-3 / 2.9e-4 | 5.7e-3 / 6.6e-4 | – | 0 / 1 of 159 (0.498 → 0.501) |
| silero-v6.2 Q8_0 | 0.29 / 4.2e-3 | 0.33 / 5.4e-3 | 0.33 / 6.0e-3 | 3 / 6 of 2119 / 10 of 2463 |
| firered-stream Q8_0 | 2.3e-2 / 1.0e-3 | 2.7e-2 / 1.1e-3 | 2.6e-2 / 1.2e-3 | 0 / 0 / 0 |
| smart-turn-v3.2 Q8_0 | 5.0e-2 / 6.0e-3 | 8.0e-2 / 9.0e-3 | 8.7e-2 / 8.5e-3 | 2 / 2 of 136 / 2 of 159 |

## Performance

Test setup: AMD Ryzen AI Max+ 395 (Zen 5), 67.8 s of audio, F32 weights. Times are in µs per chunk (per stream for the 256-stream rows), measured with `build/vad-bench` in one session (min of 3 runs). See above for F16 weights. The full matrix over models, precisions and 1/4/8/16 threads is in `bench/results/vad_cpu_vs_ggml_x86.txt`, and ARM results are in `bench/results/arm_genio.txt`.

Silero v6.2 (32 ms chunks):

| | `cpu` 1 thread | `cpu` 8 threads | ggml-cpu 1 thread | ggml-cpu 8 threads | Vulkan (8060S) | whisper.cpp |
|---|---|---|---|---|---|---|
| streaming, 1 chunk per push | 13.0 | **10.6** | 25.9 | 37.2 | 201 | 47 |
| offline | 7.1 | **4.6** | 21.5 | 11.1 | 24.6 | 38 |
| 256 streams | 5.9 | **2.4** | 20.7 | 4.3 | 2.5 | – |

FireRedVAD Stream-VAD (10 ms frames):

| | `cpu` 1 thread | `cpu` 8 threads | ggml-cpu 1 thread | ggml-cpu 8 threads | Vulkan (8060S) |
|---|---|---|---|---|---|
| streaming, 1 frame per push | 36.9 | **31.2** | 41.4 | 34.7 | 403 |
| offline | 7.8 | **2.1** | 24.6 | 4.1 | 15.8 |
| 256 streams | 12.5 | **2.2** | 35.1 | 5.7 | 22.6 |

smart-turn-v3.2 (ms per 8 s window, 7.4 GFLOP):

| | `cpu` 1 thread | `cpu` 8 threads | ggml-cpu 1 thread | ggml-cpu 8 threads | Vulkan (8060S) |
|---|---|---|---|---|---|
| single window | 34.6 | **6.4** | 103 | 14.7 | 10.9 |
| batch of 8, per window | 36.5 | 6.1 | 104 | 15.3 | **6.0** |

- The whisper.cpp results also differ from official Silero by up to 0.23 in probability.
- ggml-cpu gets slower with more threads on these tiny graphs: it synchronizes the threads after every node.
- FireRedVAD's F32 weights (3.2 MB) do not fit in L2, so single-frame streaming is bound by memory bandwidth.
- Measurements and analysis are in [bench/README.md](bench/README.md).

## Tests

```sh
scripts/prepare-test-data.sh && (cd build && ctest)
VAD_TEST_DEVICES=Vulkan0,cpu@4 build/tests/test_models   # any device; "@4" = 4 threads
VAD_TEST_DATA=/path/models VAD_ASSETS=/path/assets build/tests/test_models   # data elsewhere (e.g. on a device)
```

The tests check:
- parity with the official implementations at the original precision:
  - ONNX Runtime;
  - for FireRedVAD, kaldi-native-fbank and ONNX Runtime;
  - for smart-turn, transformers' feature extractor and ONNX Runtime.
- Quantized files are run as well, but their deviation is only reported, not checked.
- streaming with random push sizes against offline results,
- interleaved streams (state moving between graphs and slots) against each stream alone,
- sliding-window streams of lookahead models against offline, and turn-end streams against direct evaluation,
- that a model with a quantized tensor its op cannot use is refused at load,
- batched streams against single streams,
- segments against the official Python implementation,
- every kernel ISA variant the CPU supports against a reference, including the F16-weight GEMM.

Known limitations:
- ggml-vulkan computes mul_mat at F16 precision and rejects mul_mat ops that require F32. On Vulkan, probabilities differ from ONNX by about 4e-3.
- FireRedVAD's and smart-turn's fbank/log-mel projections are pinned to F32, so the scheduler runs that part on the CPU. The power spectrum spans 1e-10 to 1e11, far outside F16.
- On Vulkan, smart-turn differs from ONNX by 1.5e-4.

## Licenses of supported models

- Silero VAD: MIT.
- FireRedVAD: Apache-2.0.
- smart-turn: BSD-2-Clause.
- Nemotron-3-Diarization: OpenMDW-1.1.

## License

vad.cpp is released under the MIT license ([LICENSE](LICENSE)). Its Silero VAD support started from [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (MIT, the ggml authors), and it builds on [ggml](https://github.com/ggml-org/ggml) (MIT).
