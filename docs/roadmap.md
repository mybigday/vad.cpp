# Roadmap

## Models

| model | output | license | plan |
|---|---|---|---|
| [desert-ant-labs/who](https://huggingface.co/desert-ant-labs/who) | speaker diarization / recognition | Desert Ant Labs Source-Available | planned; only ONNX/TFLite/Core ML are published, so it needs an ONNX converter |

Done: [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) (`sortformer`, docs/gguf.md). It loads NVIDIA's GGUF as is and the safetensors through `tools/convert_sortformer.py`. Offline and streaming match the transformers implementation within 2e-5 on CPU. Open items:
- The `head.activity_head` of the published GGUF (LayerNorm + 3 classes) is not in the reference model. Expose it once its meaning is documented.
- `vad_streams_push` runs diarization streams one after another. Batching the chunk steps of several streams needs per-stream masks in one encoder graph.
- Verified on WASM (node, single-threaded, about 12 min for the test) and on a Pixel 7 Pro (Android NDK, within 1.3e-5 of the reference). The browser/WebGPU path has not been tried with the 400 MB F32 file.

## Platforms

- **Web:** WASM (CPU, SIMD128) and WebGPU are done (docs/platforms.md).
- **ARM:**
  - Verified on a MediaTek Genio board (aarch64 Linux, Cortex-A55 + A78) and a Pixel 7 Pro (Android 14, NDK build, Tensor G2): all tests pass with the `vad-cpu` kernels.
  - Benchmarks are in `bench/results/arm_genio.txt`.

## Performance

- Done on every backend: the FireRed and smart-turn spectra are an FFT op (`src/models/frame_dft.cpp`, 8 frames per pass, run on the CPU) instead of a DFT-matrix mul_mat that read 0.8 MB (FireRed) per frame. FireRed single-frame streaming: 37 → 29 µs on x86, 137 → 123 µs on a Pixel X1 core; smart-turn unchanged.
- Done on x86 (AMD Zen 5, `bench/results/`): `vad-cpu` beats ggml-cpu on every model, precision, thread count (1–16) and benchmark mode.
  - The large-batch GEMM is a cache-blocked, register-tiled microkernel with K blocking and 2D task tiling.
  - The thread pool has per-thread task ranges, work stealing and per-task completion.
  - SSM_CONV, FSMN, ROPE and softmax with masks run natively.
  - Unary math is vectorized.
- **ARM.**
  - Done: an ARMv8.2 (fp16 + dotprod) variant chosen at runtime, NEON register tiles, streaming one-row F32 GEMV, and per-ISA thread-pool tuning (timed spin; few-row GEMMs split only when the work pays for the dispatch, 2 chunks per thread for big.LITTLE).
  - `vad-cpu` beats ggml-cpu on the Genio board and the Pixel in nearly every cell (`bench/results/arm_genio.txt`), typically 1.2–3× and far more for multi-threaded streaming.
  - Closest cases:
    - Single-thread FireRed F32 streaming: at parity on the Genio, ahead on the Pixel's X1 (178 vs 204 µs). It is bound by streaming about 2.3 MB of F32 weights per 10 ms frame.
    - smart-turn F16 single-window streaming on the Genio's A78: 3–7% behind ggml-cpu at 4 and 8 threads (it wins offline, and on the Pixel by 1.6–2×). Routing its many-row GEMMs through the cache-blocked F32 kernel was tried and was slower on the Pixel.
  - To do:
    - SVE variants.
- **Stream vs offline.** FireRed differs by 1.8e-7 between streaming (or batched streams) and offline on `vad-cpu`, even for F32 files, on x86 and aarch64 (ggml-cpu is exact). Within tolerance; the cause is not located yet.
- **int8 dot product (VNNI) for Q8_0 files on x86.** On x86 the Q8_0 path is still weight-only (int8 → F32 per element), so it is slower than F32. ARM already uses int8 activations with `SDOT`.
- Fuse the LSTM cell for Silero's multi-chunk graphs.
- Batch sliding-window streams across streams in `vad_streams_push`.
