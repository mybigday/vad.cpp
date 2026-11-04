# Roadmap

## Models

No new models planned. desert-ant-labs/who was evaluated and dropped: its license does not allow porting the SDK logic its pipeline depends on.

Done: [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization) (`sortformer`, docs/gguf.md). It loads NVIDIA's GGUF as is and the safetensors through `tools/convert_sortformer.py`. Offline and streaming match the transformers implementation within 2e-5 on CPU. Open items:
- The `head.activity_head` of the published GGUF (LayerNorm + 3 classes) is not in the reference model. Expose it once its meaning is documented.
- `vad_streams_push` batches the streaming steps of several diarization streams (embedding and encoder graphs with a batch dimension, per-stream attention masks); equal to the streams run alone. Per stream and 10 ms frame: Vulkan 340 → 216 µs at 4 streams, `cpu` 16 threads 710 → 600 µs. The encoder's attention over ~300 frames per stream bounds it.
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
  - `vad_streams_push` batches the windows of sliding-window streams (up to 8 per compute; FireRed VAD, 64 streams, 8 threads: 16.4 → 11.8 µs per stream frame).
  - Q8_0 with int8 activations on AVX-512 VNNI (`avx512-vnni` variant): about 2× faster than before and faster than F32.
- **ARM.**
  - Done: an ARMv8.2 (fp16 + dotprod) variant chosen at runtime, NEON register tiles, streaming one-row F32 GEMV, and per-ISA thread-pool tuning (timed spin; few-row GEMMs split only when the work pays for the dispatch, 2 chunks per thread for big.LITTLE).
  - `vad-cpu` beats ggml-cpu on the Genio board and the Pixel in nearly every cell (`bench/results/arm_genio.txt`), typically 1.2–3× and far more for multi-threaded streaming.
  - Closest cases:
    - Single-thread FireRed F32 streaming: at parity on the Genio, ahead on the Pixel's X1 (178 vs 204 µs). It is bound by streaming about 2.3 MB of F32 weights per 10 ms frame.
    - smart-turn F16 single-window streaming on the Genio's A78: 3–7% behind ggml-cpu at 4 and 8 threads (it wins offline, and on the Pixel by 1.6–2×). Routing its many-row GEMMs through the cache-blocked F32 kernel was tried and was slower on the Pixel.
  - To do:
    - SVE variants.
- **Stream vs offline.** FireRed differs by up to 3.6e-7 between streaming and offline on `vad-cpu`, even for F32 files, on x86 and aarch64 (ggml-cpu is exact). Within tolerance; the cause is not located yet. What is known:
  - Through the API it depends on which graph size (T bucket 64/16/4/1) computes a frame: pushing the whole file at once equals `vad_detect`, pushing 1 frame at a time differs from frame 3.
  - It reproduces outside the runtime: one 1-frame graph replayed with its state resident vs one 64-frame graph, both allocated with buffer reuse (`ggml_gallocr`), differ from frame 3. With every node marked as an output (no buffer reuse) they agree exactly, and a replayed step equals a freshly built one from the same state. No GEMM sees an aliased input, and protecting any half of the nodes is not enough, so the suspect is something address-dependent across many steps (e.g. alignment-dependent vectorization in a reduction).
- **AVX-VNNI (VEX) for Q8_0 on AVX2-only x86** (e.g. Alder Lake): the AVX-512 VNNI variant is done; CPUs without AVX-512 still convert int8 weights to F32.
- Silero's LSTM cell fusion was tried (one step for the gates, cell and output after the recurrent GEMM) and dropped: the replaced element-wise steps cost about 0.3 µs of a 17 µs single-thread compute (GEMMs are 16 µs), and offline/stream timings did not change.
