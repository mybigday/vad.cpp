# Benchmarks

`vad-bench` times one model file on one or more devices. All times are µs per output frame (per stream for the multi-stream rows).

```sh
cmake -S . -B build -DGGML_NATIVE=ON && cmake --build build -j --target vad-bench
build/vad-bench -m models/silero-v6.2.gguf -f assets/test_16k.wav -d cpu,ggml-cpu -t 4 -r 3 -s 64
```

- `-d`: comma-separated devices (`cpu` = vad-cpu kernels, `ggml-cpu`, `Vulkan0`, ...).
- `-t`: CPU threads. `-r`: repetitions (the median is reported). `-s`: stream counts for the multi-stream case.
- `--windows 0`: skip the sliding-window cases of lookahead models.

Cases:
- **stream**: one hop per `vad_stream_push`, which measures streaming latency.
- **offline**: `vad_detect` over the whole file.
- **window** (lookahead models only): sliding-window streams.
- **end**: the probability for the end of the file only.
- **streams S=N**: N independent streams, one hop each, through `vad_streams_push`.

`scripts/bench-matrix.py build/vad-bench OUT.txt --threads 1,4,8,16` runs `cpu` against `ggml-cpu` for every model file, case and thread count. It prints the ratio ggml-cpu / cpu (> 1 means vad-cpu is faster) and marks slower cells.

## Results

| file | machine |
|---|---|
| [results/vad_cpu_vs_ggml_x86.txt](results/vad_cpu_vs_ggml_x86.txt) | AMD Ryzen AI Max+ 395 (Zen 5), every model, precision and 1/4/8/16 threads |
| [results/nemotron_x86.txt](results/nemotron_x86.txt) | same machine, Nemotron-3-Diarization |
| [results/arm_genio.txt](results/arm_genio.txt) | MediaTek Genio (Cortex-A55 + A78), aarch64 Linux |

## Notes from the whisper.cpp baseline

The project started from whisper.cpp's Silero VAD, which differs from the official ONNX model by up to 0.23 in probability. There are two causes:
- It reflect-pads both sides of every chunk instead of prepending the previous chunk's last 64 samples.
- `ggml_conv_1d` builds an F16 im2col even for F32 kernels, which alone accounts for 4e-3 of the error.

Its streaming API also rebuilds the graph on every call, so more threads make streaming slower.

On these small graphs, ggml-cpu's time goes into small-shape MUL_MATs, short rows that don't vectorize, and about 0.1 µs of fixed cost per node. Graph management itself costs almost nothing. The `vad-cpu` backend (`src/ggml-vad-cpu`) addresses this. It compiles each graph once into a plan of fused GEMM steps over pre-packed weights, uses its own small-shape kernels, and delegates any op it doesn't support to ggml-cpu.
