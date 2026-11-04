# Platforms

## Linux / macOS / Windows (native)

```sh
cmake -B build [-DGGML_VULKAN=ON | -DGGML_CUDA=ON | -DGGML_METAL=ON ...] && cmake --build build -j
(cd build && ctest)
VAD_TEST_DEVICES=Vulkan0 build/tests/test_models
```

## ARM Linux (cross-compiled, static)

```sh
cmake -S . -B build-arm -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-linux-gnu.cmake -DBUILD_SHARED_LIBS=OFF \
      -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod+fp16 -DGGML_OPENMP=OFF -DGGML_BACKEND_DL=OFF
cmake --build build-arm -j
# on the device, with the models and references copied over:
VAD_TEST_DATA=/tmp/vad/models VAD_ASSETS=/tmp/vad/assets ./test_models
```

Verified on a MediaTek Genio board (Yocto Linux, Cortex-A55 + A78). All tests pass with the `vad-cpu` kernels.
- **Kernel variants:** on aarch64 Linux/Android/Apple, `vad-cpu` builds a baseline NEON variant and an ARMv8.2 (fp16 + dotprod) variant, chosen at runtime (HWCAP on Linux, `hw.optional.arm.*` sysctls on Apple). The ARMv8.2 variant runs F16 weights with fp16 activations and Q8_0 weights with int8 activations (see "Weight precision" in the README). F32 weights are computed in F32 on both.
- Benchmarks are in `bench/results/arm_genio.txt`. On big.LITTLE parts, pin with `taskset` for stable timings.

## Android (NDK)

```sh
cmake -S . -B build-android -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DCMAKE_BUILD_TYPE=Release -DGGML_OPENMP=OFF -DGGML_NATIVE=OFF
cmake --build build-android -j
# copy the binaries, libvad.so, third_party/ggml/src/libggml*.so, models and assets to /data/local/tmp/vad, then:
LD_LIBRARY_PATH=/data/local/tmp/vad/lib VAD_TEST_DATA=/data/local/tmp/vad/models VAD_ASSETS=/data/local/tmp/vad/assets ./test_models
```

Verified on a Pixel 7 Pro (Android 14, Tensor G2: 2× Cortex-X1, 2× A78, 4× A55). All tests pass, and the ARMv8.2 variant is selected at runtime. The static ARM Linux build above also runs there unchanged. Spot benchmarks are in `bench/results/arm_genio.txt`. The phone throttles under sustained load, so compare builds interleaved and take minimums.

## macOS / iOS

The native build above works on Apple Silicon. For iOS:

```sh
cmake -S . -B build-ios -G Ninja -DCMAKE_SYSTEM_NAME=iOS -DCMAKE_OSX_ARCHITECTURES=arm64 \
      -DCMAKE_OSX_DEPLOYMENT_TARGET=15.1 -DCMAKE_BUILD_TYPE=Release
cmake --build build-ios
```

iOS compilers target ARMv8.0 by default, so the ARMv8.2 variant (picked at runtime) matters there: Q8_0 models run about 2x faster with it. Universal (multi-arch) builds get the baseline kernels only.

Verified on an M1 Mac mini (macOS 26): all tests pass. Benchmarks are in `bench/results/apple_m1.txt`. vad-cpu beats ggml-cpu on every model there. Metal is much slower for streaming (about 1 ms per push); it only helps the non-streaming FireRed VAD offline or with many streams, on a single CPU thread.

## WebAssembly (node and browsers)

Requires [emsdk](https://github.com/emscripten-core/emsdk) (`source ~/emsdk/emsdk_env.sh`).

**CPU (WASM SIMD128), tests under node:**

```sh
emcmake cmake -S . -B build-wasm -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_BACKEND_DL=OFF
cmake --build build-wasm -j
for t in test_kernels test_segments test_models test_turn; do node build-wasm/tests/$t.js; done
```

- Without `-DVAD_WASM_THREADS=ON`, the build is single-threaded and `n_threads` is ignored.
- With threads, browsers need SharedArrayBuffer, i.e. COOP/COEP headers.

**Browser module with WebGPU:**

```sh
emcmake cmake -S . -B build-web -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_BACKEND_DL=OFF \
      -DGGML_WEBGPU=ON -DVAD_BUILD_TESTS=OFF -DVAD_BUILD_BENCH=OFF -DVAD_BUILD_EXAMPLES=OFF
cmake --build build-web -j --target vad-web
```

`build-web/vad-web.mjs` is an ES module (`createVad()`) exporting the C API.
- With WebGPU, the calls are JSPI exports: they return Promises, so `await` them.
- `examples/web/test.html` shows the usage. It runs every model on every device against the official references.

Running the browser test:

```sh
python3 -m http.server 8765 &                 # repository root
node scripts/web-test.mjs build-web 8765      # separate headless Chrome with WebGPU enabled, prints the results
```

- Precision-critical front-end ops (power spectra up to 1e11) are pinned to the CPU on accelerators.
- On WebGPU, probabilities deviate from the references by at most 4e-3, the same as on Vulkan.
