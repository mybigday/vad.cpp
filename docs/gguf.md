# vad.cpp GGUF files

## Common rules

Metadata uses the usual GGUF layout:

| prefix | contents |
|---|---|
| `general.*` | `architecture` (`silero`, `firered`, `smart_turn`), `name`, `version`, `license`, `source.url`, `file_type` (from the type most ≥ 2-D tensors use, written by the converters and recomputed by `scripts/quantize-gguf.py`), and `quantization_version` (only when a block-quantized tensor is present) |
| `<arch>.*` | `version` (architecture variant, e.g. `silero.version = v6`, like `sortformer.version = v3`), hyperparameters, and the audio front end under `<arch>.preprocessor.*` |

There are no vad.cpp-specific keys outside these two prefixes.

Other rules:
- Converters write F32 by default; `--outtype f16|bf16` stores every ≥ 2-D tensor at that type (1-D tensors stay F32). Block quantization is done afterwards with `scripts/quantize-gguf.py`.
- Which tensors an architecture uses outside mul_mat (the `F` kind below) is part of the architecture, not of the file. The loader rejects a non-float type there and names the tensor; the quantizer knows them per architecture (`ELEMENTWISE` in `tools/gguf_common.py`) and leaves them float unless told otherwise.
- **Tensors are graph-ready.** Every re-layout or fold (conv weight order, summed biases, CMVN, merged FSMN taps) is done by the converter. The runtime uses each tensor as stored, at its stored type.
- **Shapes** follow ggml order (`ne[0]` is the fastest-varying dimension). A PyTorch tensor of shape `(a, b, c)` is stored as `[c, b, a]` with identical memory.
- **Mul_mat weights** (`W` below) may be any type the backends support: F32, F16, BF16, Q8_0, …. The other tensors (`F`) must be F32, F16 or BF16. They are widened to F32 at load.
- **Conv1d weights** are stored as `(Cout, K, Cin)` instead of PyTorch's `(Cout, Cin, K)`. This matches the graph's sliding `[K * Cin]` window.

whisper.cpp `ggml-silero-*.bin` files are also accepted. The loader re-lays them out into the silero tensors below, keeping their F16 weights.

## silero

Converter: `tools/convert_silero_onnx.py silero_vad.onnx out.gguf VERSION`. Only the 16 kHz branch is stored.

Metadata:

| key | type | value |
|---|---|---|
| `preprocessor.sample_rate` | u32 | 16000 |
| `chunk_size` | u32 | 512 |
| `context_size` | u32 | 64 |

| tensor | PyTorch shape | kind | |
|---|---|---|---|
| `stft.basis` | (258, 256) | W | STFT forward basis |
| `enc{0..3}.w` | (Cout, 3, Cin) | W | encoder convs: 129→128, 128→64, 64→64, 64→128 |
| `enc{0..3}.b` | (Cout) | F | |
| `lstm.w_ih`, `lstm.w_hh` | (512, 128) | W | |
| `lstm.b` | (512) | F | `bias_ih + bias_hh` |
| `dec.w` | (1, 128) | W | |
| `dec.b` | (1) | F | |

## firered

Converter: `tools/convert_firered_onnx.py fireredvad_{stream_,}vad.onnx cmvn.ark out.gguf`. The inputs are in FireRedVAD's `pretrained_models/onnx_models`.

Metadata (u32):

| key | value |
|---|---|
| `preprocessor.sample_rate` | 16000 |
| `preprocessor.frame_length` | 400 |
| `preprocessor.frame_shift` | 160 |
| `preprocessor.num_mel_bins` | 80 |
| `n_blocks` | R |
| `n_dnn` | M |
| `hidden` | H |
| `proj` | P |
| `lookback` | N1 |
| `lookahead` | N2 (0 for Stream-VAD) |

| tensor | PyTorch shape | kind | |
|---|---|---|---|
| `fc1.w` | (H, 80) | W | CMVN folded in: `W·diag(istd)` |
| `fc1.b` | (H) | F | `b − W·(mean·istd)` |
| `fc2.w` | (P, H) | W | |
| `fc2.b` | (P) | F | |
| `fsmn{0..R-1}.w` | (P, N1 + N2) | F | lookback taps (frames t−N1+1 … t), then lookahead taps (t+1 … t+N2) |
| `blk{0..R-2}.fc1.w` | (H, P) | W | |
| `blk{0..R-2}.fc1.b` | (H) | F | |
| `blk{0..R-2}.fc2.w` | (P, H) | W | |
| `dnn{0..M-1}.w` | (H, P or H) | W | |
| `dnn{0..M-1}.b` | (H) | F | |
| `out.w` | (1, H) | W | |
| `out.b` | (1) | F | |

The Kaldi fbank basis and mel filters are computed at load time.

Semantics:
- The first 240 samples are warm-up. Probability i covers samples [160 i, 160 i + 400), matching the official frames (`snip_edges`).
- With N2 > 0 the model has 1.6 s of lookahead. It streams through a sliding window (see the README).

## smart_turn

Converter: `tools/convert_smart_turn_onnx.py smart-turn-v3.2-gpu.onnx out.gguf v3.2`. It needs an fp32 `-gpu` export; the int8 `-cpu` exports are rejected. The ONNX MatMul weights are anonymous, so the converter names each one after the bias added to its output. `k_proj` has no bias; it is the bias-less MatMul that follows `q_proj`.

Metadata:

| key | type | value |
|---|---|---|
| `preprocessor.sample_rate` | u32 | 16000 |
| `preprocessor.n_mels` | u32 | 80 |
| `preprocessor.n_fft` | u32 | 400 |
| `preprocessor.hop` | u32 | 160 |
| `window_samples` | u32 | 128000 |
| `n_layers` | u32 | L |
| `d_model` | u32 | D |
| `n_heads` | u32 | |
| `n_ctx` | u32 | 400 |

| tensor | PyTorch shape | kind | |
|---|---|---|---|
| `fe.mel` | (80, 201) | W | slaney mel filters (identical to transformers' `WhisperFeatureExtractor`) |
| `conv1.w` / `conv1.b` | (D, 3, 80) / (D) | W / F | |
| `conv2.w` / `conv2.b` | (D, 3, D) / (D) | W / F | stride 2 |
| `pos` | (400, D) | F | positions |
| `l{i}.ln1.w/.b`, `l{i}.ln2.w/.b` | (D) | F | |
| `l{i}.wq`, `l{i}.wk`, `l{i}.wv`, `l{i}.wo` | (D, D) | W | |
| `l{i}.bq`, `l{i}.bv`, `l{i}.bo` | (D) | F | |
| `l{i}.fc1.w` / `.b` | (4D, D) / (4D) | W / F | |
| `l{i}.fc2.w` / `.b` | (D, 4D) / (D) | W / F | |
| `lnf.w` / `lnf.b` | (D) | F | |
| `pa0.w` / `pa0.b` | (256, D) / (256) | W / F | attention pooling |
| `pa2.w` / `pa2.b` | (1, 256) / (1) | W / F | |
| `c0.w` / `c0.b` | (256, D) / (256) | W / F | classifier |
| `c1.w` / `c1.b` | (256) | F | LayerNorm |
| `c4.w` / `c4.b` | (64, 256) / (64) | W / F | |
| `c6.w` / `c6.b` | (1, 64) / (1) | W / F | |

## sortformer

NVIDIA's Nemotron-3-Diarization (Streaming Sortformer v3) uses the GGUF schema NVIDIA publishes with the model (`Nemotron-3-Diarization.q8_0.gguf`, from NeMo-Speech.cpp). vad.cpp loads that file as is. `tools/convert_sortformer.py MODEL_DIR out.gguf [--outtype f32|f16|bf16]` writes the same schema from the Hugging Face checkpoint (`config.json`, `processor_config.json`, `model.safetensors`) at the original F32 precision.

Unlike vad.cpp's own architectures, the tensors keep NeMo's layout. The loader makes one exact re-layout: the sub-pixel conv weight goes from (Cout, Cin, 3) to the graph's window order. It is used at its stored type, so it must be a float type.

Metadata (prefix `sortformer.`). vad.cpp checks the keys listed below. The other keys of the published file are kept for compatibility.

| key | value |
|---|---|
| `version` | `v3` |
| `encoder.type` / `encoder.subsampling_type` | `transformer_rope` / `feature_stacking` |
| `encoder.d_model`, `n_layers`, `n_heads`, `d_ff` | 512, 31, 8, 2048 |
| `encoder.subsampling_factor`, `feat_in` | 8, 128 |
| `encoder.qkv_bias`, `use_bias`, `rope_base`, `rotary_fraction` | false, true, 10000, 1.0 |
| `transformer.hidden_size`, `num_speakers`, `upsample_factor`, `high_resolution` | 192, 8, 8, true |
| `preprocessor.sample_rate`, `n_fft`, `window_size`, `window_stride`, `features` | 16000, 512, 0.025, 0.01, 128 |
| `preprocessor.preemph`, `log_zero_guard`, `normalize` | 0.97, 2^-24, `NA` |
| `scoring.*` | speaker-cache compression: `spkcache_sil_frames_per_spk`, `pred_score_threshold`, `scores_boost_latest`, `strong_boost_rate`, `weak_boost_rate`, `min_pos_scores_rate` |
| `streaming.spkcache_len` | 264 |

The stream chunking comes from the stream parameters. vad.cpp does not read it from `streaming.*` (see below).

| tensor | shape (PyTorch) | |
|---|---|---|
| `preprocessor.fb` | (128, 257) | mel filter bank (W) |
| `encoder.pre_encode.proj.weight` | (512, 1024) | 8 stacked frames → 512 (W) |
| `encoder.embed_norm.*`, `encoder.final_norm.*` | (512) | LayerNorms |
| `encoder.layers.{i}.norm1.*`, `norm2.*` | (512) | |
| `encoder.layers.{i}.attn.w_qkv.weight` | (1536, 512) | q, k, v stacked (W) |
| `encoder.layers.{i}.attn.out_proj.weight` / `.bias` | (512, 512) / (512) | |
| `encoder.layers.{i}.ffn.net.0.*`, `ffn.net.3.*` | (2048, 512), (512, 2048) | GELU MLP |
| `encoder_proj.*` | (192, 512) | |
| `subpixel_upsample.weight` / `.bias` | (1536, 192, 3) / (1536) | conv k3 → 8 frames of 192 (float type) |
| `head.first_hidden_to_hidden.*`, `head.single_hidden_to_spks.*` | (192, 192), (8, 192) | ReLU-Linear-ReLU-Linear |
| `learnable_sil_emb` | (512) | silence embedding of the speaker cache (F) |

The published file also has `head.activity_head.*`. That head is not part of the reference model and is unused.

Semantics:
- Output frame i covers 10 ms and holds 8 speaker probabilities. Speakers are numbered by first appearance. `n_samples / 160` frames are emitted.
- **`vad_detect`** runs the reference offline forward: chunks of 340 encoder frames (27.2 s) with 40 frames (3.2 s) of look-ahead, a FIFO of 40, and a cache update period of 300.
- **Streams** run one step per chunk of `slice_samples` (in 80 ms encoder frames, default 9) with `lookahead_samples` of look-ahead. The default is the model card's preset for the chunk (9 → 4, 6 → 2, 3 → 1), otherwise half a chunk; the maximum is 40.
  - The FIFO is 264 frames with update period 222. Chunks of 340 frames or more use the offline values instead.
  - The last chunk is scored whole, without look-ahead, as in the reference processor.
