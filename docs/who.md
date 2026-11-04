# who (desert-ant-labs/who): design notes

Status: in progress. This file is the plan and the record of decisions; update it as parts land.

## Model

EEND-M2F speaker diarization, published only as ONNX (8-bit matrix products, the rest F32), Core ML and LiteRT.
Files: `onnx-int8/{encoder_6000,queries_6000,encoder_30000,queries_30000,voice_3256}.onnx` and the sidecars
`who_meta.json`, `who_mel_banks.f32` (23 x 257), `who_voice_banks.f32` (80 x 257), `who_embed.f16`.
License: Desert Ant Labs Source-Available (https://license.desertant.com/).

- **Front end** (host): 16 kHz, frame 400, shift 160, nfft 512, pre-emphasis 0.97, scale 32768, symmetric Hamming
  0.54/0.46, power spectrum -> mel banks -> log (floor 2^-23), the mean over the whole recording removed per band.
  The voice network uses the same front end with 80 bands.
- **Encoder** (`encoder_W`, W = 6000 or 30000 frames of 10 ms): depthwise conv k15 stride 10 (23 bands) + pointwise
  -> 256, LayerNorm, x0.5 scale; 6 Conformer layers at 100 ms (macaron FF with SiLU and x0.5, relative-position
  self-attention with 4 heads of 64 and per-window position tables, GLU conv module with depthwise k49, LayerNorm);
  output `memory` [W/10, 256]. Upsampler: nearest x2 + conv k3 + LN + GELU, nearest x5 + conv k5 + LN + GELU ->
  `frames` [W, 256].
- **Queries** (`queries_W`): Mask2Former decoder, 50 learned queries + query position, 6 post-LN layers
  (masked cross-attention to `memory`, self-attention, ReLU FF). The cross-attention mask comes from the previous
  prediction: mask_embed(decoder_norm(q)) . pairing(frames) at 100 ms, allowed where the logit >= 0, all allowed
  when none is. Heads: presence = Linear(256 -> 1), masks = MLP(256, 3 layers) . frames -> [50, W].
- **Voice** (`voice_3256`): ResNet over the 80-band log-mel, 3256 frames in, 407 columns x 2560 out (8 frames a
  column); `who_embed.f16` maps mean + unbiased std of a lane's solo columns (5120) to a 256-d voice.

## Host algorithm (from who_meta.json)

1. Log-mel of the whole recording; means over the recording.
2. Pieces of the longest window (30000 frames = 300 s) end to end; a piece shorter than a window is padded with its
   last mel frame and uses the shortest window that holds it.
3. Per piece: encoder, queries; a query is a speaker where sigmoid(presence) > 0.8; it talks where
   sigmoid(mask) > 0.5 (rule "mask", pool 1). Each such query of a piece is a lane.
4. One piece: lanes are the speakers. Several: each lane's voice from the columns where it speaks alone (half of
   a column's 8 frames on, no other lane on), if it has at least 1 s of solo speech; voices clustered by average
   linkage on cosine distance with a cut at cosine 0.65; voiceless lanes take the speaker of the voiced lane
   nearest in time; speakers numbered by first appearance.

## Mapping onto vad.h

- Architecture `who`, host driven (like `sortformer`). `n_outputs` = `who.max_speakers` (written by the
  converter, default 16). Output frame f = 10 ms [10f, 10f + 10). Speaker s at frame f = the largest mask
  probability among its lanes there (0 where it has none), so thresholding at 0.5 gives the SDK's turns.
- Speakers past the cap: folded into the nearest-in-time speaker within the cap (the SDK's voiceless rule).
- `vad_detect`: the exact algorithm above.
- Streams: a piece runs when its window is complete (slice selects 60 s or 300 s), with means over the audio so
  far; lanes join incrementally (average-linkage distance to existing speakers below 1 - 0.65, else a new one).
  Not equal to offline; reported, not checked.

## Conversion

`tools/convert_who_onnx.py DIR out.gguf [--outtype f32|q8_0]`: MatMulNBits (8 bit, block 128, no zero points:
implicit 128) dequantized to F32 by default, the window-specific tables stored per window.

## Findings from the ONNX graphs (tools/who_onnx.py)

- The NumPy forward of all three graphs matches ONNX Runtime (encoder 2.6e-6 / 9e-6 for the 60 s / 300 s windows,
  queries 1e-5, voice 2.8e-6).
- One weight set serves both windows: the queries weights are identical, the encoders differ only in the position
  keys, and the 600-row table is the first 600 rows of the 3000-row one. sin/cos tables are sin/cos(t * 10000^(-i/128)).
- The first mask (before decoder layer 0) uses 8-bit copies of the mask MLP (`mask_embed_q.*`); every later use, and
  the mask head, uses the float weights.
- Encoder attention: scores = ((q + content_bias) . k + rot((q + pos_bias) W_pos) . keys) / 8, where rot mixes the
  two 128-halves with sin/cos of the query position. The upsampler is two transposed convs (zero insertion);
  output frames 7..6004 and the last one repeated twice.
- Scales are F32, so Q8_0 (fp16 scales) is not an exact repacking: F32 is the reference file.

## Plan

1. Converter + Python reference (ORT graphs + host algorithm), checked against the SDK's golden fixtures.
2. ggml graphs: encoder, queries, voice, each against ORT.
3. Host session (offline, then streaming), tests with a > 300 s fixture, docs.
