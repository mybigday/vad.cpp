"""desert-ant-labs/who ONNX export -> vad.cpp GGUF (docs/who.md, docs/gguf.md).

usage: convert_who_onnx.py ONNX_DIR out.gguf [--outtype f32|f16|bf16] [--max-speakers N]

ONNX_DIR is the `onnx-int8/` folder of https://huggingface.co/desert-ant-labs/who (the five graphs and the
four sidecars). The 8-bit matrix weights are dequantized (their scales are F32, so F32 is the exact file).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_common import conv_to_window_layout, pop_outtype, write_gguf  # noqa: E402
import who_onnx  # noqa: E402

argv = list(sys.argv)
outtype = pop_outtype(argv)
max_speakers = 16
if "--max-speakers" in argv:
    i = argv.index("--max-speakers")
    max_speakers = int(argv[i + 1])
    del argv[i:i + 2]
if len(argv) != 3:
    sys.exit(__doc__)
src, dst = Path(argv[1]), argv[2]

meta = json.loads((src / "who_meta.json").read_text())
assert meta["schema"] == 2 and meta["model"]["architecture"] == "eend_m2f", meta["model"]
windows = sorted(meta["windows"], key=lambda w: w["frames"])
longest = windows[-1]
enc = who_onnx.load_encoder(src / longest["encoder"]["compiled"])            # key tables for the longest window
for w in windows[:-1]:                                                        # shorter windows: the same weights,
    e = who_onnx.load_encoder(src / w["encoder"]["compiled"])                 # key tables a prefix of the longest's
    for k, v in e.items():
        ref = enc[k][:v.shape[0]] if k.endswith(".keys") else enc[k]
        assert np.array_equal(v, ref), f"{w['encoder']['compiled']}: {k} differs"
dec = who_onnx.load_queries(src / longest["queries"]["compiled"])
for w in windows[:-1]:
    q = who_onnx.load_queries(src / w["queries"]["compiled"])
    assert q.keys() == dec.keys() and all(np.array_equal(q[k], dec[k]) for k in q), w["queries"]["compiled"]
voice = who_onnx.load_voice(src / meta["voice"]["compiled"])
mel = who_onnx.load_sidecar_banks(src / meta["frontend"]["banks"])
vmel = who_onnx.load_sidecar_banks(src / meta["voice"]["frontend"]["banks"])
ew, eb = who_onnx.load_sidecar_embed(src / meta["voice"]["embedding"]["weights"])

T = {"fe.mel": mel, "voice.fe.mel": vmel, "embed.weight": ew, "embed.bias": eb}
# graph-ready layouts (src/models/who.cpp): 1x1 convs as matrices, depthwise taps [C, K], the position
# projection transposed per head, the upsampler kernels in the [K * Cin] window layout
for k, v in enc.items():
    if k.endswith(("pointwise.weight", "inward.weight", "outward.weight")):
        v = v[:, :, 0]
    elif k.endswith("depthwise.weight"):
        v = v[:, 0, :]
    elif k.endswith(".w_pos"):
        v = np.ascontiguousarray(v.transpose(0, 2, 1))                       # [heads, 256, 64]
    elif k.endswith(("pos_bias", "content_bias")):
        v = v[:, 0, :]                                                        # [heads, 64]
    elif k in ("up.first.weight", "up.second.weight"):
        v = conv_to_window_layout(v)                                          # [Cout, K, Cin]
    T["enc." + k] = v
dec["pairing"] = dec["pairing"][:, 0, :]                                      # [256, 10]
for k, v in dec.items():
    T["dec." + k] = v
for k, v in voice.items():
    T["voice." + k] = v
for k in T:
    assert len(k) < 64, k

fe, vf, d, j = meta["frontend"], meta["voice"]["frontend"], meta["decision"], meta["join"]
assert fe["window"] == vf["window"] == "hamming_symmetric_0.54_0.46" and d["rule"] == "mask" and d["pool"] == 1
assert fe["mean"] == vf["mean"] == "removed_over_the_recording" and j["kind"] == "cluster" and j["linkage"] == "average"
assert j["metric"] == "cosine" and j["voiceless"] == "fold"
kv = {
    "preprocessor.sample_rate": meta["audio"]["sample_rate"],
    "preprocessor.frame_length": fe["frame_samples"], "preprocessor.frame_shift": fe["shift_samples"],
    "preprocessor.n_fft": fe["n_fft"], "preprocessor.num_mel_bins": fe["n_mels"],
    "preprocessor.preemphasis": float(fe["preemphasis"]), "preprocessor.scale": float(fe["scale"]),
    "preprocessor.log_floor": float(fe["log_floor"]),
    "dim": meta["model"]["dim"], "queries": meta["model"]["queries"], "n_layers": who_onnx.LAYERS,
    "n_heads": who_onnx.HEADS, "downsample": meta["frames"]["downsample"],
    "windows": ",".join(str(w["frames"]) for w in windows),
    "presence_threshold": float(d["presence_threshold"]), "mask_threshold": float(d["mask_threshold"]),
    "voice.num_mel_bins": vf["n_mels"], "voice.mel_frames": meta["voice"]["mel_frames"],
    "voice.margin_mel_frames": meta["voice"]["margin_mel_frames"],
    "voice.mel_frames_per_column": meta["voice"]["mel_frames_per_column"],
    "voice.columns_dim": meta["voice"]["columns_dim"],
    "join.cosine": float(j["cosine"]), "join.min_solo_s": float(j["min_solo_s"]),
    "max_speakers": max_speakers,
}
write_gguf(dst, "who", "who", kv, T, outtype=outtype, version=meta["model"]["id"],
           license="Desert Ant Labs Source-Available", source_url="https://huggingface.co/desert-ant-labs/who")
n = sum(v.size for v in T.values())
print(f"wrote {dst}: {len(T)} tensors, {n / 1e6:.1f} M values, windows {kv['windows']}, max_speakers {max_speakers}")
