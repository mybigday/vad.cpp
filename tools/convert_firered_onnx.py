"""Convert FireRedVAD (fireredvad_stream_vad.onnx or fireredvad_vad.onnx, + cmvn.ark) to vad.cpp GGUF.

usage: convert_firered_onnx.py fireredvad_{stream_,}vad.onnx cmvn.ark out.gguf [--outtype f32|f16|bf16]

The Stream-VAD model is causal (streaming). The non-streaming VAD model adds
lookahead filters (each layer sees N2 future frames): offline use only.

Tensor names follow the PyTorch DetectModel state_dict (without the "model."
prefix); Linear weights are stored as (out, in) like PyTorch.
"""
import re
import sys

from gguf_common import pop_outtype, write_gguf
import numpy as np
import onnx
from onnx import numpy_helper

from firered_common import read_kaldi_cmvn

outtype = pop_outtype(sys.argv)
src, cmvn_path, dst = sys.argv[1:4]
m = onnx.load(src)
inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}

W = {}
for name, arr in inits.items():
    if name.startswith("model.") and arr.dtype.kind == "f":
        W[name[len("model."):]] = arr
for node in m.graph.node:
    if node.op_type != "MatMul":
        continue
    w = [i for i in node.input if i in inits]
    if not w:
        continue
    # /model/dfsmn/fsmns.3/fc1/fc1.0/MatMul -> dfsmn.fsmns.3.fc1.0.weight
    path = node.name.split("/")[2:-1]          # drop "", "model", "MatMul"
    parts = []
    for p in path:
        if parts and p.startswith(parts[-1] + "."):
            p = p[len(parts[-1]) + 1:]         # "fc1/fc1.0" -> "fc1.0"
        parts.append(p)
    name = ".".join(parts) + ".weight"
    W[name] = inits[w[0]].T.copy()              # ONNX (in, out) -> PyTorch (out, in)

R = 1 + len({k.split(".")[2] for k in W if k.startswith("dfsmn.fsmns.")})
M = len({k.split(".")[2] for k in W if k.startswith("dfsmn.dnns.")})
H, D = W["dfsmn.fc1.0.weight"].shape
P = W["dfsmn.fc2.0.weight"].shape[0]
N1 = W["dfsmn.fsmn1.lookback_filter.weight"].shape[2]
N2 = W["dfsmn.fsmn1.lookahead_filter.weight"].shape[2] if "dfsmn.fsmn1.lookahead_filter.weight" in W else 0
for c in m.graph.node:
    if c.op_type == "Conv":
        dil = [a for a in c.attribute if a.name == "dilations"][0].ints
        assert list(dil) == [1], "lookback/lookahead strides != 1 are not supported"

means, istd = read_kaldi_cmvn(cmvn_path)
assert len(means) == D

# graph-ready tensors (docs/gguf.md): CMVN folded into fc1, lookback + lookahead taps merged
fc1, b1 = W["dfsmn.fc1.0.weight"].astype(np.float64), W["dfsmn.fc1.0.bias"].astype(np.float64)
T = {"fc1.w": fc1 * istd[None, :], "fc1.b": b1 - fc1 @ (means * istd),
     "fc2.w": W["dfsmn.fc2.0.weight"], "fc2.b": W["dfsmn.fc2.0.bias"]}


def taps(prefix):
    lb = W[prefix + "lookback_filter.weight"][:, 0, :]                      # (P, N1)
    if N2 == 0:
        return lb
    return np.concatenate([lb, W[prefix + "lookahead_filter.weight"][:, 0, :]], axis=1)   # (P, N1 + N2)


T["fsmn0.w"] = taps("dfsmn.fsmn1.")
for i in range(R - 1):
    p = f"dfsmn.fsmns.{i}."
    T[f"blk{i}.fc1.w"] = W[p + "fc1.0.weight"]
    T[f"blk{i}.fc1.b"] = W[p + "fc1.0.bias"]
    T[f"blk{i}.fc2.w"] = W[p + "fc2.weight"]
    T[f"fsmn{i + 1}.w"] = taps(p + "fsmn.")
for i in range(M):
    T[f"dnn{i}.w"] = W[f"dfsmn.dnns.{3 * i}.weight"]
    T[f"dnn{i}.b"] = W[f"dfsmn.dnns.{3 * i}.bias"]
T["out.w"] = W["out.weight"]
T["out.b"] = W["out.bias"]

write_gguf(dst, "firered", "FireRedVAD Stream-VAD" if N2 == 0 else "FireRedVAD VAD",
           {"version": "stream" if N2 == 0 else "vad", "preprocessor.sample_rate": 16000, "preprocessor.frame_length": 400, "preprocessor.frame_shift": 160,
            "preprocessor.num_mel_bins": 80, "n_blocks": R, "n_dnn": M, "hidden": H, "proj": P, "lookback": N1, "lookahead": N2},
           T, outtype=outtype,
           license="Apache-2.0", source_url="https://github.com/FireRedTeam/FireRedVAD")
print(f"wrote {dst}: R={R} M={M} D={D} H={H} P={P} N1={N1} N2={N2}, {len(T)} tensors")
