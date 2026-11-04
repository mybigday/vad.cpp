"""Convert pipecat-ai smart-turn-v3 (fp32 ONNX, e.g. smart-turn-v3.2-gpu.onnx) to vad.cpp GGUF.

usage: convert_smart_turn_onnx.py smart-turn-v3.2-gpu.onnx out.gguf VERSION [--outtype f32|f16|bf16]

Only the fp32 ("-gpu") exports are supported; the int8 "-cpu" exports are
quantized ONNX graphs. Tensor names follow the PyTorch SmartTurnV3Model
state_dict; Linear weights are stored as (out, in) like PyTorch. The Whisper
log-mel filterbank (slaney, 80 x 201) is stored as frontend.mel_filters.
"""
import sys

from gguf_common import conv_to_window_layout, pop_outtype, write_gguf
import numpy as np
import onnx
from onnx import numpy_helper


def slaney_mel_filters(n_mels=80, n_fft=400, sr=16000, fmin=0.0, fmax=8000.0):
    """librosa/transformers mel_filter_bank(norm="slaney", mel_scale="slaney")."""
    def hz_to_mel(f):
        f = np.asarray(f, np.float64)
        m = 3.0 * f / 200.0
        log = f >= 1000.0
        return np.where(log, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) * (27.0 / np.log(6.4)), m)

    def mel_to_hz(m):
        m = np.asarray(m, np.float64)
        f = 200.0 * m / 3.0
        log = m >= 15.0
        return np.where(log, 1000.0 * np.exp(np.log(6.4) / 27.0 * (m - 15.0)), f)

    n_freq = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr // 2, n_freq)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    slopes = mel_f[None, :] - fft_freqs[:, None]          # (n_freq, n_mels + 2)
    down = -slopes[:, :-2] / fdiff[:-1]
    up = slopes[:, 2:] / fdiff[1:]
    fb = np.maximum(0, np.minimum(down, up))               # (n_freq, n_mels)
    enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
    return (fb * enorm[None, :]).T                         # (n_mels, n_freq)


def main():
    outtype = pop_outtype(sys.argv)
    src, dst, version = sys.argv[1:4]
    m = onnx.load(src)
    inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    if any(n.op_type in ("DequantizeLinear", "QuantizeLinear", "MatMulInteger", "QLinearMatMul", "DynamicQuantizeLinear")
           for n in m.graph.node):
        sys.exit("quantized model: use the fp32 (-gpu) export")

    W = {k[len("inner."):]: v for k, v in inits.items() if k.startswith("inner.")}
    # Anonymous MatMul weights (dynamo export): name each by the bias added to its
    # output (q_proj.bias -> q_proj.weight). k_proj has no bias: it is the
    # bias-less MatMul that follows q_proj.
    consumers = {}
    for node in m.graph.node:
        for x in node.input:
            consumers.setdefault(x, []).append(node)
    last = None
    for node in m.graph.node:
        if node.op_type != "MatMul":
            continue
        w = [i for i in node.input if i in inits and not i.startswith("inner.") and inits[i].ndim == 2]
        if not w:
            continue
        name = None
        for c in consumers.get(node.output[0], []):
            if c.op_type == "Add":
                b = [i for i in c.input if i.startswith("inner.") and i.endswith(".bias")]
                if b:
                    name = b[0][len("inner."):-len(".bias")] + ".weight"
        if name is None:
            if last is None or not last.endswith("q_proj.weight"):
                sys.exit(f"cannot name MatMul weight {w[0]}")
            name = last.replace("q_proj", "k_proj")
        W[name] = inits[w[0]].T.copy()                     # (in, out) -> PyTorch (out, in)
        last = name

    # small biases are sometimes constants instead of initializers
    for node in m.graph.node:
        if node.op_type == "Constant" and node.output[0].startswith("inner."):
            W[node.output[0][len("inner."):]] = numpy_helper.to_array(node.attribute[0].t)

    n_layers = 1 + max(int(k.split(".")[2]) for k in W if k.startswith("encoder.layers."))
    d = W["encoder.conv1.weight"].shape[0]
    expected = ["encoder.conv1.weight", "encoder.conv2.weight", "encoder.embed_positions.weight", "encoder.layer_norm.weight",
                "pool_attention.0.weight", "pool_attention.2.weight", "pool_attention.2.bias",
                "classifier.0.weight", "classifier.4.weight", "classifier.6.weight", "classifier.6.bias"]
    for l in range(n_layers):
        for p in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            expected.append(f"encoder.layers.{l}.self_attn.{p}.weight")
        expected += [f"encoder.layers.{l}.fc1.weight", f"encoder.layers.{l}.fc2.weight"]
    missing = [k for k in expected if k not in W]
    if missing:
        sys.exit(f"missing tensors: {missing}")
    # graph-ready tensors (docs/gguf.md)
    T = {"fe.mel": slaney_mel_filters().astype(np.float32),
         "conv1.w": conv_to_window_layout(W["encoder.conv1.weight"]), "conv1.b": W["encoder.conv1.bias"],
         "conv2.w": conv_to_window_layout(W["encoder.conv2.weight"]), "conv2.b": W["encoder.conv2.bias"],
         "pos": W["encoder.embed_positions.weight"]}
    for l in range(n_layers):
        p, q = f"encoder.layers.{l}.", f"l{l}."
        T.update({q + "ln1.w": W[p + "self_attn_layer_norm.weight"], q + "ln1.b": W[p + "self_attn_layer_norm.bias"],
                  q + "wq": W[p + "self_attn.q_proj.weight"], q + "bq": W[p + "self_attn.q_proj.bias"],
                  q + "wk": W[p + "self_attn.k_proj.weight"],
                  q + "wv": W[p + "self_attn.v_proj.weight"], q + "bv": W[p + "self_attn.v_proj.bias"],
                  q + "wo": W[p + "self_attn.out_proj.weight"], q + "bo": W[p + "self_attn.out_proj.bias"],
                  q + "ln2.w": W[p + "final_layer_norm.weight"], q + "ln2.b": W[p + "final_layer_norm.bias"],
                  q + "fc1.w": W[p + "fc1.weight"], q + "fc1.b": W[p + "fc1.bias"],
                  q + "fc2.w": W[p + "fc2.weight"], q + "fc2.b": W[p + "fc2.bias"]})
    T.update({"lnf.w": W["encoder.layer_norm.weight"], "lnf.b": W["encoder.layer_norm.bias"],
              "pa0.w": W["pool_attention.0.weight"], "pa0.b": W["pool_attention.0.bias"],
              "pa2.w": W["pool_attention.2.weight"], "pa2.b": W["pool_attention.2.bias"],
              "c0.w": W["classifier.0.weight"], "c0.b": W["classifier.0.bias"],
              "c1.w": W["classifier.1.weight"], "c1.b": W["classifier.1.bias"],
              "c4.w": W["classifier.4.weight"], "c4.b": W["classifier.4.bias"],
              "c6.w": W["classifier.6.weight"], "c6.b": W["classifier.6.bias"]})
    write_gguf(dst, "smart_turn", f"Smart Turn {version}",
               {"version": version.split(".")[0], "preprocessor.sample_rate": 16000, "preprocessor.n_mels": 80, "preprocessor.n_fft": 400, "preprocessor.hop": 160,
                "window_samples": 8 * 16000, "n_layers": n_layers, "d_model": d, "n_heads": 6,
                "n_ctx": W["encoder.embed_positions.weight"].shape[0]},
               T, outtype=outtype, version=version, license="BSD-2-Clause",
               source_url="https://huggingface.co/pipecat-ai/smart-turn-v3")
    print(f"wrote {dst}: {n_layers} layers, d_model {d}, {len(T)} tensors")


if __name__ == "__main__":
    main()
