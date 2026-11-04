"""desert-ant-labs/who from its ONNX export: semantic weights, and a NumPy forward of each graph.

The ONNX graphs are traced PyTorch exports with folded constants and 8-bit MatMulNBits weights;
`load_who(dir)` walks them and returns float32 weights under stable names, `encoder` / `queries`
mirror the graphs in NumPy (checked against ONNX Runtime by `python who_onnx.py DIR`). docs/who.md.
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

LAYERS = 6
HEADS = 4          # encoder self-attention
DIM = 256


def _dequant(node, init):
    """MatMulNBits (8 bit, no zero points: implicit 128) -> W [N, K] float32, y = x @ W.T."""
    a = {x.name: onnx.helper.get_attribute_value(x) for x in node.attribute}
    assert a["bits"] == 8 and len(node.input) == 3, node.name
    q = init[node.input[1]].astype(np.int32) - 128            # [N, K / bs, bs]
    s = init[node.input[2]].astype(np.float32)                # [N, K / bs]
    w = (q * s[:, :, None]).reshape(a["N"], -1)[:, :a["K"]]
    return w.astype(np.float32)


def _graph(path):
    m = onnx.load(str(path))
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    consumers = {}
    for n in m.graph.node:
        for x in n.input:
            consumers.setdefault(x, []).append(n)
    return m, init, consumers


def _bias_name(out, init, consumers):
    """The bias initializer added to a matmul output (`<name>.bias`), or None."""
    for c in consumers.get(out, []):
        if c.op_type == "Add":
            for x in c.input:
                if x in init and x.endswith(".bias"):
                    return x[:-len(".bias")].replace("model.", "")
    return None


def _linears(m, init, consumers, prefix):
    """Every matmul whose output gets a named bias: {prefix + name: (W [N, K], b [N])}."""
    out = {}
    alias = {n.output[0]: n.input[0] for n in m.graph.node if n.op_type == "Identity"}
    for n in m.graph.node:
        b_in = alias.get(n.input[1], n.input[1]) if len(n.input) > 1 else None
        if n.op_type == "MatMulNBits":
            w = _dequant(n, init)
        elif n.op_type == "MatMul" and b_in in init and init[b_in].ndim == 2:
            w = init[b_in].T.astype(np.float32)                 # MatMul(x, W[K, N])
        else:
            continue
        name = _bias_name(n.output[0], init, consumers)
        if name is None:
            continue
        if name.startswith("mask_embed") and n.op_type == "MatMulNBits":
            name = name.replace("mask_embed", "mask_embed_q")   # the first mask's MLP uses 8-bit copies
        b = next(init[x] for c in consumers[n.output[0]] if c.op_type == "Add" for x in c.input if x in init and x.endswith(".bias"))
        out[prefix + name] = (w, b.astype(np.float32))
    return out


def load_encoder(path):
    m, init, cons = _graph(path)
    W = {}
    for k, (w, b) in _linears(m, init, cons, "").items():
        W[k + ".weight"] = w
        W[k + ".bias"] = b
    keep = ("down.", "encoder.", "conv.", "up.")
    for k, v in init.items():
        k2 = k.replace("model.", "")
        if k2.startswith(keep) and not k2.endswith("mask") and v.dtype == np.float32 and k2 not in W:
            W[k2] = v.astype(np.float32)
    nodes = list(m.graph.node)
    # per layer: content / position biases (u, v) and the position projection, the key tables
    for n in nodes:
        if n.op_type == "MatMulNBits" and n.input[1].startswith("attn.") and n.input[1].endswith(".keys_Q8"):
            L = int(n.input[1].split(".")[1])
            W[f"attn.{L}.keys"] = _dequant(n, init)                  # [T, 256]
        if n.op_type == "MatMul" and n.input[1] in init and n.input[1].endswith(".w_pos"):
            L = int(n.input[1].split(".")[1])
            W[f"attn.{L}.w_pos"] = init[n.input[1]].astype(np.float32)   # [heads, 64, 256]
            add = next(p for p in nodes if p.output[0] == n.input[0])
            W[f"attn.{L}.pos_bias"] = init[next(x for x in add.input if x in init)].astype(np.float32)   # [heads, 1, 64]
            # the other add of the same q feeds the content score
            q = next(x for x in add.input if x not in init)
            other = next(p for p in cons[q] if p.op_type == "Add" and p is not add)
            W[f"attn.{L}.content_bias"] = init[next(x for x in other.input if x in init)].astype(np.float32)
    consts = {k: float(init[k]) for k in ("val_27", "val_35", "val_85") if k in init}
    assert consts == {"val_27": 16.0, "val_35": 0.5, "val_85": 8.0}, consts
    return W


def load_queries(path):
    m, init, cons = _graph(path)
    W = {}
    for k, (w, b) in _linears(m, init, cons, "").items():
        W[k + ".weight"] = w
        W[k + ".bias"] = b
    for k, v in init.items():
        k2 = k.replace("model.", "")
        if (k2.startswith(("decoder", "pairing")) and k2.endswith((".weight", ".bias")) or k2 == "pairing") and k2 not in W:
            W[k2] = v.astype(np.float32)
    # learned query features (input of the first decoder_norm) and query positions
    first_ln = next(n for n in m.graph.node if n.op_type == "LayerNormalization" and n.input[0] in init)
    W["query_feat"] = init[first_ln.input[0]][0].astype(np.float32)          # [50, 256]
    adds = [n for n in m.graph.node if n.op_type == "Add" and any(x in init and init[x].shape == (1, 50, DIM) for x in n.input)]
    pos = {x for n in adds for x in n.input if x in init and init[x].shape == (1, 50, DIM) and x != first_ln.input[0]}
    assert len(pos) == 1, pos
    W["query_pos"] = init[pos.pop()][0].astype(np.float32)
    return W


VOICE_STAGES = (3, 4, 6, 3)


def load_voice(path):
    """ResNet34 (batch norm folded): {name: weight [O, C, kh, kw]}, {name: bias [O]}."""
    _, init, _ = _graph(path)
    W = {}
    for k, v in init.items():
        if k.startswith("released.") and k.endswith(".weight"):
            n = k[len("released."):-len(".weight")]
            W[n + ".weight"] = v.astype(np.float32)
            W[n + ".bias"] = init[k + "_bias"].astype(np.float32)
    return W


def load_sidecar_banks(path):
    b = np.fromfile(path, np.uint8)
    mels, bins = np.frombuffer(b[:8].tobytes(), "<u4")
    return np.frombuffer(b[8:].tobytes(), "<f4").reshape(mels, bins).astype(np.float32)


def load_sidecar_embed(path):
    b = np.fromfile(path, np.uint8)
    out, inner = np.frombuffer(b[:8].tobytes(), "<u4")
    v = np.frombuffer(b[8:].tobytes(), "<f2").astype(np.float32)
    return v[:out * inner].reshape(out, inner), v[out * inner:]


def load_who(d):
    d = Path(d)
    meta = json.loads((d / "who_meta.json").read_text())
    enc = {w["frames"]: load_encoder(d / w["encoder"]["compiled"]) for w in meta["windows"]}
    qry = {w["frames"]: load_queries(d / w["queries"]["compiled"]) for w in meta["windows"]}
    return meta, enc, qry


# ---------------------------------------------------------------- NumPy forward

def ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def lin(W, k, x):
    return x @ W[k + ".weight"].T + W[k + ".bias"]


def silu(x):
    return x / (1 + np.exp(-x))


def gelu(x):
    from math import erf
    return 0.5 * x * (1 + np.vectorize(erf)(x / np.sqrt(2)))


def conv1d(x, w, b, stride=1, pad=0, groups=1):
    """x [C, T], w [O, C / groups, K] -> [O, T']."""
    C, T = x.shape
    O, Cg, K = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad)))
    To = (T + 2 * pad - K) // stride + 1
    idx = np.arange(To)[:, None] * stride + np.arange(K)[None, :]       # [To, K]
    out = np.empty((O, To), np.float32)
    og = O // groups
    for g in range(groups):
        xs = xp[g * Cg:(g + 1) * Cg][:, idx]                             # [Cg, To, K]
        ws = w[g * og:(g + 1) * og]                                      # [og, Cg, K]
        out[g * og:(g + 1) * og] = np.einsum("ctk,ock->ot", xs, ws)
    if b is not None:
        out += b[:, None]
    return out


def rope_tables(T, n=128, base=10000.0):
    inv = base ** (-np.arange(n) / n)
    a = np.arange(T)[:, None] * inv[None, :]
    return np.sin(a).astype(np.float32), np.cos(a).astype(np.float32)


def encoder(W, mel):
    """mel [F, 23] (F = window frames) -> memory [F/10, 256], frames [F, 256]."""
    F = mel.shape[0]
    x = conv1d(mel.T, W["down.depthwise.weight"], W["down.depthwise.bias"], stride=10, pad=7, groups=mel.shape[1])
    x = conv1d(x, W["down.pointwise.weight"], W["down.pointwise.bias"]).T             # [T, 256]
    x = ln(x, W["down.norm.weight"], W["down.norm.bias"]) * 16.0
    T = x.shape[0]
    sin, cos = rope_tables(T)
    for L in range(LAYERS):
        e = f"encoder.{L}."
        h = ln(x, W[e + "norm_ff1.weight"], W[e + "norm_ff1.bias"])
        x = x + 0.5 * lin(W, e + "ff1.net.3", silu(lin(W, e + "ff1.net.0", h)))
        h = ln(x, W[e + "norm_attn.weight"], W[e + "norm_attn.bias"])
        a = f"attn.{L}."
        q = lin(W, a + "attn.q", h).reshape(T, HEADS, 64).transpose(1, 0, 2)          # [H, T, 64]
        k = lin(W, a + "attn.k", h).reshape(T, HEADS, 64).transpose(1, 0, 2)
        v = lin(W, a + "attn.v", h).reshape(T, HEADS, 64).transpose(1, 0, 2)
        p = (q + W[a + "pos_bias"]) @ W[a + "w_pos"]                                  # [H, T, 256]
        s1, s2 = p[..., :128], p[..., 128:]
        rot = np.concatenate([s1 * sin + s2 * cos, s2 * sin - s1 * cos], -1)
        pos = rot @ W[a + "keys"].T                                                    # [H, T, T]
        cnt = (q + W[a + "content_bias"]) @ k.transpose(0, 2, 1)
        sc = (cnt + pos) / 8.0
        sc = np.exp(sc - sc.max(-1, keepdims=True))
        sc /= sc.sum(-1, keepdims=True)
        o = (sc @ v).transpose(1, 0, 2).reshape(T, DIM)
        x = x + lin(W, a + "attn.out", o)
        h = ln(x, W[e + "norm_conv.weight"], W[e + "norm_conv.bias"]).T                # [256, T]
        c = f"conv.{L}.conv."
        h = conv1d(h, W[c + "inward.weight"], W[c + "inward.bias"])
        h = h[:DIM] * (1 / (1 + np.exp(-h[DIM:])))
        h = conv1d(h, W[c + "depthwise.weight"], W[c + "depthwise.bias"], pad=24, groups=DIM).T
        h = silu(ln(h, W[c + "norm.weight"], W[c + "norm.bias"])).T
        x = x + conv1d(h, W[c + "outward.weight"], W[c + "outward.bias"]).T
        h = ln(x, W[e + "norm_ff2.weight"], W[e + "norm_ff2.bias"])
        x = x + 0.5 * lin(W, e + "ff2.net.3", silu(lin(W, e + "ff2.net.0", h)))
        x = ln(x, W[e + "norm_out.weight"], W[e + "norm_out.bias"])
    memory = x
    # upsampler: zero-insert x2 + conv k3, LN, GELU; zero-insert x5 + conv k5, LN, GELU
    def up(x, factor, w, b, k, keep):
        z = np.zeros((x.shape[0], x.shape[1] * factor), np.float32)
        z[:, ::factor] = x
        y = conv1d(z, w, b, pad=k - 1)
        return y[:, :keep]
    y = up(memory.T, 2, W["up.first.weight"], W["up.first.bias"], 3, 2 * T + 1)       # [256, 2T + 1]
    y = gelu(ln(y.T, W["up.up.norm1.weight"], W["up.up.norm1.bias"])).T
    y = up(y, 5, W["up.second.weight"], W["up.second.bias"], 5, 5 * (2 * T + 1))      # [256, 10T + 5]
    y = gelu(ln(y.T, W["up.up.norm2.weight"], W["up.up.norm2.bias"]))                 # [10T + 5, 256]
    y = y[7:7 + F]
    frames = np.concatenate([y, np.repeat(y[-1:], F - y.shape[0], 0)], 0)
    return memory.astype(np.float32), frames.astype(np.float32)


def queries(W, memory, frames, heads=4, heads_self=None):
    """memory [T, 256], frames [F, 256] -> presence logits [50], queries [50, 256], mask logits [50, F]."""
    T = memory.shape[0]
    low = conv1d(frames.T, W["pairing"], None, stride=10, groups=DIM)                  # [256, T]
    dh = DIM // heads

    def mask_logits(q, f, mlp="mask_embed"):
        e = q
        for i in (0, 2):
            e = np.maximum(lin(W, f"{mlp}.{i}", e), 0)
        e = lin(W, f"{mlp}.4", e)
        return e @ f                                                                     # [50, len]

    def attn(prefix, xq, xk, xv, bias=None, heads=heads):
        dh = DIM // heads
        q = lin(W, prefix + ".q", xq).reshape(-1, heads, dh).transpose(1, 0, 2)
        k = lin(W, prefix + ".k", xk).reshape(-1, heads, dh).transpose(1, 0, 2)
        v = lin(W, prefix + ".v", xv).reshape(-1, heads, dh).transpose(1, 0, 2)
        s = q @ k.transpose(0, 2, 1) / np.sqrt(dh)
        if bias is not None:
            s = s + bias
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        return lin(W, prefix + ".out", (s @ v).transpose(1, 0, 2).reshape(-1, DIM))

    x = W["query_feat"]
    qp = W["query_pos"]
    for L in range(LAYERS):
        d = f"decoder.{L}."
        ml = mask_logits(ln(x, W["decoder_norm.weight"], W["decoder_norm.bias"]), low,     # [50, T]
                         "mask_embed_q" if L == 0 and "mask_embed_q.0.weight" in W else "mask_embed")
        allowed = (ml >= 0).astype(np.float32)
        allowed = np.where(allowed.max(-1, keepdims=True) > 0, allowed, 1.0)
        bias = (allowed - 1.0) * 10000.0
        x = ln(x + attn(d + "cross", x + qp, memory, memory, bias), W[d + "norm1.weight"], W[d + "norm1.bias"])
        x = ln(x + attn(d + "self_attn", x + qp, x + qp, x, heads=heads_self or heads), W[d + "norm2.weight"], W[d + "norm2.bias"])
        x = ln(x + lin(W, d + "ff.2", np.maximum(lin(W, d + "ff.0", x), 0)), W[d + "norm3.weight"], W[d + "norm3.bias"])
    qn = ln(x, W["decoder_norm.weight"], W["decoder_norm.bias"])
    presence = lin(W, "presence", qn)[:, 0]
    masks = mask_logits(qn, frames.T)
    return presence.astype(np.float32), qn.astype(np.float32), masks.astype(np.float32)


def conv2d(x, w, b, stride=1, pad=1):
    """x [C, H, W], w [O, C, kh, kw] -> [O, H', W']."""
    C, H, Wd = x.shape
    O, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    Ho, Wo = (H + 2 * pad - kh) // stride + 1, (Wd + 2 * pad - kw) // stride + 1
    cols = np.empty((C, kh, kw, Ho, Wo), np.float32)
    for i in range(kh):
        for j in range(kw):
            cols[:, i, j] = xp[:, i:i + stride * Ho:stride, j:j + stride * Wo:stride]
    out = w.reshape(O, -1) @ cols.reshape(-1, Ho * Wo)
    return (out + b[:, None]).reshape(O, Ho, Wo).astype(np.float32)


def voice(W, mel):
    """mel [F, 80] -> columns [F / 8, 2560] (256 channels x 10 bands, channel major)."""
    relu = lambda t: np.maximum(t, 0)
    x = relu(conv2d(mel.T[None], W["conv1.weight"], W["conv1.bias"]))
    for s, n in enumerate(VOICE_STAGES):
        for blk in range(n):
            p = f"stages.{s}.{blk}."
            stride = 2 if s > 0 and blk == 0 else 1
            h = relu(conv2d(x, W[p + "conv1.weight"], W[p + "conv1.bias"], stride))
            h = conv2d(h, W[p + "conv2.weight"], W[p + "conv2.bias"])
            sc = conv2d(x, W[p + "shortcut.0.weight"], W[p + "shortcut.0.bias"], stride, 0) if p + "shortcut.0.weight" in W else x
            x = relu(h + sc)
    C, H, T = x.shape
    return x.reshape(C * H, T).T.copy()


# ---------------------------------------------------------------- check against ONNX Runtime

def _check(d):
    import onnxruntime as ort
    d = Path(d)
    meta, enc, qry = load_who(d)
    rng = np.random.default_rng(1)
    for w in meta["windows"]:
        F = w["frames"]
        if F > 6000 and "--all" not in sys.argv:
            continue
        mel = rng.standard_normal((F, meta["frontend"]["n_mels"])).astype(np.float32)
        se = ort.InferenceSession(str(d / w["encoder"]["compiled"]))
        mem_r, fr_r = se.run(["memory", "frames"], {"mel": mel[None]})
        mem, fr = encoder(enc[F], mel)
        print(f"encoder {F}: memory {np.abs(mem - mem_r[0]).max():.3g} (max {np.abs(mem_r).max():.3g}), "
              f"frames {np.abs(fr - fr_r[0]).max():.3g} (max {np.abs(fr_r).max():.3g})")
        sq = ort.InferenceSession(str(d / w["queries"]["compiled"]))
        feeds = {"memory": mem_r}
        at = 0
        for i, c in enumerate(w["chunks"]):
            feeds[f"frames_{i}"] = fr_r[:, at:at + c]
            at += c
        outs = sq.run(None, feeds)
        names = [o.name for o in sq.get_outputs()]
        r = dict(zip(names, outs))
        for heads, hs in ((4, 4),):
            pr, qn, mk = queries(qry[F], mem_r[0], fr_r[0], heads, hs)
            ref_masks = np.concatenate([r[f"masks_{i}"][0] for i in range(len(w["chunks"]))], -1)
            print(f"queries {F} heads {heads}/{hs}: presence {np.abs(pr - r['presence'][0]).max():.3g}, "
                  f"queries {np.abs(qn - r['queries'][0]).max():.3g}, masks {np.abs(mk - ref_masks).max():.3g} (max {np.abs(ref_masks).max():.3g})")


def _check_voice(d):
    import onnxruntime as ort
    d = Path(d)
    meta = json.loads((d / "who_meta.json").read_text())
    v = meta["voice"]
    W = load_voice(d / v["compiled"])
    mel = np.random.default_rng(2).standard_normal((v["mel_frames"], v["frontend"]["n_mels"])).astype(np.float32)
    ref = ort.InferenceSession(str(d / v["compiled"])).run(None, {"mel": mel[None]})[0][0]
    got = voice(W, mel)
    print(f"voice: {np.abs(got - ref).max():.3g} (max {np.abs(ref).max():.3g}) shape {got.shape}")


if __name__ == "__main__":
    if "--voice" in sys.argv:
        _check_voice(sys.argv[1])
    else:
        _check(sys.argv[1])
