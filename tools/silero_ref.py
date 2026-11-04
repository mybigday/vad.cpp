"""Numpy reference of Silero VAD (16 kHz) checked against official ONNX Runtime.

Streaming semantics follow silero's OnnxWrapper: each 512-sample chunk is
prefixed with the last 64 samples of the previous chunk (zeros at start),
LSTM state carries over.
"""
import sys, time
import numpy as np, soundfile as sf, onnxruntime as ort
from silero_onnx_weights import load

N_WIN, N_CTX = 512, 64


def conv1d(x, w, b, stride, pad):  # x (Cin, T), w (Cout, Cin, K)
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad)))
    K = w.shape[2]
    T = (x.shape[1] - K) // stride + 1
    cols = np.stack([x[:, t * stride:t * stride + K] for t in range(T)], 0)  # T,Cin,K
    y = np.einsum("tck,ock->ot", cols, w)
    return y + b[:, None] if b is not None else y


class Ref:
    def __init__(self, W):
        self.W = W
        self.reset()

    def reset(self):
        self.h = np.zeros(128, np.float32); self.c = np.zeros(128, np.float32)
        self.ctx = np.zeros(N_CTX, np.float32)

    def step(self, chunk):
        W = self.W
        x = np.concatenate([self.ctx, chunk]); self.ctx = chunk[-N_CTX:].copy()
        x = np.pad(x, (0, 64), mode="reflect")[None]
        s = conv1d(x, W["stft.forward_basis_buffer"], None, 128, 0)  # 258,4
        mag = np.sqrt(s[:129] ** 2 + s[129:] ** 2)
        y = mag
        for i, st in enumerate([1, 2, 2, 1]):
            y = np.maximum(conv1d(y, W[f"encoder.{i}.reparam_conv.weight"], W[f"encoder.{i}.reparam_conv.bias"], st, 1), 0)
        x = y[:, 0]
        g = W["decoder.rnn.weight_ih"] @ x + W["decoder.rnn.bias_ih"] + W["decoder.rnn.weight_hh"] @ self.h + W["decoder.rnn.bias_hh"]
        sig = lambda v: 1 / (1 + np.exp(-v))
        i, f, gg, o = sig(g[:128]), sig(g[128:256]), np.tanh(g[256:384]), sig(g[384:])
        self.c = f * self.c + i * gg; self.h = o * np.tanh(self.c)
        z = W["decoder.decoder.2.weight"][0, :, 0] @ np.maximum(self.h, 0) + W["decoder.decoder.2.bias"][0]
        return float(sig(z))


def run_ort(path, audio):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    state = np.zeros((2, 1, 128), np.float32); ctx = np.zeros((1, N_CTX), np.float32)
    sr = np.array(16000, np.int64); out = []
    t0 = time.perf_counter()
    for i in range(len(audio) // N_WIN):
        ch = audio[i * N_WIN:(i + 1) * N_WIN][None]
        p, state = s.run(None, {"input": np.concatenate([ctx, ch], 1), "state": state, "sr": sr})
        ctx = ch[:, -N_CTX:]; out.append(float(p[0, 0]))
    return np.array(out), time.perf_counter() - t0


if __name__ == "__main__":
    onnx_path, wav, out_npy = sys.argv[1:4]  # benchmark reference
    audio, sr = sf.read(wav, dtype="float32"); assert sr == 16000
    p_ort, t = run_ort(onnx_path, audio)
    n = len(p_ort)
    print(f"ORT(py, 1 thread): {n} chunks, {t*1e3:.1f} ms, {t/n*1e6:.1f} us/chunk, RTF {t/(len(audio)/16000):.5f}")
    ref = Ref(load(onnx_path)); p_ref = np.array([ref.step(audio[i*N_WIN:(i+1)*N_WIN]) for i in range(n)])
    print(f"numpy ref vs ORT: max|diff| = {np.abs(p_ref - p_ort).max():.2e}")
    np.save(out_npy, p_ort.astype(np.float32))
