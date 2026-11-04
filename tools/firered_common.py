"""FireRedVAD helpers shared by the converter and the reference generator."""
import struct

import numpy as np


def read_kaldi_cmvn(path):
    """Kaldi binary matrix (2 x (dim+1)) of CMVN stats -> (means, inverse std), as in fireredvad CMVN."""
    b = open(path, "rb").read()
    assert b[:2] == b"\0B", "expected Kaldi binary"
    p = 2
    while b[p:p + 1] == b" ":
        p += 1
    tok = b[p:p + 3]
    assert tok in (b"DM ", b"FM "), tok
    p += 3
    assert b[p] == 4
    rows = struct.unpack("<i", b[p + 1:p + 5])[0]
    assert b[p + 5] == 4
    cols = struct.unpack("<i", b[p + 6:p + 10])[0]
    p += 10
    dt = np.float64 if tok == b"DM " else np.float32
    stats = np.frombuffer(b, dt, rows * cols, p).reshape(rows, cols).astype(np.float64)
    dim = cols - 1
    count = stats[0, dim]
    means = stats[0, :dim] / count
    var = np.maximum(stats[1, :dim] / count - means * means, 1e-20)
    return means, 1.0 / np.sqrt(var)


def fbank_knf(samples_int16):
    """Official FireRedVAD features (kaldi-native-fbank, 80 mel, 25/10 ms, snip_edges)."""
    import kaldi_native_fbank as knf
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.frame_length_ms = 25
    opts.frame_opts.frame_shift_ms = 10
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = 80
    opts.mel_opts.debug_mel = False
    fb = knf.OnlineFbank(opts)
    fb.accept_waveform(16000, samples_int16.astype(np.float32).tolist())
    return np.array([fb.get_frame(i) for i in range(fb.num_frames_ready)], np.float32)


def fbank_linear(samples_float):
    """Same features from the closed form vad.cpp uses: one linear map per frame
    (int16 scale, DC removal, pre-emphasis 0.97, Povey window, 512-pt DFT) -> power -> mel -> log."""
    N, NFFT = 400, 512
    n = np.arange(N)
    win = (0.5 - 0.5 * np.cos(2 * np.pi * n / (N - 1))) ** 0.85
    dc = np.eye(N) - 1.0 / N
    pre = np.eye(N) - 0.97 * np.eye(N, k=-1)
    pre[0, 0] = 1 - 0.97
    k = np.arange(NFFT // 2 + 1)[:, None]
    dft = np.concatenate([np.cos(2 * np.pi * k * n / NFFT), -np.sin(2 * np.pi * k * n / NFFT)])
    L = dft @ np.diag(win) @ pre @ dc * 32768.0          # (514, 400)
    mel = mel_banks()                                    # (80, 257)
    T = 1 + (len(samples_float) - N) // 160
    frames = np.stack([samples_float[160 * t:160 * t + N] for t in range(T)]).astype(np.float64)
    spec = frames @ L.T
    power = spec[:, :257] ** 2 + spec[:, 257:] ** 2
    return np.log(np.maximum(power @ mel.T, np.finfo(np.float32).eps)).astype(np.float32)


def mel_banks(num_bins=80, nfft=512, sr=16000, low=20.0, high=8000.0):
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)
    lo, hi = mel(low), mel(high)
    delta = (hi - lo) / (num_bins + 1)
    m = mel(np.arange(nfft // 2) * sr / nfft)
    W = np.zeros((num_bins, nfft // 2 + 1))
    for b in range(num_bins):
        l, c, r = lo + b * delta, lo + (b + 1) * delta, lo + (b + 2) * delta
        up = (m > l) & (m <= c)
        down = (m > c) & (m < r)
        W[b, :nfft // 2][up] = (m[up] - l) / (c - l)
        W[b, :nfft // 2][down] = (r - m[down]) / (r - c)
    return W
