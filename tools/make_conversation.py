"""Builds a deterministic multi-speaker test recording for the diarization model.

usage: make_conversation.py jfk.wav en-Alice_woman.wav en-Carter_man.wav en-Frank_man.wav out.f32 [--minutes M]

Four speakers (jfk: public domain; the VibeVoice voices: MIT), 16 kHz mono, about
84 s: A, B, C overlapping B for 1 s, D, then A again. Long enough for the speaker
cache to be compressed both offline (> 27 s) and with the streaming presets (> 56 s).
Written as raw float32 samples, the exact input of the references and the tests.

With --minutes M: a longer meeting of the same four speakers (turns in a fixed pseudo-random order, some
overlapping), at least M minutes, for models that cut recordings into pieces and join speakers across them.
"""
import sys

import librosa
import numpy as np

SR = 16000


def load(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)


def meeting(voices, minutes):
    rng = np.random.default_rng(7)
    out = np.zeros(0, np.float32)
    last = -1
    while len(out) < minutes * 60 * SR:
        k = int(rng.integers(len(voices)))
        if k == last:
            continue
        v = voices[k]
        a = int(rng.integers(0, len(v) // 2))                      # a stretch of 3..all of the clip
        seg = v[a:a + int(rng.integers(3 * SR, len(v) - a + 1))] if len(v) - a > 3 * SR else v[a:]
        gap = int(rng.integers(-SR // 2, SR))                      # up to 0.5 s of overlap, up to 1 s of silence
        if gap >= 0:
            out = np.concatenate([out, np.zeros(gap, np.float32), seg])
        else:
            tail = np.zeros(len(out) + gap + len(seg), np.float32)
            tail[:len(out)] = out
            tail[len(out) + gap:] += seg
            out = tail
        last = k
    return out


def main():
    minutes = None
    if "--minutes" in sys.argv:
        i = sys.argv.index("--minutes")
        minutes = float(sys.argv[i + 1])
        del sys.argv[i:i + 2]
    jfk, alice, carter, frank, dst = sys.argv[1:6]
    A, B, C, D = load(jfk), load(alice), load(carter), load(frank)
    if minutes:
        x = meeting([A, B, C, D], minutes)
        x = (0.9 * x / np.abs(x).max()).astype(np.float32)
        x.tofile(dst)
        print(f"wrote {dst}: {len(x) / SR:.1f} s")
        return
    gap = np.zeros(SR // 2, np.float32)
    overlap = SR                                   # C starts 1 s before B ends
    bc = np.zeros(len(B) + len(C) - overlap, np.float32)
    bc[:len(B)] += B
    bc[len(B) - overlap:] += C
    x = np.concatenate([A, gap, bc, gap, D, gap, A])
    x = (0.9 * x / np.abs(x).max()).astype(np.float32)
    x.tofile(dst)
    print(f"wrote {dst}: {len(x) / SR:.1f} s")


if __name__ == "__main__":
    main()
