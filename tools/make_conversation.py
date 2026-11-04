"""Builds a deterministic multi-speaker test recording for the diarization model.

usage: make_conversation.py jfk.wav en-Alice_woman.wav en-Carter_man.wav en-Frank_man.wav out.f32

Four speakers (jfk: public domain; the VibeVoice voices: MIT), 16 kHz mono, about
84 s: A, B, C overlapping B for 1 s, D, then A again. Long enough for the speaker
cache to be compressed both offline (> 27 s) and with the streaming presets (> 56 s).
Written as raw float32 samples, the exact input of the references and the tests.
"""
import sys

import librosa
import numpy as np

SR = 16000


def load(path):
    y, _ = librosa.load(path, sr=SR, mono=True)
    return y.astype(np.float32)


def main():
    jfk, alice, carter, frank, dst = sys.argv[1:6]
    A, B, C, D = load(jfk), load(alice), load(carter), load(frank)
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
