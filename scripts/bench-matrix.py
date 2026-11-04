#!/usr/bin/env python3
"""cpu (vad-cpu kernels) vs ggml-cpu for every model file, case and thread count.

usage: bench-matrix.py BENCH_BIN OUT.txt [--threads 1,4,16] [--models a.gguf,...] [--audio f.wav] [--reps N]
Prints one line per (file, threads, case): microseconds per output frame on both
devices and the ratio ggml-cpu / cpu (> 1: vad-cpu is faster).
"""
import argparse
import re
import subprocess
import sys

DEFAULT = ["silero-v6.2.gguf", "silero-v6.2-f16.gguf", "silero-v6.2-q8_0.gguf", "ggml-silero-v6.2.0.bin",
           "firered-stream-vad.gguf", "firered-stream-vad-f16.gguf", "firered-stream-vad-q8_0.gguf",
           "firered-vad.gguf", "firered-vad-f16.gguf", "firered-vad-q8_0.gguf",
           "smart-turn-v3.2.gguf", "smart-turn-v3.2-f16.gguf", "smart-turn-v3.2-q8_0.gguf"]


def run(bench, model, audio, dev, t, reps, streams, extra):
    cmd = [bench, "-m", model, "-f", audio, "-d", dev, "-t", str(t), "-r", str(reps), "-s", streams] + extra
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    res = {}
    for line in out.splitlines():
        m = re.match(r"\s+(.+?)\s{2,}([\d.]+)$", line)
        if m:
            res[m.group(1).strip()] = float(m.group(2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bench")
    ap.add_argument("out")
    ap.add_argument("--threads", default="1,4,16")
    ap.add_argument("--models", default=",".join("models/" + m for m in DEFAULT))
    ap.add_argument("--audio", default="assets/jfk.wav")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--streams", default="16,256")
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    lines = []
    worst = []
    for model in a.models.split(","):
        for t in map(int, a.threads.split(",")):
            c = run(a.bench, model, a.audio, "cpu", t, a.reps, a.streams, a.extra.split())
            g = run(a.bench, model, a.audio, "ggml-cpu", t, a.reps, a.streams, a.extra.split())
            for case in c:
                if case not in g:
                    continue
                r = g[case] / c[case] if c[case] else 0
                line = f"{model.split('/')[-1]:34s} t={t:<3d} {case:32s} cpu {c[case]:11.2f}  ggml {g[case]:11.2f}  x{r:5.2f}{'  <-- slower' if r < 1 else ''}"
                print(line, flush=True)
                lines.append(line)
                worst.append((r, line))
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\nslower than ggml-cpu:")
    for r, line in sorted(worst):
        if r < 1:
            print(line)


if __name__ == "__main__":
    main()
