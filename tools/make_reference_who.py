"""Reference output of desert-ant-labs/who for the tests: ONNX Runtime on the published graphs and the host
algorithm described in who_meta.json (docs/who.md), in NumPy.

usage: make_reference_who.py ONNX_DIR audio.f32 out_prefix [--max-speakers N]

Writes out_prefix.who.probs.f32 ([frames, N] float32: per 10 ms frame, each speaker's largest mask
probability among its lanes, 0 where it has none) and out_prefix.who.json (pieces, lanes, speakers, turns).
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
import who_onnx  # noqa: E402

SR = 16000


def frontend(x, fe, banks):
    """Log-mel [frames, mels] of the whole recording, and the per-band means over it."""
    n, hop, nfft = fe["frame_samples"], fe["shift_samples"], fe["n_fft"]
    frames = 0 if len(x) < n else 1 + (len(x) - n) // hop
    win = (0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))).astype(np.float32)
    idx = np.arange(frames)[:, None] * hop + np.arange(n)[None, :]
    s = x[idx].astype(np.float32) * np.float32(fe["scale"])
    s -= s.mean(1, keepdims=True)
    p = np.float32(fe["preemphasis"])
    s[:, 1:] -= p * s[:, :-1].copy()
    s[:, 0] -= p * s[:, 0]
    spec = np.fft.rfft(s * win, nfft)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    mel = np.log(np.maximum(power @ banks.T, np.float32(fe["log_floor"]))).astype(np.float32)
    means = mel.astype(np.float64).mean(0).astype(np.float32)
    return mel, means


def main():
    argv = list(sys.argv)
    max_speakers = 16
    if "--max-speakers" in argv:
        i = argv.index("--max-speakers")
        max_speakers = int(argv[i + 1])
        del argv[i:i + 2]
    src, audio, prefix = Path(argv[1]), argv[2], argv[3]
    meta = json.loads((src / "who_meta.json").read_text())
    x = np.fromfile(audio, np.float32)
    fe, vm, d, j = meta["frontend"], meta["voice"], meta["decision"], meta["join"]
    step = meta["frames"]["step_s"]
    mel, means = frontend(x, fe, who_onnx.load_sidecar_banks(src / fe["banks"]))
    feats = mel - means
    F = len(feats)
    windows = sorted(meta["windows"], key=lambda w: w["frames"])
    longest = windows[-1]["frames"]
    pieces = [(0, F)] if F <= longest else [(a, min(a + longest, F)) for a in range(0, F, longest)]
    sess = {}

    def run(path, feeds, outs=None):
        if path not in sess:
            sess[path] = ort.InferenceSession(str(src / path))
        return sess[path].run(outs, feeds)

    sig = lambda z: 1 / (1 + np.exp(-z))
    lanes = []
    for pi, (a, b) in enumerate(pieces):
        n = b - a
        w = next(w for w in windows if w["frames"] >= n) if n <= longest else windows[-1]
        inp = np.concatenate([feats[a:b], np.repeat(feats[b - 1:b], w["frames"] - n, 0)], 0)
        memory, frames = run(w["encoder"]["compiled"], {"mel": inp[None]}, ["memory", "frames"])
        feeds = {"memory": memory}
        at = 0
        for i, c in enumerate(w["chunks"]):
            feeds[f"frames_{i}"] = frames[:, at:at + c]
            at += c
        outs = dict(zip([o.name for o in sess[w["queries"]["compiled"]].get_outputs()] if w["queries"]["compiled"] in sess else
                        [o.name for o in ort.InferenceSession(str(src / w["queries"]["compiled"])).get_outputs()],
                        run(w["queries"]["compiled"], feeds)))
        presence = sig(outs["presence"][0])
        masks = sig(np.concatenate([outs[f"masks_{i}"][0] for i in range(len(w["chunks"]))], -1)[:, :n])   # [Q, n]
        on = masks > d["mask_threshold"]
        decided = [q for q in range(len(presence)) if presence[q] > d["presence_threshold"] and on[q].any()]
        for q in decided:
            others = np.zeros(n, bool)
            for r in decided:
                if r != q:
                    others |= on[r]
            lanes.append({"piece": pi, "query": q, "first": a, "prob": masks[q], "on": on[q], "solo": on[q] & ~others})

    def turns(lane):
        """Stretches of a lane on the recording's clock, [start, end) in seconds (half a frame either side)."""
        out = []
        on = np.concatenate([[False], lane["on"], [False]])
        edges = np.flatnonzero(on[1:] != on[:-1])
        for s, e in zip(edges[::2], edges[1::2]):
            c0, c1 = (lane["first"] + s) * step + step / 2, (lane["first"] + e - 1) * step + step / 2
            out.append((max(c0 - step / 2, 0.0), c1 + step / 2))
        return out

    for ln in lanes:
        ln["turns"] = turns(ln)

    if len(pieces) == 1:
        labels = list(range(len(lanes)))
    else:
        # voices from the solo columns of each lane with enough solo speech
        vf = vm["frontend"]
        vmel, vmeans = frontend(x, vf, who_onnx.load_sidecar_banks(src / vf["banks"]))
        vfeats = vmel - vmeans
        per, width, margin = vm["mel_frames_per_column"], vm["mel_frames"], vm["margin_mel_frames"]
        used = min(len(vfeats), F) // per * per
        cols = used // per
        minimum = int(round(j["min_solo_s"] / step))
        colmask = []
        for ln in lanes:
            if ln["solo"].sum() < minimum:
                colmask.append(None)
                continue
            full = np.zeros(cols * per, bool)
            lo, hi = ln["first"], min(ln["first"] + len(ln["solo"]), cols * per)
            full[lo:hi] = ln["solo"][:hi - lo]
            colmask.append(full.reshape(cols, per).sum(1) * 2 >= per)
        # windows of the voice graph: `width` frames, overlapping by `margin` a side
        wins = []
        if used <= width:
            wins.append((0, 0, cols, 0))
        else:
            kept, start = 0, 0
            while kept < cols:
                last = start + width >= used
                if last:
                    start = used - width
                end = cols if last else (start + width - margin) // per
                wins.append((start, kept - start // per, end - start // per, kept))
                kept = end
                start += width - 2 * margin
        sums = [None if m is None else [np.zeros(vm["columns_dim"]), np.zeros(vm["columns_dim"]), 0] for m in colmask]
        for start, k0, k1, place in wins:
            placed = slice(place, place + k1 - k0)
            if not any(m is not None and m[placed].any() for m in colmask):
                continue
            inp = np.zeros((width, vf["n_mels"]), np.float32)
            avail = min(width, used - start)
            inp[:avail] = vfeats[start:start + avail]
            out = run(vm["compiled"], {"mel": inp[None]})[0][0]          # [width / per, columns_dim]
            for row in range(k0, k1):
                c = place + row - k0
                for g, m in enumerate(colmask):
                    if m is not None and m[c]:
                        v = out[row].astype(np.float64)
                        sums[g][0] += v
                        sums[g][1] += v * v
                        sums[g][2] += 1
        ew, eb = who_onnx.load_sidecar_embed(src / vm["embedding"]["weights"])
        voices, voiced = [], []
        for g, s in enumerate(sums):
            if s is None or s[2] < 2:
                continue
            nn = s[2]
            mean = s[0] / nn
            var = np.maximum((s[1] - nn * mean * mean) / (nn - 1), 0)
            stats = np.concatenate([mean, np.sqrt(var)]).astype(np.float32)
            e = (eb + ew @ stats).astype(np.float32)
            voices.append(e / (np.sqrt(e @ e) + 1e-8))
            voiced.append(g)
        # average linkage on cosine distance, cut at 1 - cosine
        n = len(voices)
        member = list(range(n))
        if n > 1:
            V = np.array(voices)
            dist = (1 - V @ V.T).astype(np.float32)
            size = [1] * n
            alive = [True] * n
            while True:
                best, pair = np.inf, None
                for p in range(n):
                    if not alive[p]:
                        continue
                    for q in range(p + 1, n):
                        if alive[q] and dist[p, q] < best:
                            best, pair = dist[p, q], (p, q)
                if pair is None or best >= 1 - j["cosine"]:
                    break
                p, q = pair
                for c in range(n):
                    if alive[c] and c not in pair:
                        dist[p, c] = dist[c, p] = (size[p] * dist[p, c] + size[q] * dist[q, c]) / (size[p] + size[q])
                size[p] += size[q]
                alive[q] = False
                member = [p if m == q else m for m in member]
        order = {}
        labels = [-1] * len(lanes)
        for g, m in zip(voiced, member):
            labels[g] = order.setdefault(m, len(order))
        # voiceless lanes: the speaker of the voiced lane nearest in time (smallest gap, then most overlap)
        if voiced:
            for g in range(len(lanes)):
                if labels[g] >= 0:
                    continue
                best, bg, bo = voiced[0], np.inf, -np.inf
                for v in voiced:
                    gap, ov = np.inf, 0.0
                    for a0, a1 in lanes[g]["turns"]:
                        for b0, b1 in lanes[v]["turns"]:
                            sh = min(a1, b1) - max(a0, b0)
                            gap = min(gap, max(0.0, -sh))
                            ov += max(0.0, sh)
                    if gap < bg or (gap == bg and ov > bo):
                        best, bg, bo = v, gap, ov
                labels[g] = labels[best]
            renum = {}
            labels = [renum.setdefault(l, len(renum)) for l in labels]
        else:
            labels = list(range(len(lanes)))
    # speakers numbered by the start of their first turn (ties: the lower label)
    first = {}
    for ln, l in zip(lanes, labels):
        for t0, _ in ln["turns"]:
            first[l] = min(first.get(l, (np.inf, l)), (t0, l))
    order = {l: i for i, (_, l) in enumerate(sorted((v, k) for k, v in first.items()))}
    speakers = [order[l] for l in labels]
    # past the cap: the speaker of the nearest kept lane in time
    probs = np.zeros((F, max_speakers), np.float32)
    for ln, s in zip(lanes, speakers):
        if s >= max_speakers:
            raise SystemExit(f"{s + 1} speakers: past --max-speakers {max_speakers} (not exercised by the fixtures)")
        n = len(ln["prob"])
        probs[ln["first"]:ln["first"] + n, s] = np.maximum(probs[ln["first"]:ln["first"] + n, s], ln["prob"])
    probs.tofile(prefix + ".who.probs.f32")
    info = {"frames": F, "pieces": pieces, "speakers": max(speakers) + 1 if speakers else 0,
            "lanes": [{"piece": ln["piece"], "query": ln["query"], "solo_frames": int(ln["solo"].sum()), "speaker": s,
                       "turns": [[round(a, 3), round(b, 3)] for a, b in ln["turns"]]} for ln, s in zip(lanes, speakers)]}
    Path(prefix + ".who.json").write_text(json.dumps(info, indent=1))
    print(f"{audio}: {F} frames, {len(pieces)} pieces, {len(lanes)} lanes, {info['speakers']} speakers")


if __name__ == "__main__":
    main()
