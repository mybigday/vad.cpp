"""Reference data for the tests.

usage: make_reference.py AUDIO.wav OUT_DIR MODEL.onnx [MODEL.onnx ...]

For every model: OUT_DIR/<audio>.<model>.probs.f32.
- silero_vad*.onnx: ONNX Runtime streaming probabilities (official 64-sample
  context semantics, last chunk zero padded).
- fireredvad_stream_vad.onnx (cmvn.ark next to it): official full-utterance
  inference (kaldi-native-fbank + CMVN + ONNX), one probability per 10 ms frame,
  frame t covering samples [160 t, 160 t + 400).
Also OUT_DIR/segments.txt: official get_speech_timestamps_from_probs results
for a set of probability sequences and parameters (see segment_cases()).
"""
import ast
import os
import sys

import numpy as np
import soundfile as sf

from silero_ref import run_ort

UTILS_URL = "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2.3/src/silero_vad/utils_vad.py"


def official_segments_fn():
    """Load get_speech_timestamps_from_probs from the official source without torch."""
    import urllib.request
    src = urllib.request.urlopen(UTILS_URL).read().decode()
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_speech_timestamps_from_probs")
    ns = {"List": list}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "utils_vad.py", "exec"), ns)
    return ns["get_speech_timestamps_from_probs"]


def segment_cases(jfk_probs):
    rng = np.random.default_rng(0)
    # smooth random speech/silence pattern with noise
    blocks = rng.integers(0, 2, 60).repeat(rng.integers(3, 40, 60))
    synth = np.clip(blocks * 0.8 + rng.normal(0.1, 0.15, len(blocks)), 0, 1).astype(np.float32)
    params = [
        dict(),
        dict(threshold=0.6, min_speech_duration_ms=100, min_silence_duration_ms=300, speech_pad_ms=100),
        dict(max_speech_duration_s=1.5),
        dict(max_speech_duration_s=3.0, speech_pad_ms=0, neg_threshold=0.2),
    ]
    for name, probs in (("jfk", jfk_probs), ("synth", synth)):
        for p in params:
            yield name, probs, p


def firered_reference(onnx_path, audio, out_path):
    import onnxruntime as ort
    from firered_common import fbank_knf, read_kaldi_cmvn
    means, istd = read_kaldi_cmvn(os.path.join(os.path.dirname(onnx_path), "cmvn.ark"))
    pcm = np.round(audio * 32768.0).clip(-32768, 32767).astype(np.int16)
    feat = ((fbank_knf(pcm) - means) * istd).astype(np.float32)
    s = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    probs = s.run(None, {"feat": feat[None]})[0].reshape(-1).astype(np.float32)
    probs.tofile(out_path)
    print(f"{os.path.basename(out_path)}: {len(probs)} probabilities")


def smart_turn_reference(onnx_path, audio, audio_path, out_dir):
    """Official pipecat inference.py: last 8 s (left zero padded), WhisperFeatureExtractor
    (chunk_length=8, do_normalize=True), ONNX -> one probability per window.
    Writes smart-turn.txt lines: "<audio file> <end sample> <probability>"."""
    import onnxruntime as ort
    from transformers import WhisperFeatureExtractor
    fe = WhisperFeatureExtractor(chunk_length=8)
    s = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    n8 = 8 * 16000
    with open(os.path.join(out_dir, "smart-turn.txt"), "w") as f:
        for end in (16000, 35200, 70400, 99200, 123200, len(audio)):
            x = audio[:end][-n8:]
            x = np.pad(x, (n8 - len(x), 0)) if len(x) < n8 else x
            feats = fe(x, sampling_rate=16000, return_tensors="np", padding="max_length", max_length=n8,
                       truncation=True, do_normalize=True).input_features.astype(np.float32)
            p = float(s.run(None, {"input_features": feats})[0].reshape(-1)[0])
            f.write(f"{os.path.basename(audio_path)} {end} {p:.8f}\n")
            print(f"smart-turn end={end}: {p:.6f}")


def main():
    audio_path, out_dir, models = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(out_dir, exist_ok=True)
    audio, sr = sf.read(audio_path, dtype="float32")
    assert sr == 16000 and audio.ndim == 1
    padded = np.concatenate([audio, np.zeros((-len(audio)) % 512, np.float32)])
    # the exact float input every reference was computed from (for runtimes without a WAV reader, e.g. the browser test)
    base0 = os.path.splitext(os.path.basename(audio_path))[0]
    audio.astype(np.float32).tofile(os.path.join(out_dir, f"{base0}.f32"))
    base = os.path.splitext(os.path.basename(audio_path))[0]
    probs_by_model = {}
    for m in models:
        if "smart-turn" in os.path.basename(m):
            smart_turn_reference(m, audio, audio_path, out_dir)
            continue
        if "fireredvad" in os.path.basename(m):
            tag = "firered-stream" if "stream" in os.path.basename(m) else "firered-vad"
            firered_reference(m, audio, os.path.join(out_dir, f"{base}.{tag}.probs.f32"))
            continue
        probs, _ = run_ort(m, padded)
        tag = os.path.splitext(os.path.basename(m))[0].replace("silero_vad_", "silero-")
        probs.astype(np.float32).tofile(os.path.join(out_dir, f"{base}.{tag}.probs.f32"))
        probs_by_model[tag] = probs
        print(f"{tag}: {len(probs)} probabilities")

    if not probs_by_model:
        return
    fn = official_segments_fn()
    jfk_probs = probs_by_model[sorted(probs_by_model)[-1]].astype(np.float32)
    with open(os.path.join(out_dir, "segments.txt"), "w") as f:
        # line: name n_samples threshold neg_threshold min_speech min_silence pad max_speech | probs-file | start end ...
        for i, (name, probs, p) in enumerate(segment_cases(jfk_probs)):
            pf = f"segcase{i}.f32"
            probs.astype(np.float32).tofile(os.path.join(out_dir, pf))
            n_samples = len(audio) if name == "jfk" else len(probs) * 512
            segs = fn([float(x) for x in probs], audio_length_samples=n_samples, **p)
            thr = p.get("threshold", 0.5)
            fields = [pf, n_samples, thr, p.get("neg_threshold", -1), p.get("min_speech_duration_ms", 250),
                      p.get("min_silence_duration_ms", 100), p.get("speech_pad_ms", 30), p.get("max_speech_duration_s", 0)]
            f.write(" ".join(str(x) for x in fields) + " | " + " ".join(f"{s['start']} {s['end']}" for s in segs) + "\n")
    print("wrote", os.path.join(out_dir, "segments.txt"))


if __name__ == "__main__":
    main()
