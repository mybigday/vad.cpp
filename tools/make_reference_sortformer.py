"""Reference speaker probabilities of Nemotron-3-Diarization from the official
transformers implementation (CPU, float32, eager attention).

usage: make_reference_sortformer.py MODEL_DIR AUDIO.f32 OUT_PREFIX

MODEL_DIR holds the Hugging Face checkpoint (config.json, processor_config.json,
model.safetensors). Writes, as float32 [frames, 8]:
  OUT_PREFIX.offline.probs.f32           offline forward (the model's own chunking)
  OUT_PREFIX.<mode>.probs.f32            streaming, one chunk per forward, for every
                                         processor streaming mode
Offline output has 1 + len // 160 frames; the last one belongs to the masked frame
the feature extractor adds, vad.cpp emits the first len // 160.
"""
import sys

import numpy as np
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor


def stream(model, processor, audio):
    sr = processor.feature_extractor.sampling_rate
    chunks = [processor(audio[: processor.num_samples_first_audio_chunk], sampling_rate=sr, is_streaming=True,
                        is_first_audio_chunk=True)]
    mel = processor.num_mel_frames_per_step
    start = processor.audio_chunk_start(mel)
    while (end := start + processor.num_samples_per_audio_chunk) <= audio.shape[0]:
        chunks.append(processor(audio[start:end], sampling_rate=sr, is_streaming=True, is_first_audio_chunk=False))
        mel += processor.num_mel_frames_per_step
        start = processor.audio_chunk_start(mel)
    chunks.append(processor(audio[start:], sampling_rate=sr, is_streaming=True, is_first_audio_chunk=False,
                            is_last_audio_chunk=True))
    cache, out = None, []
    for inp in chunks:
        o = model(**inp, speaker_cache=cache)
        out.append(o.logits)
        cache = o.speaker_cache
    return torch.cat(out, 1)


def main():
    model_dir, audio_path, prefix = sys.argv[1:4]
    torch.manual_seed(0)
    processor = AutoProcessor.from_pretrained(model_dir)
    model = AutoModelForAudioFrameClassification.from_pretrained(model_dir, attn_implementation="eager").eval()
    audio = np.fromfile(audio_path, dtype=np.float32)
    with torch.inference_mode():
        inputs = processor(audio, sampling_rate=16000)
        probs = model(**inputs).logits.sigmoid()[0].numpy()
        probs.astype(np.float32).tofile(f"{prefix}.offline.probs.f32")
        print(f"offline: {probs.shape[0]} frames, active speakers {sorted(set(np.nonzero(probs > 0.5)[1].tolist()))}")
        for mode in processor.streaming_modes:
            processor.set_streaming_mode(mode)
            p = stream(model, processor, audio).sigmoid()[0].numpy()
            p.astype(np.float32).tofile(f"{prefix}.{mode}.probs.f32")
            n = min(len(p), len(probs))
            print(f"{mode}: {p.shape[0]} frames, max |stream - offline| {np.abs(p[:n] - probs[:n]).max():.3f}")


if __name__ == "__main__":
    main()
