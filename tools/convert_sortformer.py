"""Convert NVIDIA Nemotron-3-Diarization (Hugging Face checkpoint) to GGUF in the
official "sortformer" schema, the one of the Nemotron-3-Diarization.q8_0.gguf that
NVIDIA publishes (NeMo-Speech.cpp): same metadata keys, tensor names and layout.
vad.cpp loads both; this converter gives the original F32 precision.

usage: convert_sortformer.py MODEL_DIR out.gguf [--outtype f32|f16|bf16]

MODEL_DIR: config.json, processor_config.json and model.safetensors of
nvidia/Nemotron-3-Diarization. The log-mel filter bank (preprocessor.fb) is the
librosa slaney bank the transformers feature extractor uses.
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np
from safetensors.numpy import load_file

from gguf_common import pop_outtype, write_gguf


def main():
    outtype = pop_outtype(sys.argv)
    src, dst = Path(sys.argv[1]), sys.argv[2]
    cfg = json.loads((src / "config.json").read_text())
    fe = json.loads((src / "processor_config.json").read_text())["feature_extractor"]
    W = load_file(src / "model.safetensors")
    a, h, st = cfg["audio_config"], cfg["head_config"], cfg["streaming_config"]
    D, L, sub, nmel, sr = a["hidden_size"], a["num_hidden_layers"], a["subsampling_factor"], a["num_mel_bins"], fe["sampling_rate"]
    if a.get("partial_rotary_factor", 1.0) != 1.0 or a["num_key_value_heads"] != a["num_attention_heads"]:
        sys.exit("unsupported configuration")

    T = {"preprocessor.fb": librosa.filters.mel(sr=sr, n_fft=fe["n_fft"], n_mels=nmel, fmin=0.0, fmax=sr / 2, norm="slaney"),
         "encoder.pre_encode.proj.weight": W["model.audio_tower.embedder.projection.weight"],
         "encoder.embed_norm.weight": W["model.audio_tower.input_layer_norm.weight"],
         "encoder.embed_norm.bias": W["model.audio_tower.input_layer_norm.bias"]}
    for l in range(L):
        p, q = f"model.audio_tower.layers.{l}.", f"encoder.layers.{l}."
        T[q + "norm1.weight"] = W[p + "layer_norm1.weight"]
        T[q + "norm1.bias"] = W[p + "layer_norm1.bias"]
        T[q + "attn.w_qkv.weight"] = np.concatenate([W[p + f"self_attn.{n}_proj.weight"] for n in "qkv"], 0)
        T[q + "attn.out_proj.weight"] = W[p + "self_attn.o_proj.weight"]
        T[q + "attn.out_proj.bias"] = W[p + "self_attn.o_proj.bias"]
        T[q + "norm2.weight"] = W[p + "layer_norm2.weight"]
        T[q + "norm2.bias"] = W[p + "layer_norm2.bias"]
        T[q + "ffn.net.0.weight"] = W[p + "mlp.fc1.weight"]
        T[q + "ffn.net.0.bias"] = W[p + "mlp.fc1.bias"]
        T[q + "ffn.net.3.weight"] = W[p + "mlp.fc2.weight"]
        T[q + "ffn.net.3.bias"] = W[p + "mlp.fc2.bias"]
    T.update({"encoder.final_norm.weight": W["model.audio_tower.layer_norm.weight"],
              "encoder.final_norm.bias": W["model.audio_tower.layer_norm.bias"],
              "learnable_sil_emb": W["silence_embeds"],
              "head.first_hidden_to_hidden.weight": W["classifier.dense.weight"],
              "head.first_hidden_to_hidden.bias": W["classifier.dense.bias"],
              "head.single_hidden_to_spks.weight": W["classifier.out_proj.weight"],
              "head.single_hidden_to_spks.bias": W["classifier.out_proj.bias"],
              "encoder_proj.weight": W["model.proj.weight"],
              "encoder_proj.bias": W["model.proj.bias"],
              "subpixel_upsample.weight": W["model.upsampler.conv.weight"],     # (Cout, Cin, 3), as published
              "subpixel_upsample.bias": W["model.upsampler.conv.bias"]})

    # metadata keys and values of the published GGUF; the streaming.* keys hold the
    # model card's offline configuration (vad.cpp takes chunking from the stream parameters)
    kv = {"version": "v3",
          "encoder.d_model": D, "encoder.n_layers": L, "encoder.n_heads": a["num_attention_heads"],
          "encoder.d_ff": a["intermediate_size"], "encoder.conv_kernel_size": 0, "encoder.subsampling_factor": sub,
          "encoder.subsampling_conv_channels": 256, "encoder.feat_in": nmel, "encoder.xscaling": False,
          "encoder.use_bias": True, "encoder.pos_emb_max_len": a["max_position_embeddings"],
          "encoder.conv_norm": "batch_norm", "encoder.conv_context": "symmetric", "encoder.att_context_style": "regular",
          "encoder.type": "transformer_rope", "encoder.subsampling_type": "feature_stacking", "encoder.qkv_bias": False,
          "encoder.qk_norm": False, "encoder.pre_block_norm": True,
          "encoder.rope_base": float(a["rope_parameters"]["rope_theta"]), "encoder.rotary_fraction": 1.0,
          "transformer.n_layers": 0, "transformer.hidden_size": h["hidden_size"], "transformer.inner_size": 0,
          "transformer.n_heads": 0, "transformer.pre_ln": True,
          "num_speakers": h["num_speakers"], "high_resolution": True, "output_subsampling_factor": 1,
          "upsample_factor": h["subsampling_factor"], "learnable_silence": True,
          "preprocessor.sample_rate": sr, "preprocessor.window_size": fe["win_length"] / sr,
          "preprocessor.window_stride": fe["hop_length"] / sr, "preprocessor.n_fft": fe["n_fft"],
          "preprocessor.features": nmel, "preprocessor.normalize": "NA", "preprocessor.preemph": float(fe["preemphasis"]),
          "preprocessor.dither": 0.0, "preprocessor.log_zero_guard": 2.0 ** -24,
          "scoring.spkcache_sil_frames_per_spk": st["speaker_cache_silence_frames_per_speaker"],
          "scoring.pred_score_threshold": float(st["prediction_score_threshold"]),
          "scoring.scores_boost_latest": float(st["latest_frames_score_boost"]), "scoring.sil_threshold": 0.2,
          "scoring.strong_boost_rate": float(st["strong_boost_rate"]), "scoring.weak_boost_rate": float(st["weak_boost_rate"]),
          "scoring.min_pos_scores_rate": float(st["min_positive_scores_rate"]),
          "streaming.spkcache_len": st["speaker_cache_length"], "streaming.fifo_len": cfg["fifo_length"],
          "streaming.chunk_len": cfg["chunk_length"], "streaming.spkcache_update_period": cfg["speaker_cache_update_period"],
          "streaming.chunk_left_context": cfg.get("chunk_left_context", 0), "streaming.chunk_right_context": cfg["chunk_right_context"]}
    write_gguf(dst, "sortformer", "Nemotron-3-Diarization", kv, T, outtype=outtype, license="OpenMDW-1.1",
               source_url="https://huggingface.co/nvidia/Nemotron-3-Diarization")
    print(f"wrote {dst}: {L} layers, d_model {D}, {len(T)} tensors")


if __name__ == "__main__":
    main()
