"""Convert an official silero_vad.onnx (16 kHz branch) to vad.cpp GGUF .

usage: convert_silero_onnx.py silero_vad.onnx out.gguf VERSION [--outtype f32|f16|bf16]
"""
import sys

from gguf_common import conv_to_window_layout, pop_outtype, write_gguf
from silero_onnx_weights import load

outtype = pop_outtype(sys.argv)
src, dst, version = sys.argv[1], sys.argv[2], sys.argv[3]
W = load(src)
T = {"stft.basis": W["stft.forward_basis_buffer"][:, 0, :]}              # (258, 256)
for l in range(4):
    T[f"enc{l}.w"] = conv_to_window_layout(W[f"encoder.{l}.reparam_conv.weight"])
    T[f"enc{l}.b"] = W[f"encoder.{l}.reparam_conv.bias"]
T["lstm.w_ih"] = W["decoder.rnn.weight_ih"]
T["lstm.w_hh"] = W["decoder.rnn.weight_hh"]
T["lstm.b"] = W["decoder.rnn.bias_ih"] + W["decoder.rnn.bias_hh"]
T["dec.w"] = W["decoder.decoder.2.weight"].reshape(1, 128)
T["dec.b"] = W["decoder.decoder.2.bias"]
write_gguf(dst, "silero", f"Silero VAD {version}",
           {"version": version.split(".")[0], "preprocessor.sample_rate": 16000, "chunk_size": 512, "context_size": 64}, T, outtype=outtype,
           version=version, license="MIT", source_url="https://github.com/snakers4/silero-vad")
print("wrote", dst)
