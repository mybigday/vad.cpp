"""Writing vad.cpp GGUF files (graph-ready tensors, see docs/gguf.md)."""
import re
import sys
from collections import Counter

import gguf
import numpy as np
from gguf import GGMLQuantizationType as Q

# general.file_type from the type that most >= 2-D tensors use (llama.cpp's "mostly X")
_FILE_TYPES = {Q.F32: "ALL_F32", Q.F16: "MOSTLY_F16", Q.BF16: "MOSTLY_BF16", Q.Q8_0: "MOSTLY_Q8_0",
               Q.Q4_0: "MOSTLY_Q4_0", Q.Q4_1: "MOSTLY_Q4_1", Q.Q5_0: "MOSTLY_Q5_0", Q.Q5_1: "MOSTLY_Q5_1"}

# Tensors each architecture uses outside mul_mat (elementwise / conv taps / positions).
# They must stay float (F32/F16/BF16); the quantizer leaves them alone unless told otherwise.
# Keep in sync with the f32() calls in src/models/*.cpp.
ELEMENTWISE = {
    "firered": r"^fsmn\d+\.w$",
    "smart_turn": r"^pos$",
}


def is_elementwise(arch, name):
    pat = ELEMENTWISE.get(arch)
    return bool(pat and re.search(pat, name))


def file_type(types_by_ndim):
    """types_by_ndim: [(ndim, GGMLQuantizationType)] -> gguf.LlamaFileType."""
    c = Counter(t for nd, t in types_by_ndim if nd >= 2)
    t = c.most_common(1)[0][0] if c else Q.F32
    return gguf.LlamaFileType[_FILE_TYPES.get(t, "GUESSED")]


def pop_outtype(argv):
    """Removes '--outtype f32|f16|bf16' from argv and returns the type (default F32)."""
    if "--outtype" not in argv:
        return Q.F32
    i = argv.index("--outtype")
    name = argv[i + 1].upper()
    del argv[i:i + 2]
    if name not in ("F32", "F16", "BF16"):
        sys.exit(f"--outtype {name}: use f32, f16 or bf16 (block quantization: scripts/quantize-gguf.py)")
    return Q[name]


def write_gguf(dst, arch, name, kv, tensors, outtype=Q.F32, version=None, license=None, source_url=None):
    """Metadata as in other GGUF ecosystems:
      general.*   name, version, license, source, file type
      <arch>.*    hyperparameters; audio front end under <arch>.preprocessor.*
    kv: {key: str | bool | int | float} without the "<arch>." prefix (ints as u32, floats as f32).
    tensors: {name: np.ndarray in PyTorch order (ggml ne reversed)}.
    outtype: type of the >= 2-D tensors (F32, F16 or BF16); 1-D tensors stay F32."""
    w = gguf.GGUFWriter(dst, arch)
    w.add_name(name)
    if version:
        w.add_version(version)
    if license:
        w.add_license(license)
    if source_url:
        w.add_source_url(source_url)
    for k, v in kv.items():
        if isinstance(v, str):
            w.add_string(f"{arch}.{k}", v)
        elif isinstance(v, bool):
            w.add_bool(f"{arch}.{k}", v)
        elif isinstance(v, float):
            w.add_float32(f"{arch}.{k}", v)
        else:
            w.add_uint32(f"{arch}.{k}", int(v))
    types = []
    for n, t in tensors.items():
        t = np.ascontiguousarray(t, dtype=np.float32)
        ty = outtype if t.ndim >= 2 else Q.F32
        if ty == Q.F32:
            w.add_tensor(n, t)
        elif ty == Q.F16:
            w.add_tensor(n, t.astype(np.float16))
        else:
            q = gguf.quants.quantize(t, ty)
            w.add_tensor(n, q, raw_shape=q.shape, raw_dtype=ty)
        types.append((t.ndim, ty))
    w.add_file_type(file_type(types))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def conv_to_window_layout(w):
    """Conv1d weight (Cout, Cin, K) -> (Cout, K, Cin): matches the graph's [K * Cin] window view."""
    return np.ascontiguousarray(np.transpose(w, (0, 2, 1)))
