#!/usr/bin/env python3
"""Requantize a vad.cpp GGUF per tensor.

usage: quantize-gguf.py in.gguf out.gguf --type Q8_0 [--tensor-type 'REGEX=TYPE' ...]

Rules, first match wins:
  --tensor-type REGEX=TYPE  (repeatable, re.search on the tensor name; TYPE may be
                            F32 or KEEP = leave the source type)
  --type TYPE               every other tensor that can take it: at least 2-D, row
                            length (ne0) a multiple of the block size, and not one
                            the architecture uses outside mul_mat (tools/gguf_common.py
                            ELEMENTWISE, e.g. FireRed's FSMN taps; those take F16/BF16
                            but are never block-quantized by default)
  otherwise                 the source type is kept
Forcing a type onto a tensor the model cannot use makes vad_model_load fail and
name the tensor. The input may be at any type; non-F32 tensors that get a new type
are dequantized first.

vad.cpp computes at the precision stored in the file. Quantization does not
guarantee quality: only the original precision is checked against the reference
implementations. Which tensors to quantize is the application's choice.
general.file_type is recomputed from the written tensors.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import gguf
from gguf import GGMLQuantizationType as Q
from gguf.constants import GGML_QUANT_SIZES

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from gguf_common import file_type, is_elementwise  # noqa: E402

FLOAT = {Q.F32, Q.F16, Q.BF16}
KEEP = "KEEP"


def field_value(f):
    t = f.types[0]
    if t == gguf.GGUFValueType.STRING:
        return bytes(f.parts[f.data[0]]).decode()
    if t == gguf.GGUFValueType.ARRAY:
        return [bytes(f.parts[i]).decode() if f.types[1] == gguf.GGUFValueType.STRING else f.parts[i][0].item() for i in f.data]
    return f.parts[f.data[0]][0].item()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--type", default="Q8_0", help="default type (e.g. F16, BF16, Q8_0, Q4_0)")
    ap.add_argument("--tensor-type", action="append", default=[], help="REGEX=TYPE override (TYPE may be KEEP), repeatable")
    a = ap.parse_args()

    qtype = lambda name: KEEP if name.upper() == KEEP else Q[name.upper()]
    default = qtype(a.type)
    overrides = []
    for o in a.tensor_type:
        pat, _, t = o.rpartition("=")
        overrides.append((re.compile(pat), qtype(t)))

    r = gguf.GGUFReader(a.src)
    arch = field_value(r.fields["general.architecture"])
    w = gguf.GGUFWriter(a.dst, arch)
    for name, f in r.fields.items():
        if name.startswith("GGUF.") or name in ("general.architecture", "general.file_type", "general.quantization_version"):
            continue
        t = f.types[0]
        v = field_value(f)
        if t == gguf.GGUFValueType.ARRAY:
            w.add_array(name, v)
        elif t == gguf.GGUFValueType.STRING:
            w.add_string(name, v)
        else:
            w.add_key_value(name, v, t)

    types = []
    for t in r.tensors:
        src = Q(t.tensor_type)
        shape = [int(x) for x in reversed(t.shape)]       # PyTorch order
        target = None
        for pat, ty in overrides:
            if pat.search(t.name):
                target = ty
                break
        if target is None:
            ok = len(shape) >= 2 and shape[-1] % GGML_QUANT_SIZES[default][0] == 0
            if is_elementwise(arch, t.name) and default not in FLOAT:
                ok = False
            target = default if ok else KEEP
        if target == KEEP or target == src:
            target = src
            w.add_tensor(t.name, np.asarray(t.data), raw_shape=t.data.shape, raw_dtype=src)
        else:
            data = np.asarray(t.data) if src == Q.F32 else gguf.quants.dequantize(t.data, src)
            data = np.ascontiguousarray(data, dtype=np.float32).reshape(shape)
            if target == Q.F32:
                w.add_tensor(t.name, data)
            elif target == Q.F16:
                w.add_tensor(t.name, data.astype(np.float16))
            else:
                q = gguf.quants.quantize(data, target)
                w.add_tensor(t.name, q, raw_shape=q.shape, raw_dtype=target)
        types.append((len(shape), target))
        print(f"{t.name:28s} {str(shape):16s} {src.name:>5s} -> {target.name}")
    w.add_file_type(file_type(types))
    if any(ty not in FLOAT for _, ty in types):
        w.add_quantization_version(gguf.GGML_QUANT_VERSION)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    print("wrote", a.dst)


if __name__ == "__main__":
    main()
