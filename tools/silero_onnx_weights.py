"""Extract the 16 kHz Silero VAD weights from an official silero_vad.onnx."""
import sys
import numpy as np
import onnx
from onnx import numpy_helper


def load(path):
    m = onnx.load(path)
    w = {}

    def walk(g):
        for n in g.node:
            if n.op_type == "Constant" and n.output[0].count("__") and "Inline_0__" in n.output[0]:
                name = n.output[0].split("Inline_0__", 1)[1]
                if "/" in name:
                    continue
                for a in n.attribute:
                    if a.name == "value":
                        w.setdefault(branch[0] + name, numpy_helper.to_array(a.t))
            for a in n.attribute:
                if a.g.node:
                    walk(a.g)

    # top-level If: then = sr==8000? find which branch is 16k by basis size
    for n in m.graph.node:
        if n.op_type == "If":
            for a in n.attribute:
                branch = [a.name + ":"]
                walk(a.g)
    out = {}
    for k, v in w.items():
        br, name = k.split(":", 1)
        out.setdefault(br, {})[name] = v
    # 16 kHz branch: STFT filter length 256 -> basis (258, 1, 256)
    for br, d in out.items():
        if d["stft.forward_basis_buffer"].shape[-1] == 256:
            return d
    raise RuntimeError("16k branch not found")


if __name__ == "__main__":
    for k, v in load(sys.argv[1]).items():
        print(k, v.shape, v.dtype)
