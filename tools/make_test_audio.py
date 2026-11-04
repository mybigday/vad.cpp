"""Build assets/test_16k.wav: jfk.wav + codec.cpp test_long.wav (resampled) with silences."""
import sys
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

parts = []
for p in sys.argv[2:]:
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != 16000:
        g = np.gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    parts += [x, np.zeros(16000 * 2, np.float32)]
y = np.concatenate(parts)
sf.write(sys.argv[1], y, 16000, subtype="FLOAT")
print(sys.argv[1], len(y) / 16000, "s")
