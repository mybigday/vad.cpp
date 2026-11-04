#include "wav.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>

bool read_f32(const std::string & path, std::vector<float> & out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    const std::streamsize n = f.tellg();
    out.resize(n / sizeof(float));
    f.seekg(0);
    return (bool) f.read((char *) out.data(), out.size() * sizeof(float));
}

static std::vector<float> resample(const std::vector<float> & x, int from, int to) {
    // windowed-sinc (Hann, 16 zero crossings), cutoff at the lower Nyquist
    const double ratio = (double) to / from;
    const double cutoff = std::min(1.0, ratio);
    const int half = 16;
    const size_t n_out = (size_t) std::floor(x.size() * ratio);
    std::vector<float> y(n_out);
    for (size_t i = 0; i < n_out; ++i) {
        const double t = i / ratio;
        const long c = (long) std::floor(t);
        const int w = (int) std::ceil(half / cutoff);
        double acc = 0, norm = 0;
        for (long j = c - w + 1; j <= c + w; ++j) {
            const double d = (t - j) * cutoff;
            if (std::fabs(d) >= half) continue;
            const double s = d == 0 ? 1.0 : std::sin(M_PI * d) / (M_PI * d);
            const double win = 0.5 + 0.5 * std::cos(M_PI * d / half);
            const double k = s * win;
            norm += k;
            if (j >= 0 && j < (long) x.size()) acc += k * x[j];
        }
        y[i] = (float) (norm > 0 ? acc / norm : 0);
    }
    return y;
}

bool read_audio(const std::string & path, int target_rate, std::vector<float> & out, std::string & err) {
    if (path.size() > 4 && path.compare(path.size() - 4, 4, ".f32") == 0) {
        if (!read_f32(path, out)) { err = "cannot read " + path; return false; }
        return true;
    }
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = "cannot open " + path; return false; }
    char riff[12];
    if (!f.read(riff, 12) || std::memcmp(riff, "RIFF", 4) || std::memcmp(riff + 8, "WAVE", 4)) { err = "not a WAV file"; return false; }

    uint16_t fmt = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    std::vector<uint8_t> data;
    while (f) {
        char id[4]; uint32_t size = 0;
        if (!f.read(id, 4) || !f.read((char *) &size, 4)) break;
        if (!std::memcmp(id, "fmt ", 4)) {
            std::vector<uint8_t> b(size);
            f.read((char *) b.data(), size);
            if (size < 16) { err = "bad fmt chunk"; return false; }
            std::memcpy(&fmt, &b[0], 2); std::memcpy(&channels, &b[2], 2);
            std::memcpy(&rate, &b[4], 4); std::memcpy(&bits, &b[14], 2);
            if (fmt == 0xFFFE && size >= 26) std::memcpy(&fmt, &b[24], 2);   // WAVE_FORMAT_EXTENSIBLE subformat
        } else if (!std::memcmp(id, "data", 4)) {
            data.resize(size);
            f.read((char *) data.data(), size);
            data.resize(f.gcount());
            break;
        } else {
            f.seekg(size + (size & 1), std::ios::cur);
        }
    }
    if (!channels || !rate || data.empty()) { err = "WAV without fmt/data"; return false; }
    const int bps = bits / 8;
    const size_t n = data.size() / (bps * channels);
    out.assign(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        double acc = 0;
        for (int c = 0; c < channels; ++c) {
            const uint8_t * p = &data[(i * channels + c) * bps];
            double v;
            if (fmt == 3 && bits == 32)      { float s; std::memcpy(&s, p, 4); v = s; }
            else if (fmt == 3 && bits == 64) { double s; std::memcpy(&s, p, 8); v = s; }
            else if (fmt == 1 && bits == 8)  v = (p[0] - 128) / 128.0;
            else if (fmt == 1 && bits == 16) { int16_t s; std::memcpy(&s, p, 2); v = s / 32768.0; }
            else if (fmt == 1 && bits == 24) { int32_t s = (int32_t) ((uint32_t) p[0] << 8 | (uint32_t) p[1] << 16 | (uint32_t) p[2] << 24) >> 8; v = s / 8388608.0; }
            else if (fmt == 1 && bits == 32) { int32_t s; std::memcpy(&s, p, 4); v = s / 2147483648.0; }
            else { err = "unsupported WAV sample format"; return false; }
            acc += v;
        }
        out[i] = (float) (acc / channels);
    }
    if ((int) rate != target_rate) out = resample(out, (int) rate, target_rate);
    return true;
}
