// Speaker diarization (Nemotron-3-Diarization, Streaming Sortformer) vs the official
// transformers implementation: offline (vad_detect) and the streaming presets
// (one chunk per step, FIFO + speaker cache), on an 84 s four-speaker recording.
// Devices: VAD_TEST_DEVICES (comma separated, "dev@threads"), default "cpu@8,ggml-cpu@8".
// CPU devices must match the reference within 1e-4. Accelerators compute mul_mat
// at F16: offline within 1e-2, and since rounding can flip a speaker-cache top-k
// choice in long streams, streams are checked by their speaker decisions
// (p > 0.5) instead: at most 0.5% of frame x speaker decisions may differ.
#include "test_common.h"
#include "vad.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

static std::vector<float> read_f32(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    std::vector<float> v((size_t) f.tellg() / sizeof(float));
    f.seekg(0);
    f.read((char *) v.data(), v.size() * sizeof(float));
    return v;
}

// share of frame x speaker activity decisions (p > 0.5) that differ
static double decision_mismatch(const std::vector<float> & a, const std::vector<float> & b, size_t n) {
    size_t bad = 0, m = 0;
    for (size_t i = 0; i < n && i < a.size() && i < b.size(); ++i, ++m) bad += (a[i] > 0.5f) != (b[i] > 0.5f);
    return m ? (double) bad / m : 0.0;
}

static float max_diff(const std::vector<float> & a, const std::vector<float> & b, size_t n) {
    float d = 0;
    for (size_t i = 0; i < n && i < a.size() && i < b.size(); ++i) d = std::max(d, std::fabs(a[i] - b[i]));
    return d;
}

// pushes audio[0, n) in pieces of `piece` samples; flushes when n is the whole audio
static std::vector<float> run_stream(vad_model * m, vad_stream_params sp, const std::vector<float> & audio, size_t n, int piece) {
    const int no = vad_model_n_outputs(m);
    vad_stream * st = vad_stream_new_with_params(m, sp);
    CHECK(st, "stream");
    std::vector<float> out;
    if (!st) return out;
    for (size_t at = 0; at < n; at += piece) {
        const int len = (int) std::min<size_t>(piece, n - at);
        const int k = vad_stream_n_pending(st, len);
        std::vector<float> p((size_t) k * no);
        CHECK(vad_stream_push(st, audio.data() + at, len, p.data(), k) == k, "push");
        out.insert(out.end(), p.begin(), p.end());
    }
    if (n == audio.size()) {
        const int k = vad_stream_n_flush(st);
        std::vector<float> p((size_t) k * no);
        CHECK(vad_stream_flush(st, p.data(), k) == k, "flush");
        out.insert(out.end(), p.begin(), p.end());
    }
    vad_stream_free(st);
    return out;
}

int main() {
    const std::string data = env_or("VAD_TEST_DATA", VAD_TEST_DATA);
    require_file(data + "/nemotron-3-diarization.gguf");
    require_file(data + "/ref/conversation.f32");
    require_file(data + "/ref/conversation.sortformer.offline.probs.f32");
    const std::vector<float> audio = read_f32(data + "/ref/conversation.f32");
    const std::vector<float> ref_off = read_f32(data + "/ref/conversation.sortformer.offline.probs.f32");
    const int NS = 8, SR = 16000, E = 1280;   // speakers, rate, samples per encoder frame
    const size_t n_frames = audio.size() / 160;
    // (reference mode, chunk, look-ahead in encoder frames, seconds of audio to run; 0 = all + flush)
    struct preset { const char * mode; int chunk, rc; double seconds; };
    const preset presets[] = { { "low_latency", 9, 4, 0 }, { "very_low_latency", 6, 2, 20 }, { "ultra_low_latency", 3, 1, 12 } };

    std::vector<std::string> devices;
    {
        const char * env = getenv("VAD_TEST_DEVICES");
        std::stringstream ss(env ? env : "cpu@8,ggml-cpu@8");
        for (std::string d; std::getline(ss, d, ',');) if (!d.empty()) devices.push_back(d);
    }
    // file, checked against the reference (the official Q8_0 GGUF: deviation reported only)
    const std::vector<std::pair<std::string, bool>> files = {
        { "nemotron-3-diarization.gguf", true }, { "official/nemotron/Nemotron-3-Diarization.q8_0.gguf", false } };

    for (const auto & spec : devices) for (const auto & [file, checked] : files) {
        if (!file_exists(data + "/" + file)) continue;
        const size_t at = spec.find('@');
        const std::string dev = spec.substr(0, at);
        vad_model_params mp = vad_model_default_params();
        mp.device = dev.c_str();
        mp.n_threads = at == std::string::npos ? 1 : atoi(spec.c_str() + at + 1);
        const bool cpu_dev = dev == "cpu" || dev == "ggml-cpu";
        vad_model * m = vad_model_load((data + "/" + file).c_str(), mp);
        CHECK(m, "load %s on %s", file.c_str(), spec.c_str());
        if (!m) continue;
        CHECK(vad_model_output(m) == VAD_OUTPUT_SPEAKERS && vad_model_n_outputs(m) == NS, "speaker model");
        CHECK(vad_model_hop_samples(m) == 160 && vad_model_max_lookahead_samples(m) == 40 * E, "model properties");

        // offline: the reference forward's own chunking
        std::vector<float> off(n_frames * NS);
        const int n = vad_detect(m, audio.data(), (int) audio.size(), off.data(), (int) n_frames);
        CHECK(n == (int) n_frames, "offline frames %d", n);
        const float d_off = max_diff(off, ref_off, n_frames * NS);
        fprintf(stderr, "%-10s %-52s %-17s max|p - official| %.2e, decisions differ %.3f%%\n", spec.c_str(), file.c_str(),
                "offline", d_off, 100 * decision_mismatch(off, ref_off, n_frames * NS));
        if (checked) CHECK(d_off < (cpu_dev ? 1e-4f : 1e-2f), "offline deviation %g", d_off);

        if (!checked) { vad_model_free(m); continue; }
        // streaming presets vs the reference's chunk-per-forward streaming
        for (const preset & p : presets) {
            if (p.seconds > 0 && spec != devices[0]) continue;
            const std::vector<float> ref = read_f32(data + "/ref/conversation.sortformer." + p.mode + ".probs.f32");
            vad_stream_params sp = vad_stream_default_params();
            sp.slice_samples = p.chunk * E;               // look-ahead: the recommended one for the slice
            const size_t n_audio = p.seconds > 0 ? (size_t) (p.seconds * SR) : audio.size();
            const std::vector<float> got = run_stream(m, sp, audio, n_audio, 3333);
            const size_t want = p.seconds > 0 ? ((n_audio - (p.rc * E + 144)) / (p.chunk * E)) * p.chunk * 8 : n_frames;
            CHECK(got.size() == want * NS, "%s: %zu frames, expected %zu", p.mode, got.size() / NS, want);
            const size_t nc = std::min(got.size(), want * NS);
            const float d = max_diff(got, ref, nc);
            const double mis = decision_mismatch(got, ref, nc);
            fprintf(stderr, "%-10s %-52s %-17s max|p - official| %.2e, decisions differ %.3f%% (%zu frames)\n", spec.c_str(),
                    file.c_str(), p.mode, d, 100 * mis, got.size() / NS);
            if (cpu_dev) CHECK(d < 1e-4f, "%s deviation %g", p.mode, d);
            else CHECK(mis <= 0.005, "%s: %.3f%% of the decisions differ", p.mode, 100 * mis);
        }
        // the offline configuration as a stream: vad_detect but for the last chunk
        {
            vad_stream_params sp = vad_stream_default_params();
            sp.slice_samples = 340 * E;
            sp.lookahead_samples = 40 * E;
            const std::vector<float> got = run_stream(m, sp, audio, audio.size(), 16000);
            CHECK(got.size() == n_frames * NS, "offline-config stream frames");
            const size_t exact = (n_frames / (340 * 8) - 1) * 340 * 8;   // frames before the last full chunk
            const float d = max_diff(got, off, exact * NS);
            CHECK(d == 0.0f, "offline-config stream differs from vad_detect: %g", d);
        }
        // several streams through vad_streams_push (encoder steps batched across streams) == each alone
        if (spec == devices[0]) {
            const int S = 3;
            const size_t n_audio = 30 * SR, piece = 3333;
            std::vector<std::vector<float>> au(S), alone(S), got(S);
            std::vector<vad_stream *> st(S);
            vad_stream_params sp = vad_stream_default_params();
            for (int i = 0; i < S; ++i) {
                au[i].assign(audio.begin() + (size_t) i * 7 * SR, audio.begin() + (size_t) i * 7 * SR + n_audio);
                alone[i] = run_stream(m, sp, au[i], n_audio - 1, piece);   // no flush: n < size
                st[i] = vad_stream_new_with_params(m, sp);
            }
            for (size_t at = 0; at < n_audio - 1; at += piece) {
                const int len = (int) std::min<size_t>(piece, n_audio - 1 - at);
                std::vector<const float *> ptr(S);
                std::vector<int> lens(S, len), maxp(S), nout(S);
                std::vector<std::vector<float>> out(S);
                std::vector<float *> outp(S);
                for (int i = 0; i < S; ++i) {
                    ptr[i] = au[i].data() + at;
                    maxp[i] = vad_stream_n_pending(st[i], len);
                    out[i].resize((size_t) maxp[i] * NS + 1);
                    outp[i] = out[i].data();
                }
                CHECK(vad_streams_push(st.data(), S, ptr.data(), lens.data(), outp.data(), maxp.data(), nout.data()) == 0, "streams_push");
                for (int i = 0; i < S; ++i) got[i].insert(got[i].end(), out[i].begin(), out[i].begin() + (size_t) nout[i] * NS);
            }
            float d = 0;
            for (int i = 0; i < S; ++i) {
                CHECK(got[i].size() == alone[i].size() && !got[i].empty(), "batched stream %d: %zu vs %zu values", i, got[i].size(), alone[i].size());
                d = std::max(d, max_diff(got[i], alone[i], got[i].size()));
                vad_stream_free(st[i]);
            }
            fprintf(stderr, "%-10s %-52s %-17s max|p - alone| %.2e\n", spec.c_str(), file.c_str(), "batched x3", d);
            CHECK(d < (cpu_dev ? 1e-4f : 1e-2f), "batched streams vs alone %g", d);
        }
        // look-ahead beyond the model's maximum is refused
        {
            vad_stream_params sp = vad_stream_default_params();
            sp.lookahead_samples = 41 * E;
            vad_stream * st = vad_stream_new_with_params(m, sp);
            CHECK(!st, "excessive look-ahead accepted");
            vad_stream_free(st);
        }
        vad_model_free(m);
    }
    return finish("test_diarization");
}
