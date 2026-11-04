// Every speech model through the common API: parity with the official
// implementation, streaming == offline, batched streams == single.
// Devices: VAD_TEST_DEVICES (comma separated, "dev@threads"), default "cpu,cpu@4,ggml-cpu".
#include "test_common.h"
#include "vad.h"
#include "wav.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <sstream>
#include <vector>

static float max_diff(const float * a, const float * b, int n) {
    float m = 0;
    for (int i = 0; i < n; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

// tol < 0: quantized file, quality not guaranteed: deviation is reported, not checked
struct model_case { std::string file, ref; float tol; };

int main() {
    const std::string data = env_or("VAD_TEST_DATA", VAD_TEST_DATA), assets = env_or("VAD_ASSETS", VAD_ASSETS);
    require_file(data + "/ref/jfk.silero-v6.2.probs.f32");

    std::vector<float> audio, tmp;
    std::string err;
    if (!read_audio(assets + "/jfk.wav", 16000, audio, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }

    // whisper.cpp files store conv/STFT weights as F16, so they can't match the
    // F32 ONNX model exactly; the tolerance covers that quantization only.
    const std::vector<model_case> cases = {
        { "silero-v6.2.gguf",         "jfk.silero-v6.2.probs.f32",   1e-4f },
        { "silero-v5.1.2.gguf",       "jfk.silero-v5.1.2.probs.f32", 1e-4f },
        // (CPUs with fp16 arithmetic, e.g. ARMv8.2, run F16 weights with F16 activations: 2e-2, see below)
        { "ggml-silero-v6.2.0.bin",   "jfk.silero-v6.2.probs.f32",   5e-3f },
        { "ggml-silero-v5.1.2.bin",   "jfk.silero-v5.1.2.probs.f32", 5e-3f },
        // official features come from kaldi-native-fbank (float32 FFT); log-mel differs by ~1e-5 on average
        { "firered-stream-vad.gguf",  "jfk.firered-stream.probs.f32", 1e-4f },
        { "firered-vad.gguf",         "jfk.firered-vad.probs.f32",    1e-4f },   // offline only (lookahead)
        // quantized files from scripts/quantize-gguf.py
        { "silero-v6.2-f16.gguf",     "jfk.silero-v6.2.probs.f32",    -1 },
        { "silero-v6.2-q8_0.gguf",    "jfk.silero-v6.2.probs.f32",    -1 },
        { "firered-stream-vad-f16.gguf", "jfk.firered-stream.probs.f32", -1 },
        { "firered-stream-vad-q8_0.gguf", "jfk.firered-stream.probs.f32", -1 },
        { "firered-vad-q8_0.gguf",    "jfk.firered-vad.probs.f32",    -1 },
    };

    std::vector<std::string> devices;
    {
        const char * env = getenv("VAD_TEST_DEVICES");
        std::stringstream ss(env ? env : "cpu,cpu@4,ggml-cpu");
        for (std::string d; std::getline(ss, d, ',');) if (!d.empty()) devices.push_back(d);
    }

    for (const auto & spec : devices) {
        // "dev[@threads]"
        const std::string spec0 = spec;
        const size_t at = spec0.find('@');
        const std::string dev = spec0.substr(0, at);
        const int n_threads = at == std::string::npos ? 1 : atoi(spec0.c_str() + at + 1);
        for (const auto & mc : cases) {
            const std::string path = data + "/" + mc.file;
            if (!file_exists(path)) { fprintf(stderr, "skip %s (missing)\n", mc.file.c_str()); continue; }
            std::vector<float> ref;
            read_f32(data + "/ref/" + mc.ref, ref);

            vad_model_params mp = vad_model_default_params();
            mp.device = dev.c_str();
            mp.n_threads = n_threads;
            vad_model * m = vad_model_load(path.c_str(), mp);
            CHECK(m, "load %s on %s", mc.file.c_str(), dev.c_str());
            if (!m) continue;
            const int cs = vad_model_hop_samples(m), wu = vad_model_warmup_samples(m);
            const bool streaming = vad_model_max_lookahead_samples(m) == 0;   // causal: exact per-hop streaming
            // offline count: streaming models zero pad the last chunk, offline-only models use full frames
            const int n = streaming ? (int) ((audio.size() - wu + cs - 1) / cs) : (int) ((audio.size() - wu) / cs);
            const int nf = (int) ((audio.size() - wu) / cs);           // full chunks
            CHECK(nf <= (int) ref.size() && (int) ref.size() <= n, "reference length %zu vs %d..%d", ref.size(), nf, n);

            // offline (the official references may omit the zero-padded last chunk)
            std::vector<float> off(n);
            CHECK(vad_detect(m, audio.data(), (int) audio.size(), off.data(), n) == n, "vad_detect count");
            const int nr = std::min<int>(n, (int) ref.size());
            // accelerators: ggml-vulkan computes mul_mat at F16 precision (and rejects
            // F32-only mul_mat), giving ~4e-3; CPU backends must match ONNX closely
            const bool cpu_dev = dev == "cpu" || dev == "ggml-cpu";
            // F16 weights: ~1e-3 deviation from the F32 reference is expected
            const bool quantized = mc.tol < 0;
            const bool f16_file = mc.file.find(".bin") != std::string::npos;
            const float tol = quantized ? 1e9f : !cpu_dev ? 1e-2f : f16_file ? 2e-2f : mc.tol;
            // F16 / quantized weights may run with reduced-precision activations (fp16, int8)
            // where the CPU supports it, so streaming and offline agree only approximately
            const float tol_same = cpu_dev && !quantized && !f16_file ? 1e-5f : 1e-2f;
            const float d_off = max_diff(off.data(), ref.data(), nr);
            CHECK(d_off <= tol, "%s %s: offline vs ONNX %.3g > %.3g", spec.c_str(), mc.file.c_str(), d_off, tol);

            if (!streaming) {
                // sliding-window streaming: push (random sizes) + flush == vad_detect with the maximum
                // lookahead; with less lookahead the deviation is reported (accuracy/latency trade-off)
                auto run_stream = [&](int lookahead, int slice, std::vector<float> & got, const std::vector<float> * src = nullptr) {
                    const std::vector<float> & audio_in = src ? *src : audio;
                    vad_stream_params sp = vad_stream_default_params();
                    sp.lookahead_samples = lookahead;
                    sp.slice_samples = slice;
                    vad_stream * st = vad_stream_new_with_params(m, sp);
                    CHECK(st, "window stream");
                    if (!st) return;
                    std::mt19937 r3(5);
                    size_t at = 0;
                    while (at < audio_in.size()) {
                        const int len = std::min<int>((int) (audio_in.size() - at), std::uniform_int_distribution<int>(1, 6000)(r3));
                        std::vector<float> p(vad_stream_n_pending(st, len));
                        const int g = vad_stream_push(st, audio_in.data() + at, len, p.data(), (int) p.size());
                        CHECK(g == (int) p.size(), "window push %d vs %zu", g, p.size());
                        got.insert(got.end(), p.begin(), p.end());
                        at += len;
                    }
                    std::vector<float> p(vad_stream_n_flush(st));
                    CHECK(vad_stream_flush(st, p.data(), (int) p.size()) == (int) p.size(), "flush");
                    got.insert(got.end(), p.begin(), p.end());
                    vad_stream_free(st);
                };
                std::vector<float> full;
                run_stream(-1, 4000, full);
                CHECK((int) full.size() == n, "window stream produced %zu, expected %d", full.size(), n);
                const float d_w = max_diff(full.data(), off.data(), std::min<int>(n, (int) full.size()));
                CHECK(d_w <= tol_same, "%s %s: window stream vs offline %.3g", spec.c_str(), mc.file.c_str(), d_w);
                std::string extra;
                for (int la_ms : { 0, 400, 1000 }) {
                    std::vector<float> g;
                    run_stream(la_ms * 16, 8000, g);
                    char b[64];
                    snprintf(b, sizeof b, "  la %dms %.2e", la_ms, max_diff(g.data(), off.data(), std::min<int>(n, (int) g.size())));
                    extra += b;
                }
                // several window streams through vad_streams_push (windows batched) == each alone
                {
                    const int S = 3;
                    std::vector<std::vector<float>> au(S), alone(S), got(S);
                    std::vector<vad_stream *> st(S);
                    vad_stream_params sp = vad_stream_default_params();
                    sp.lookahead_samples = 400 * 16;
                    sp.slice_samples = 4000;
                    for (int i = 0; i < S; ++i) {
                        au[i] = audio;
                        std::rotate(au[i].begin(), au[i].begin() + (size_t) i * 7919 % au[i].size(), au[i].end());
                        st[i] = vad_stream_new_with_params(m, sp);
                    }
                    // the same push sizes for the batched and the single runs (with less than the
                    // maximum lookahead, results depend on when windows run)
                    std::vector<std::vector<int>> lens_of(S);
                    {
                        std::mt19937 r4(9);
                        std::vector<size_t> pos(S, 0);
                        for (bool more = true; more;) {
                            more = false;
                            for (int i = 0; i < S; ++i) {
                                const int l = std::min<int>((int) (au[i].size() - pos[i]), std::uniform_int_distribution<int>(1, 6000)(r4));
                                lens_of[i].push_back(l);
                                pos[i] += l;
                                more |= pos[i] < au[i].size();
                            }
                        }
                    }
                    for (int i = 0; i < S; ++i) {
                        vad_stream * one = vad_stream_new_with_params(m, sp);
                        size_t pos = 0;
                        for (int l : lens_of[i]) {
                            std::vector<float> p(vad_stream_n_pending(one, l) + 1);
                            const int g = vad_stream_push(one, au[i].data() + pos, l, p.data(), (int) p.size());
                            alone[i].insert(alone[i].end(), p.begin(), p.begin() + std::max(0, g));
                            pos += l;
                        }
                        std::vector<float> p(vad_stream_n_flush(one));
                        vad_stream_flush(one, p.data(), (int) p.size());
                        alone[i].insert(alone[i].end(), p.begin(), p.end());
                        vad_stream_free(one);
                    }
                    size_t step = 0;
                    std::vector<size_t> at(S, 0);
                    for (bool more = true; more;) {
                        more = false;
                        std::vector<const float *> ptr(S);
                        std::vector<int> len(S), maxp(S), nout(S);
                        std::vector<std::vector<float>> out(S);
                        std::vector<float *> outp(S);
                        for (int i = 0; i < S; ++i) {
                            len[i] = lens_of[i][step];
                            ptr[i] = au[i].data() + at[i];
                            out[i].resize(vad_stream_n_pending(st[i], len[i]) + 1);
                            outp[i] = out[i].data(); maxp[i] = (int) out[i].size();
                        }
                        CHECK(vad_streams_push(st.data(), S, ptr.data(), len.data(), outp.data(), maxp.data(), nout.data()) == 0, "window streams_push");
                        ++step;
                        for (int i = 0; i < S; ++i) {
                            got[i].insert(got[i].end(), out[i].begin(), out[i].begin() + nout[i]);
                            at[i] += len[i];
                            more |= at[i] < au[i].size();
                        }
                    }
                    float d_b = 0;
                    for (int i = 0; i < S; ++i) {
                        std::vector<float> p(vad_stream_n_flush(st[i]));
                        CHECK(vad_stream_flush(st[i], p.data(), (int) p.size()) == (int) p.size(), "flush");
                        got[i].insert(got[i].end(), p.begin(), p.end());
                        CHECK(got[i].size() == alone[i].size(), "batched window stream %d: %zu vs %zu frames", i, got[i].size(), alone[i].size());
                        d_b = std::max(d_b, max_diff(got[i].data(), alone[i].data(), (int) std::min(got[i].size(), alone[i].size())));
                        vad_stream_free(st[i]);
                    }
                    CHECK(d_b <= tol_same, "%s %s: batched window streams vs alone %.3g", spec.c_str(), mc.file.c_str(), d_b);
                    char b[48];
                    snprintf(b, sizeof b, "  batched %.2e", d_b);
                    extra += b;
                }
                fprintf(stderr, "%-9s %-28s vs-official %.2e%s  window-stream %.2e%s\n", spec.c_str(), mc.file.c_str(), d_off,
                        quantized ? " (not checked)" : "", d_w, extra.c_str());
                vad_model_free(m);
                continue;
            }
            // streaming in random-sized pushes == offline (full chunks only)
            vad_stream * s = vad_stream_new(m);
            std::mt19937 rng(42);
            std::vector<float> st;
            size_t pos = 0;
            while (pos < audio.size()) {
                const int len = std::min<int>((int) (audio.size() - pos), std::uniform_int_distribution<int>(1, 3000)(rng));
                std::vector<float> p(vad_stream_n_pending(s, len));
                const int got = vad_stream_push(s, audio.data() + pos, len, p.data(), (int) p.size());
                CHECK(got == (int) p.size(), "push returned %d, expected %zu", got, p.size());
                st.insert(st.end(), p.begin(), p.begin() + std::max(got, 0));
                pos += len;
            }
            CHECK((int) st.size() == nf, "stream produced %zu, expected %d", st.size(), nf);
            const float d_st = max_diff(st.data(), off.data(), std::min<int>(nf, (int) st.size()));
            CHECK(d_st <= tol_same, "%s %s: stream vs offline %.3g", spec.c_str(), mc.file.c_str(), d_st);

            // batched streams: 3 streams at different offsets == each alone
            const int S = 3, steps = 100;
            std::vector<vad_stream *> ss(S);
            std::vector<std::vector<float>> single(S);
            for (int k = 0; k < S; ++k) {
                const float * a0 = audio.data() + (size_t) k * 37 * cs;
                ss[k] = vad_stream_new(m);
                CHECK(vad_stream_push(ss[k], a0, wu, nullptr, 0) == 0, "warmup push");
                vad_stream * one = vad_stream_new(m);
                single[k].resize(steps);
                CHECK(vad_stream_push(one, a0, wu + steps * cs, single[k].data(), steps) == steps, "single push");
                vad_stream_free(one);
            }
            float d_b = 0;
            for (int t = 0; t < steps; ++t) {
                const float * chunks[S];
                float pr[S];
                for (int k = 0; k < S; ++k) chunks[k] = audio.data() + ((size_t) k * 37 + t) * cs + wu;
                int lens[S], maxp[S], nout[S];
                float * outp[S];
                for (int k = 0; k < S; ++k) { lens[k] = cs; maxp[k] = 1; outp[k] = &pr[k]; }
                CHECK(vad_streams_push(ss.data(), S, chunks, lens, outp, maxp, nout) == 0, "vad_streams_push");
                for (int k = 0; k < S; ++k) CHECK(nout[k] == 1, "streams_push output count");
                for (int k = 0; k < S; ++k) d_b = std::max(d_b, std::fabs(pr[k] - single[k][t]));
            }
            CHECK(d_b <= tol_same, "%s %s: batched vs single %.3g", spec.c_str(), mc.file.c_str(), d_b);
            for (auto & x : ss) vad_stream_free(x);
            vad_stream_free(s);

            // state residency: two streams interleaved with random push sizes (moves
            // between graphs and columns, evictions, a reset) == each stream alone.
            // Both streams start at sample 0, so each must reproduce the offline result.
            float d_r = 0;
            {
                vad_stream * x[2] = { vad_stream_new(m), vad_stream_new(m) };
                std::vector<float> got[2];
                size_t at[2] = { 0, 0 };
                std::mt19937 r2(7);
                for (int it = 0; it < 80; ++it) {
                    const int k = (int) (r2() % 2);
                    const int len = std::uniform_int_distribution<int>(1, 9 * cs)(r2);
                    if (at[k] + len > audio.size()) continue;
                    std::vector<float> p(vad_stream_n_pending(x[k], len));
                    vad_stream_push(x[k], audio.data() + at[k], len, p.data(), (int) p.size());
                    got[k].insert(got[k].end(), p.begin(), p.end());
                    at[k] += len;
                    if (it == 40) { vad_stream_reset(x[1]); got[1].clear(); at[1] = 0; }
                }
                for (int k = 0; k < 2; ++k) {
                    d_r = std::max(d_r, max_diff(got[k].data(), off.data(), std::min<int>((int) got[k].size(), nf)));
                    vad_stream_free(x[k]);
                }
            }
            CHECK(d_r <= tol_same, "%s %s: interleaved streams vs alone %.3g", spec.c_str(), mc.file.c_str(), d_r);

            fprintf(stderr, "%-9s %-28s vs-official %.2e%s  stream %.2e  batch %.2e  interleaved %.2e\n",
                    spec.c_str(), mc.file.c_str(), d_off, quantized ? " (not checked)" : "", d_st, d_b, d_r);
            vad_model_free(m);
        }
    }
    return finish("test_models");
}
