// Turn detection (smart-turn-v3) vs the official pipecat inference
// (WhisperFeatureExtractor + ONNX Runtime), single and batched.
// Devices: VAD_TEST_DEVICES (comma separated, "dev@threads"), default "cpu,cpu@4,ggml-cpu".
#include "test_common.h"
#include "vad.h"
#include "wav.h"

#include <climits>
#include <cmath>
#include <sstream>
#include <vector>

// probability for the audio ending at n: a stream with no slices, push, flush
static float end_prob(vad_model * m, const float * x, int n) {
    vad_stream_params sp = vad_stream_default_params();
    sp.slice_samples = INT32_MAX;
    float p = -1;
    CHECK(vad_detect_with_params(m, sp, x, n, &p, 1) == 1, "detect (end only)");
    return p;
}

int main() {
    const std::string data = env_or("VAD_TEST_DATA", VAD_TEST_DATA), assets = env_or("VAD_ASSETS", VAD_ASSETS);
    require_file(data + "/ref/smart-turn.txt");
    require_file(data + "/smart-turn-v3.2.gguf");
    // file, checked against the reference (quantized files: deviation reported only)
    const std::vector<std::pair<std::string, bool>> files = {
        { "smart-turn-v3.2.gguf", true }, { "smart-turn-v3.2-f16.gguf", false }, { "smart-turn-v3.2-q8_0.gguf", false } };

    struct ref_case { std::string file; int end; float prob; };
    std::vector<ref_case> cases;
    {
        std::ifstream f(data + "/ref/smart-turn.txt");
        for (ref_case r; f >> r.file >> r.end >> r.prob;) cases.push_back(r);
    }
    CHECK(!cases.empty(), "no reference cases");
    std::vector<float> audio;
    std::string err;
    if (!read_audio(assets + "/" + cases[0].file, 16000, audio, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }

    std::vector<std::string> devices;
    {
        const char * env = getenv("VAD_TEST_DEVICES");
        std::stringstream ss(env ? env : "cpu,cpu@4,ggml-cpu");
        for (std::string d; std::getline(ss, d, ',');) if (!d.empty()) devices.push_back(d);
    }
    for (const auto & spec : devices) for (const auto & [file, checked] : files) {
        if (!file_exists(data + "/" + file)) continue;
        // "dev[@threads]"
        const std::string spec0 = spec;
        const size_t at = spec0.find('@');
        const std::string dev = spec0.substr(0, at);
        vad_model_params mp = vad_model_default_params();
        mp.device = dev.c_str();
        mp.n_threads = at == std::string::npos ? 1 : atoi(spec0.c_str() + at + 1);
        vad_model * m = vad_model_load((data + "/" + file).c_str(), mp);
        CHECK(m, "load on %s", spec.c_str());
        if (!m) continue;
        CHECK(vad_model_output(m) == VAD_OUTPUT_TURN_END, "model output");
        // streaming: one probability per slice (window ending at the boundary), flush = window ending now
        {
            vad_stream_params sp = vad_stream_default_params();
            sp.slice_samples = 24000;
            vad_stream * st = vad_stream_new_with_params(m, sp);
            CHECK(st, "turn stream");
            std::vector<float> got;
            size_t at = 0;
            const size_t n_audio = std::min<size_t>(audio.size(), 100000);
            while (at < n_audio) {
                const int len = (int) std::min<size_t>(n_audio - at, 7777);
                std::vector<float> p(vad_stream_n_pending(st, len));
                CHECK(vad_stream_push(st, audio.data() + at, len, p.data(), (int) p.size()) == (int) p.size(), "turn push");
                got.insert(got.end(), p.begin(), p.end());
                at += len;
            }
            float fl = -1;
            CHECK(vad_stream_flush(st, &fl, 1) == 1, "turn flush");
            vad_stream_free(st);
            float d_s = 0;
            CHECK(got.size() == n_audio / 24000, "turn stream produced %zu", got.size());
            for (size_t i = 0; i < got.size(); ++i) {
                d_s = std::max(d_s, std::fabs(end_prob(m, audio.data(), (int) ((i + 1) * 24000)) - got[i]));
            }
            d_s = std::max(d_s, std::fabs(end_prob(m, audio.data(), (int) n_audio) - fl));
            CHECK(d_s <= 1e-5f || !(dev == "cpu" || dev == "ggml-cpu"), "%s: turn stream vs predict %.3g", spec.c_str(), d_s);
            fprintf(stderr, "%-9s %-26s turn stream (%zu slices + flush) vs predict %.2e\n", spec.c_str(), file.c_str(), got.size(), d_s);
        }
        const bool cpu_dev = dev == "cpu" || dev == "ggml-cpu";
        const float tol = !checked ? 1e9f : cpu_dev ? 1e-4f : 2e-2f;

        float worst = 0;
        std::vector<const float *> ptrs;
        std::vector<int> lens;
        for (const auto & r : cases) {
            worst = std::max(worst, std::fabs(end_prob(m, audio.data(), r.end) - r.prob));
            ptrs.push_back(audio.data());
            lens.push_back(r.end);
        }
        // batched: one stream per case whose slice ends exactly at the case end -> one window each, one compute
        std::vector<float> pb(cases.size());
        {
            const int n = (int) cases.size();
            std::vector<vad_stream *> st(n);
            std::vector<float *> outp(n);
            std::vector<int> maxp(n, 1), nout(n);
            for (int i = 0; i < n; ++i) {
                vad_stream_params sp = vad_stream_default_params();
                sp.slice_samples = lens[i];
                st[i] = vad_stream_new_with_params(m, sp);
                outp[i] = &pb[i];
            }
            CHECK(vad_streams_push(st.data(), n, ptrs.data(), lens.data(), outp.data(), maxp.data(), nout.data()) == 0, "streams_push");
            for (int i = 0; i < n; ++i) { CHECK(nout[i] == 1, "one window per stream"); vad_stream_free(st[i]); }
        }
        float worst_b = 0;
        for (size_t i = 0; i < cases.size(); ++i) worst_b = std::max(worst_b, std::fabs(pb[i] - cases[i].prob));
        CHECK(worst <= tol && worst_b <= tol, "%s: max |p - official| %.3g (batch %.3g) > %.3g", spec.c_str(), worst, worst_b, tol);
        fprintf(stderr, "%-9s %-26s vs-official %.2e  batch %.2e%s\n", spec.c_str(), file.c_str(), worst, worst_b,
                checked ? "" : "  (quantized: not checked)");
        vad_model_free(m);
    }
    // a tensor used outside mul_mat may not be quantized: loading must fail
    if (file_exists(data + "/invalid-smart-turn-pos-q8_0.gguf")) {
        vad_model * bad = vad_model_load((data + "/invalid-smart-turn-pos-q8_0.gguf").c_str(), vad_model_default_params());
        CHECK(bad == nullptr, "loading a model with a quantized non-mul_mat tensor must fail");
        if (bad) vad_model_free(bad);
    }
    return finish("test_turn");
}
