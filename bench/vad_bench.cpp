// Benchmarks through the public API; every model is measured the same way.
// usage: vad-bench -m MODEL -f AUDIO [-d DEVICE[,DEVICE...]] [-r REPS] [-s STREAMS,...] [-t THREADS] [--only-streams 1] [--windows 0]
//   stream  : vad_stream_push of one hop at a time (latency path)
//   offline : vad_detect over the whole file
//   window  : lookahead models only, sliding-window streams (slice x lookahead)
//   end     : the probability for the end of the file only (stream without slices + flush)
//   S=n     : vad_streams_push over n concurrent streams, one hop each
// Times are microseconds per output probability (per stream for S=n).
#include "vad.h"
#include "wav.h"
#ifdef VAD_WITH_CPU_KERNELS
#include "vad-cpu.h"
#endif

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

static double now_us() { return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static double median_us(int reps, const std::function<void()> & fn) {
    fn();
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) { double t0 = now_us(); fn(); t.push_back(now_us() - t0); }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

static std::vector<std::string> split(const std::string & s) {
    std::vector<std::string> v;
    std::stringstream ss(s);
    for (std::string x; std::getline(ss, x, ',');) if (!x.empty()) v.push_back(x);
    return v;
}

// pushes the whole buffer in pieces of `piece` samples, then flushes; returns the output frame count
static int run_stream(vad_stream * st, const std::vector<float> & audio, int piece, int n_outputs) {
    std::vector<float> pr;
    int n = 0;
    for (size_t at = 0; at < audio.size(); at += piece) {
        const int len = (int) std::min<size_t>(piece, audio.size() - at);
        const int k = vad_stream_n_pending(st, len);
        pr.resize((size_t) std::max(1, k) * n_outputs);
        n += vad_stream_push(st, audio.data() + at, len, pr.data(), k);
    }
    const int k = vad_stream_n_flush(st);
    pr.resize((size_t) std::max(1, k) * n_outputs);
    return n + vad_stream_flush(st, pr.data(), k);
}

int main(int argc, char ** argv) {
    std::string model_path, audio_path, devs = "cpu,ggml-cpu", streams = "16,256";
    int reps = 5, threads = 1;
    bool only_streams = false, windows = true;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "-m") model_path = argv[i + 1];
        else if (a == "-f") audio_path = argv[i + 1];
        else if (a == "-d") devs = argv[i + 1];
        else if (a == "-r") reps = atoi(argv[i + 1]);
        else if (a == "-s") streams = argv[i + 1];
        else if (a == "-t") threads = atoi(argv[i + 1]);
        else if (a == "--only-streams") only_streams = atoi(argv[i + 1]) != 0;
        else if (a == "--windows") windows = atoi(argv[i + 1]) != 0;
    }
    setvbuf(stdout, nullptr, _IOLBF, 0);
    if (model_path.empty() || audio_path.empty()) {
        fprintf(stderr, "usage: %s -m MODEL -f AUDIO [-d DEV,...] [-r REPS] [-s S,...] [-t THREADS] [--only-streams 1]\n", argv[0]);
        return 1;
    }
#ifdef VAD_WITH_CPU_KERNELS
    printf("vad-cpu kernels: %s\n", ggml_backend_vad_cpu_kernels());
#endif

    for (const auto & dev : split(devs)) {
        vad_model_params mp = vad_model_default_params();
        mp.device = dev.c_str();
        mp.n_threads = threads;
        vad_model * m = vad_model_load(model_path.c_str(), mp);
        if (!m) continue;
        const int sr = vad_model_sample_rate(m), hop = vad_model_hop_samples(m), wu = vad_model_warmup_samples(m);
        const int max_la = vad_model_max_lookahead_samples(m), no = vad_model_n_outputs(m);
        std::vector<float> audio;
        std::string err;
        if (!read_audio(audio_path, sr, audio, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
        const int n_hops = (int) ((audio.size() - std::min<size_t>(audio.size(), wu)) / hop);
        const vad_output kind = vad_model_output(m);
        printf("\n[%s] %s (%s), %.1f s audio, hop %d, lookahead %d (us per output frame)\n", dev.c_str(), vad_model_arch(m),
               kind == VAD_OUTPUT_SPEECH ? "speech" : kind == VAD_OUTPUT_TURN_END ? "turn end" : "speakers",
               audio.size() / (double) sr, hop, max_la);

        if (!only_streams) {
            int n_out = 0;
            double t = median_us(reps, [&] {
                vad_stream * st = vad_stream_new(m);
                n_out = run_stream(st, audio, hop, no);
                vad_stream_free(st);
            });
            printf("  %-32s %10.2f\n", "stream (1 hop per push)", t / n_out);

            std::vector<float> probs((audio.size() / hop + 2) * no);
            t = median_us(reps, [&] { n_out = vad_detect(m, audio.data(), (int) audio.size(), probs.data(), (int) (probs.size() / no)); });
            printf("  %-32s %10.2f\n", "offline (vad_detect)", t / n_out);

            if (vad_model_output(m) == VAD_OUTPUT_TURN_END) {
                vad_stream_params sp = vad_stream_default_params();
                sp.slice_samples = INT32_MAX;
                float p;
                t = median_us(reps, [&] { vad_detect_with_params(m, sp, audio.data(), (int) audio.size(), &p, 1); });
                printf("  %-32s %10.2f\n", "end of audio only", t);
            }
            if (max_la > 0 && windows) {
                for (int slice_ms : { 250, 500, 1000 }) for (int la : { -1, 8000 }) {
                    vad_stream_params sp = vad_stream_default_params();
                    sp.slice_samples = slice_ms * sr / 1000;
                    sp.lookahead_samples = la;
                    t = median_us(std::max(1, reps / 2), [&] {
                        vad_stream * st = vad_stream_new_with_params(m, sp);
                        n_out = run_stream(st, audio, hop, no);
                        vad_stream_free(st);
                    });
                    char name[64];
                    snprintf(name, sizeof name, "window slice %dms la %s", slice_ms, la < 0 ? "max" : "500ms");
                    printf("  %-32s %10.2f\n", name, t / n_out);
                }
            }
        }

        // S streams, each advancing by one hop per call (batched where the model allows)
        for (const auto & ss : split(streams)) {
            const int S = atoi(ss.c_str());
            std::vector<vad_stream *> st(S);
            std::vector<const float *> ch(S);
            std::vector<int> len(S, hop), maxp(S, 64), nout(S);
            std::vector<std::vector<float>> pr(S, std::vector<float>((size_t) 64 * no));
            std::vector<float *> prp(S);
            for (int k = 0; k < S; ++k) prp[k] = pr[k].data();
            long total = 0;
            const double t = median_us(std::max(1, reps / 2), [&] {
                total = 0;
                for (auto & x : st) { x = vad_stream_new(m); std::vector<float> d((size_t) 64 * no); vad_stream_push(x, audio.data(), wu, d.data(), 64); }
                for (int i = 0; i < n_hops; ++i) {
                    for (int k = 0; k < S; ++k) ch[k] = audio.data() + wu + (size_t) i * hop;
                    if (vad_streams_push(st.data(), S, ch.data(), len.data(), prp.data(), maxp.data(), nout.data()) != 0) {
                        fprintf(stderr, "vad_streams_push failed\n");
                        exit(1);
                    }
                    for (int k = 0; k < S; ++k) total += nout[k];
                }
                for (auto & x : st) vad_stream_free(x);
            });
            char name[32];
            snprintf(name, sizeof name, "streams S=%d", S);
            printf("  %-32s %10.2f\n", name, total ? t / total : 0.0);
        }
        vad_model_free(m);
    }
    return 0;
}
