// vad-cli: speech (or per-speaker) segments, or per-frame probabilities, for an audio file.
#include "vad.h"
#include "wav.h"

#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m MODEL -f AUDIO [options]\n"
        "  -m, --model PATH        model file (GGUF or whisper.cpp ggml-silero-*.bin)\n"
        "  -f, --file PATH         WAV (any rate/channels) or raw float32 (*.f32, model rate)\n"
        "  -d, --device NAME       cpu (default), ggml-cpu, or a device from --list-devices\n"
        "  -t, --threads N         CPU threads (default 1)\n"
        "  --threshold P           speech threshold (default 0.5)\n"
        "  --min-speech-ms N       (default 250)\n"
        "  --min-silence-ms N      (default 100)\n"
        "  --speech-pad-ms N       (default 30)\n"
        "  --max-speech-s S        (default: no limit)\n"
        "  --probs                 print per-chunk probabilities instead of segments\n"
        "  --list-devices          list devices and exit\n", argv0);
}

int main(int argc, char ** argv) {
    std::string model_path, audio_path;
    vad_model_params mp = vad_model_default_params();
    vad_segment_params sp = vad_segment_default_params();
    bool print_probs = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char * {
            if (i + 1 >= argc) { usage(argv[0]); exit(1); }
            return argv[++i];
        };
        if (a == "-m" || a == "--model") model_path = next();
        else if (a == "-f" || a == "--file") audio_path = next();
        else if (a == "-d" || a == "--device") mp.device = next();
        else if (a == "-t" || a == "--threads") mp.n_threads = atoi(next());
        else if (a == "--threshold") sp.threshold = (float) atof(next());
        else if (a == "--min-speech-ms") sp.min_speech_ms = atoi(next());
        else if (a == "--min-silence-ms") sp.min_silence_ms = atoi(next());
        else if (a == "--speech-pad-ms") sp.speech_pad_ms = atoi(next());
        else if (a == "--max-speech-s") sp.max_speech_s = (float) atof(next());
        else if (a == "--probs") print_probs = true;
        else if (a == "--list-devices") {
            for (int d = 0; d < vad_device_count(); ++d) printf("%-12s %s\n", vad_device_name(d), vad_device_description(d));
            return 0;
        } else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || audio_path.empty()) { usage(argv[0]); return 1; }

    vad_model * model = vad_model_load(model_path.c_str(), mp);
    if (!model) return 1;
    const int sr = vad_model_sample_rate(model), cs = vad_model_hop_samples(model);

    std::vector<float> audio;
    std::string err;
    if (!read_audio(audio_path, sr, audio, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    if (vad_model_output(model) == VAD_OUTPUT_TURN_END && !print_probs) {
        // turn-end probability at the end of the file: a stream without slices, push, flush
        vad_stream_params sp = vad_stream_default_params();
        sp.slice_samples = INT32_MAX;
        float p = 0;
        const auto t0 = std::chrono::steady_clock::now();
        if (vad_detect_with_params(model, sp, audio.data(), (int) audio.size(), &p, 1) != 1) { fprintf(stderr, "error: detection failed\n"); return 1; }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        fprintf(stderr, "%s on %s: %.1f s of audio in %.2f ms\n", vad_model_arch(model), vad_model_device(model), audio.size() / (double) sr, ms);
        printf("turn end probability: %.4f (%s)\n", p, p > 0.5f ? "complete" : "incomplete");
        vad_model_free(model);
        return 0;
    }
    const int no = vad_model_n_outputs(model);
    std::vector<float> probs((audio.size() / cs + 1) * no);
    const auto t0 = std::chrono::steady_clock::now();
    const int n = vad_detect(model, audio.data(), (int) audio.size(), probs.data(), (int) (probs.size() / no));
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (n < 0) { fprintf(stderr, "error: detection failed\n"); return 1; }
    fprintf(stderr, "%s on %s: %.1f s of audio, %d chunks in %.2f ms (%.1f us/chunk)\n",
            vad_model_arch(model), vad_model_device(model), audio.size() / (double) sr, n, ms, ms * 1e3 / n);

    if (print_probs) {
        // probability i refers to the audio ending at warmup + (i + 1) * hop (the last one may be a flush at the end)
        const int wu = vad_model_warmup_samples(model);
        for (int i = 0; i < n; ++i) {
            printf("%.3f", std::min<double>(audio.size(), wu + (double) (i + 1) * cs) / sr);
            for (int k = 0; k < no; ++k) printf("\t%.6f", probs[(size_t) i * no + k]);
            printf("\n");
        }
    } else {
        const int n_seg = vad_segments_from_probs(probs.data(), n, no, cs, sr, (int64_t) audio.size(), sp, nullptr, 0);
        std::vector<vad_segment> seg(n_seg);
        vad_segments_from_probs(probs.data(), n, no, cs, sr, (int64_t) audio.size(), sp, seg.data(), n_seg);
        const bool spk = vad_model_output(model) == VAD_OUTPUT_SPEAKERS;
        for (auto & s : seg) {
            if (spk) printf("%.3f\t%.3f\tspeaker_%d\n", s.start / (double) sr, s.end / (double) sr, s.channel);
            else printf("%.3f\t%.3f\n", s.start / (double) sr, s.end / (double) sr);
        }
    }
    vad_model_free(model);
    return 0;
}
