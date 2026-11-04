// vad_segments_from_probs vs the official silero-vad get_speech_timestamps_from_probs
// (references from tools/make_reference.py).
#include "test_common.h"
#include "vad.h"
#include "wav.h"

#include <sstream>
#include <vector>

int main() {
    const std::string dir = env_or("VAD_TEST_DATA", VAD_TEST_DATA) + "/ref";
    require_file(dir + "/segments.txt");
    std::ifstream f(dir + "/segments.txt");
    std::string line;
    int n_cases = 0;
    while (std::getline(f, line)) {
        const size_t bar = line.find('|');
        std::istringstream head(line.substr(0, bar)), tail(line.substr(bar + 1));
        std::string file;
        long long n_samples;
        vad_segment_params p = vad_segment_default_params();
        head >> file >> n_samples >> p.threshold >> p.neg_threshold >> p.min_speech_ms >> p.min_silence_ms >> p.speech_pad_ms >> p.max_speech_s;
        std::vector<long long> expect;
        for (long long v; tail >> v;) expect.push_back(v);

        std::vector<float> probs;
        CHECK(read_f32(dir + "/" + file, probs), "read %s", file.c_str());
        const int n = vad_segments_from_probs(probs.data(), (int) probs.size(), 1, 512, 16000, n_samples, p, nullptr, 0);
        std::vector<vad_segment> seg(n);
        vad_segments_from_probs(probs.data(), (int) probs.size(), 1, 512, 16000, n_samples, p, seg.data(), n);
        CHECK(2 * n == (int) expect.size(), "%s: %d segments, expected %zu", file.c_str(), n, expect.size() / 2);
        for (int i = 0; i < n && 2 * i + 1 < (int) expect.size(); ++i)
            CHECK(seg[i].start == expect[2 * i] && seg[i].end == expect[2 * i + 1], "%s seg %d: [%lld, %lld] expected [%lld, %lld]",
                  file.c_str(), i, (long long) seg[i].start, (long long) seg[i].end, expect[2 * i], expect[2 * i + 1]);
        ++n_cases;
    }
    CHECK(n_cases > 0, "no cases");
    return finish("test_segments");
}
