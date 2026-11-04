// Small conveniences for JavaScript: the public API takes structs by value,
// which is awkward across the wasm boundary. Everything else is exported as is.
#include "vad.h"

#include <emscripten/emscripten.h>

extern "C" {

// load with a device name ("cpu", "WebGPU", ... from vad_device_name) and thread count
EMSCRIPTEN_KEEPALIVE vad_model * vad_web_model_load(const char * path, const char * device, int n_threads) {
    vad_model_params p = vad_model_default_params();
    p.device = device;
    p.n_threads = n_threads;
    return vad_model_load(path, p);
}

// vad_detect_with_params with explicit slice / lookahead (0 / -1 = defaults)
EMSCRIPTEN_KEEPALIVE int vad_web_detect(vad_model * m, int slice_samples, int lookahead_samples, const float * samples,
                                        int n_samples, float * probs, int max_probs) {
    vad_stream_params p = vad_stream_default_params();
    p.slice_samples = slice_samples;
    p.lookahead_samples = lookahead_samples;
    return vad_detect_with_params(m, p, samples, n_samples, probs, max_probs);
}

}
