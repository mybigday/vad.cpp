// Debug: run one model graph on a device and on the CPU (without the scheduler, so
// ops the device rejects still run on it), print the first nodes whose outputs differ. usage: compare_backends MODEL DEVICE [B] [T]
// Across builds (e.g. native vs WASM): VAD_COMPARE_DUMP=file writes every node output of the
// DEVICE run, VAD_COMPARE_REF=file compares the DEVICE run against such a dump instead of the CPU.
#include "../../src/vad-impl.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace vad;

struct cb_state { int shown = 0; FILE * dump = nullptr; FILE * ref = nullptr; double tol = 1e-3; };

static bool cb(int i, ggml_tensor * t1, ggml_tensor * t2, void * ud) {
    auto * st = (cb_state *) ud;
    if (t1->type != GGML_TYPE_F32 || ggml_is_empty(t1)) return true;
    const int64_t n = ggml_nelements(t1);
    std::vector<float> a(n), b(n);
    ggml_backend_tensor_get(t1, a.data(), 0, ggml_nbytes(t1));   // contiguous copies made by the compare helper
    if (st->dump) {
        fwrite(&n, sizeof n, 1, st->dump);
        fwrite(a.data(), sizeof(float), n, st->dump);
        return true;
    }
    if (st->ref) {
        int64_t rn = 0;
        if (fread(&rn, sizeof rn, 1, st->ref) != 1 || rn != n) { printf("node %d: reference dump does not match the graph\n", i); return false; }
        if (fread(b.data(), sizeof(float), n, st->ref) != (size_t) n) return false;
    } else {
        ggml_backend_tensor_get(t2, b.data(), 0, ggml_nbytes(t2));
    }
    double md = 0, mx = 0;
    for (int64_t k = 0; k < n; ++k) { md = std::max(md, (double) std::fabs(a[k] - b[k])); mx = std::max(mx, (double) std::fabs(b[k])); }
    if (md > st->tol * (1 + mx) && st->shown < 15) {
        printf("node %4d %-10s %-14s [%lld,%lld,%lld] max|diff| %.3g (max|x| %.3g)\n", i, ggml_op_desc(t1), t1->name,
               (long long) t1->ne[0], (long long) t1->ne[1], (long long) t1->ne[2], md, mx);
        st->shown++;
    }
    return true;
}

#ifdef VAD_COMPARE_NO_MAIN
extern "C" int vad_compare_backends(int argc, char ** argv) {
#else
int main(int argc, char ** argv) {
#endif
    if (argc < 3) { fprintf(stderr, "usage: %s MODEL DEVICE [B] [T]\n", argv[0]); return 1; }
    const int B = argc > 3 ? atoi(argv[3]) : 4, T = argc > 4 ? atoi(argv[4]) : 1;
    model_file mf; std::string err;
    if (!load_gguf(argv[1], mf, err)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    auto a = mf.arch == "firered" ? make_firered(mf, err) : mf.arch == "smart_turn" ? make_smart_turn(mf, err) : make_silero(mf, err);
    if (!a) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name(argv[2]);
    if (!dev) { fprintf(stderr, "no device %s\n", argv[2]); return 1; }
    ggml_backend_t be = ggml_backend_dev_init(dev, nullptr), cpu = ggml_backend_cpu_init();

    weight_uploads up;
    ggml_init_params wp = { ggml_tensor_overhead() * 512, nullptr, true };
    ggml_context * wctx = ggml_init(wp);
    weight_builder wb(wctx, up, mf, err);
    if (!a->create_weights(wb)) { fprintf(stderr, "%s\n", err.c_str()); return 1; }
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, be);
    for (auto & [t, d] : up.items) ggml_backend_tensor_set(t, d.data(), 0, d.size());

    const size_t nmax = a->max_nodes(B, T);
    std::vector<uint8_t> meta(ggml_tensor_overhead() * nmax * 2 + ggml_graph_overhead_custom(nmax, false));
    ggml_init_params gp = { meta.size(), meta.data(), true };
    ggml_context * gctx = ggml_init(gp);
    ggml_cgraph * gf = ggml_new_graph_custom(gctx, nmax, false);
    graph_opts o; o.host = false;
    ggml_init_params sp = { ggml_tensor_overhead() * 2, nullptr, true };
    ggml_context * sctx = ggml_init(sp);
    ggml_tensor * state = ggml_new_tensor_2d(sctx, GGML_TYPE_F32, std::max<int64_t>(1, a->state_size()), B);   // stateless models: a dummy
    ggml_backend_alloc_ctx_tensors(sctx, be);
    graph_io io = a->build_graph(gctx, gf, B, T, o, state);
    if (io.frame_mask) {
        std::vector<float> mk(ggml_nelements(io.frame_mask), 1.0f);
        ggml_backend_tensor_set(io.frame_mask, mk.data(), 0, ggml_nbytes(io.frame_mask));
    }
    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    ggml_gallocr_alloc_graph(ga, gf);
    // same inputs on every standard library (distributions are implementation defined)
    std::mt19937 rng(3);
    auto nd = [&] { return ((float) (rng() >> 8) / 16777216.0f - 0.5f) * 0.35f; };
    std::vector<float> x(ggml_nelements(io.audio)), s(ggml_nelements(state));
    for (auto & v : x) v = nd();
    for (auto & v : s) v = nd();
    ggml_backend_tensor_set(io.audio, x.data(), 0, ggml_nbytes(io.audio));
    ggml_backend_tensor_set(state, s.data(), 0, ggml_nbytes(state));
    cb_state st;
    if (const char * f = getenv("VAD_COMPARE_DUMP")) st.dump = fopen(f, "wb");
    if (const char * f = getenv("VAD_COMPARE_REF"))  st.ref  = fopen(f, "rb");
    if (const char * f = getenv("VAD_COMPARE_TOL"))  st.tol  = atof(f);   // relative threshold, default 1e-3
    ggml_backend_compare_graph_backend(be, cpu, gf, cb, &st, nullptr, 0);
    if (st.dump) fclose(st.dump);
    if (st.ref) fclose(st.ref);
    printf("%d differing nodes shown\n", st.shown);
    return 0;
}
