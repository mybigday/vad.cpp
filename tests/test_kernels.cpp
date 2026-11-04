// Every kernel variant this CPU supports vs a double-precision reference.
#include "test_common.h"
#include "kernels.h"
#include "ggml.h"

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

int main() {
    const vadk::table * vs[8];
    const int nv = vadk::available(vs, 8);
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    for (int v = 0; v < nv; ++v) {
        const vadk::table & k = *vs[v];
        // many-row GEMM (F32 / F16 / Q8 weights) vs the per-row kernels: row tails of the
        // register tile, column tails, K across several cache blocks, strided X/Y.
        // Must be bit-identical: streams run the narrow kernels, offline runs this one.
        if (k.gemm_big) {
            for (int M : {16, 17, 29, 64}) for (int N : {31, 64, 100, 513}) for (int K : {64, 300, 544}) {
                const int ldx = K + 3, ldy = N + 5, ldw = (N + 31) / 32 * 32;
                std::vector<float> X((size_t) M * ldx), W((size_t) K * ldw, 0.0f), bias(N), Y((size_t) M * ldy, 0.0f), Yr = Y;
                std::vector<uint16_t> Wh((size_t) K * ldw, 0);
                std::vector<int8_t> Q((size_t) K * ldw, 0);
                std::vector<float> S((size_t) (K / 32) * ldw, 0.0f);
                for (auto & x : X) x = nd(rng);
                for (int kk = 0; kk < K; ++kk) for (int n = 0; n < N; ++n) {
                    const size_t i = (size_t) kk * ldw + n;
                    W[i] = nd(rng);
                    Wh[i] = ggml_fp32_to_fp16(W[i]);
                    Q[i] = (int8_t) ((int) (nd(rng) * 30) % 127);
                }
                for (auto & v : S) v = 0.01f + std::fabs(nd(rng)) * 0.01f;
                for (auto & b : bias) b = nd(rng);
                for (int type = 0; type < 3; ++type) for (bool relu : {false, true}) {
                    if (type == 2 && K % 32) continue;
                    vadk::weights w;
                    w.ldw = ldw;
                    if (type == 0) { w.f32 = W.data(); k.gemm(X.data(), ldx, M, W.data(), ldw, K, N, bias.data(), relu, Yr.data(), ldy); }
                    if (type == 1) { w.f16 = Wh.data(); k.gemm_f16(X.data(), ldx, M, Wh.data(), ldw, K, N, bias.data(), relu, Yr.data(), ldy); }
                    if (type == 2) { w.q8 = Q.data(); w.qs = S.data(); k.gemm_q8(X.data(), ldx, M, Q.data(), S.data(), ldw, K, N, bias.data(), relu, Yr.data(), ldy); }
                    k.gemm_big(X.data(), ldx, M, w, K, N, bias.data(), relu, Y.data(), ldy);
                    float worst = 0;
                    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n)
                        worst = std::max(worst, std::fabs(Y[(size_t) m * ldy + n] - Yr[(size_t) m * ldy + n]));
                    CHECK(worst == 0.0f, "%s gemm_big type %d M=%d N=%d K=%d relu=%d differs by %.3g", k.name, type, M, N, K, relu, worst);
                }
            }
        }
        // GEMM over awkward shapes (M blocks 4/2/1, N tails, strided X/Y)
        for (int M : {1, 2, 3, 4, 5, 9}) for (int N : {1, 31, 64, 129, 258, 512}) for (int K : {3, 64, 387}) {
            const int ldx = K + 3, ldy = N + 5, ldw = (N + 31) / 32 * 32;
            std::vector<float> X((size_t) M * ldx), W((size_t) K * ldw, 0.0f), bias(N), Y((size_t) M * ldy, 0.0f);
            for (auto & x : X) x = nd(rng);
            for (int kk = 0; kk < K; ++kk) for (int n = 0; n < N; ++n) W[(size_t) kk * ldw + n] = nd(rng);
            for (auto & b : bias) b = nd(rng);
            for (bool relu : {false, true}) {
                k.gemm(X.data(), ldx, M, W.data(), ldw, K, N, bias.data(), relu, Y.data(), ldy);
                double worst = 0;
                for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
                    double r = bias[n], mag = std::fabs(bias[n]);
                    for (int kk = 0; kk < K; ++kk) {
                        const double pr = (double) X[(size_t) m * ldx + kk] * W[(size_t) kk * ldw + n];
                        r += pr; mag += std::fabs(pr);
                    }
                    if (relu && r < 0) r = 0;
                    // float32 summation error scales with the sum of |terms|
                    worst = std::max(worst, std::fabs(r - Y[(size_t) m * ldy + n]) / (1e-30 + mag * 1.2e-7 * std::sqrt((double) K)));
                }
                CHECK(worst < 1.0, "%s gemm M=%d N=%d K=%d relu=%d err %.3g x bound", k.name, M, N, K, relu, worst);
            }
        }
        // one row (own code path for big weights on some targets) == that row inside a block of rows
        for (int N : {64, 513}) for (int K : {387, 1024}) {
            const int ldw = (N + 31) / 32 * 32;
            std::vector<float> X((size_t) 5 * K), W((size_t) K * ldw, 0.0f), bias(N), Y1(N), Y5((size_t) 5 * N);
            std::vector<uint16_t> Wh((size_t) K * ldw, 0);
            std::vector<int8_t> Q((size_t) K * ldw, 0);
            std::vector<float> S((size_t) (K / 32) * ldw, 0.01f);
            for (auto & x : X) x = nd(rng);
            for (auto & b : bias) b = nd(rng);
            for (int kk = 0; kk < K; ++kk) for (int n = 0; n < N; ++n) {
                W[(size_t) kk * ldw + n] = nd(rng);
                Wh[(size_t) kk * ldw + n] = ggml_fp32_to_fp16(W[(size_t) kk * ldw + n]);
                Q[(size_t) kk * ldw + n] = (int8_t) ((int) (nd(rng) * 30) % 127);
            }
            for (int type = 0; type < 3; ++type) {
                if (type == 2 && K % 32) continue;
                auto run = [&](int M, float * Y) {
                    if (type == 0) k.gemm(X.data(), K, M, W.data(), ldw, K, N, bias.data(), true, Y, N);
                    if (type == 1) k.gemm_f16(X.data(), K, M, Wh.data(), ldw, K, N, bias.data(), true, Y, N);
                    if (type == 2) k.gemm_q8(X.data(), K, M, Q.data(), S.data(), ldw, K, N, bias.data(), true, Y, N);
                };
                run(1, Y1.data());
                run(5, Y5.data());
                float worst = 0;
                for (int n = 0; n < N; ++n) worst = std::max(worst, std::fabs(Y1[n] - Y5[n]));
                CHECK(worst == 0.0f, "%s type %d one row vs block N=%d K=%d differs by %.3g", k.name, type, N, K, worst);
            }
        }
        // F16 weights: same as the F32 GEMM on the half-rounded weights
        for (int M : {1, 3, 4, 9}) for (int N : {31, 129, 512}) for (int K : {64, 387}) {
            const int ldx = K, ldy = N, ldw = (N + 31) / 32 * 32;
            std::vector<float> X((size_t) M * ldx), Wf((size_t) K * ldw, 0.0f), Y((size_t) M * ldy), Yr((size_t) M * ldy);
            std::vector<uint16_t> Wh((size_t) K * ldw, 0);
            for (auto & x : X) x = nd(rng);
            for (int kk = 0; kk < K; ++kk) for (int n = 0; n < N; ++n) {
                Wh[(size_t) kk * ldw + n] = ggml_fp32_to_fp16(nd(rng));
                Wf[(size_t) kk * ldw + n] = ggml_fp16_to_fp32(Wh[(size_t) kk * ldw + n]);
            }
            k.gemm_f16(X.data(), ldx, M, Wh.data(), ldw, K, N, nullptr, false, Y.data(), ldy);
            k.gemm(X.data(), ldx, M, Wf.data(), ldw, K, N, nullptr, false, Yr.data(), ldy);
            float worst = 0;
            for (size_t i = 0; i < Y.size(); ++i) worst = std::max(worst, std::fabs(Y[i] - Yr[i]));
            CHECK(worst == 0.0f, "%s gemm_f16 M=%d N=%d K=%d differs from F32 on rounded weights by %.3g", k.name, M, N, K, worst);
        }
        // int8 weights with per-(32-block, column) scales == F32 GEMM on the dequantized weights
        for (int M : {1, 4, 7}) for (int N : {31, 256}) for (int K : {64, 384}) {
            const int ldw = (N + 31) / 32 * 32;
            std::vector<float> X((size_t) M * K), Wf((size_t) K * ldw, 0.0f), S((size_t) (K / 32) * ldw, 0.0f), Y((size_t) M * N), Yr((size_t) M * N);
            std::vector<int8_t> Q((size_t) K * ldw, 0);
            for (auto & x : X) x = nd(rng);
            for (int b = 0; b < K / 32; ++b) for (int n = 0; n < N; ++n) S[(size_t) b * ldw + n] = 0.01f * (1 + (n + b) % 7);
            for (int kk = 0; kk < K; ++kk) for (int n = 0; n < N; ++n) {
                Q[(size_t) kk * ldw + n] = (int8_t) ((int) (nd(rng) * 40) % 127);
                Wf[(size_t) kk * ldw + n] = Q[(size_t) kk * ldw + n] * S[(size_t) (kk / 32) * ldw + n];
            }
            k.gemm_q8(X.data(), K, M, Q.data(), S.data(), ldw, K, N, nullptr, false, Y.data(), N);
            k.gemm(X.data(), K, M, Wf.data(), ldw, K, N, nullptr, false, Yr.data(), N);
            float worst = 0;
            for (size_t i = 0; i < Y.size(); ++i) worst = std::max(worst, std::fabs(Y[i] - Yr[i]));
            float mag = 0;
            for (float v : Yr) mag = std::max(mag, std::fabs(v));
            CHECK(worst <= 1e-5f * (1 + mag), "%s gemm_q8 M=%d N=%d K=%d differs by %.3g", k.name, M, N, K, worst);
        }
        // opt-in reduced precision: close to the exact kernels; overflowing fp16 blocks fall back
        if (k.gemm_f16_lp) {
            for (int M : {1, 2, 3, 4, 7, 33}) for (int N : {31, 64, 100, 513}) for (int K : {64, 387, 544}) {
                const int ldx = K + 3, ldy = N + 5, ldw = (N + 31) / 32 * 32;
                std::vector<float> X((size_t) M * ldx), Y((size_t) M * ldy), Yr((size_t) M * ldy), S((size_t) (K / 32 + 1) * ldw), bias(N);
                std::vector<uint16_t> Wh((size_t) K * ldw, 0);
                std::vector<int8_t> Q((size_t) K * ldw, 0), Q4((size_t) K * ldw);
                for (auto & x : X) x = nd(rng);
                if (M == 33) X[5] = 3e5f;   // overflows fp16: that block is recomputed exactly
                for (auto & b : bias) b = nd(rng);
                for (auto & v : S) v = ggml_fp16_to_fp32(ggml_fp32_to_fp16(0.01f + std::fabs(nd(rng)) * 0.01f));   // file scales are fp16
                for (int kk = 0; kk < K; ++kk) for (int n = 0; n < N; ++n) {
                    Wh[(size_t) kk * ldw + n] = ggml_fp32_to_fp16(nd(rng) * 0.1f);
                    Q[(size_t) kk * ldw + n] = (int8_t) ((int) (nd(rng) * 30) % 127);
                }
                auto err = [&](const char * what) {
                    double worst = 0, mag = 0;
                    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) mag = std::max(mag, (double) std::fabs(Yr[(size_t) m * ldy + n]));
                    for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n)
                        worst = std::max(worst, (double) std::fabs(Y[(size_t) m * ldy + n] - Yr[(size_t) m * ldy + n]) / (1 + mag));
                    CHECK(worst < 1e-2, "%s %s M=%d N=%d K=%d rel err %.3g", k.name, what, M, N, K, worst);
                };
                std::vector<uint16_t> Wr((size_t) N * (K + 2));   // file layout [N][K], padded rows
                for (int kk = 0; kk < K; ++kk) for (int n = 0; n < N; ++n) Wr[(size_t) n * (K + 2) + kk] = Wh[(size_t) kk * ldw + n];
                k.gemm_f16(X.data(), ldx, M, Wh.data(), ldw, K, N, bias.data(), false, Yr.data(), ldy);
                k.gemm_f16_lp(X.data(), ldx, M, Wr.data(), K + 2, Wh.data(), ldw, K, N, bias.data(), false, Y.data(), ldy);
                err("gemm_f16_lp");
                if (K % 32) continue;
                vadk::pack_q8_interleaved(Q.data(), K, ldw, Q4.data());
                const int64_t ldr = (K / 32) * 34 + 6;   // file layout: block_q8_0 rows {fp16 d, int8 qs[32]}, padded
                std::vector<uint8_t> Qr((size_t) N * ldr, 0);
                for (int n = 0; n < N; ++n) for (int b = 0; b < K / 32; ++b) {
                    uint8_t * blk = Qr.data() + (size_t) n * ldr + 34 * b;
                    const uint16_t d = ggml_fp32_to_fp16(S[(size_t) b * ldw + n]);
                    std::memcpy(blk, &d, 2);
                    for (int i = 0; i < 32; ++i) blk[2 + i] = (uint8_t) Q[(size_t) (32 * b + i) * ldw + n];
                }
                k.gemm_q8(X.data(), ldx, M, Q.data(), S.data(), ldw, K, N, bias.data(), true, Yr.data(), ldy);
                k.gemm_q8_lp(X.data(), ldx, M, Qr.data(), ldr, Q4.data(), S.data(), ldw, K, N, bias.data(), true, Y.data(), ldy);
                err("gemm_q8_lp");
            }
        }
        // unary
        std::vector<float> x(1003), y(1003);
        for (size_t i = 0; i < x.size(); ++i) x[i] = -30.0f + 60.0f * i / x.size();
        struct { vadk::UOp op; double (*ref)(double); const char * name; } us[] = {
            { vadk::UOp::SIGMOID, [](double t) { return 1 / (1 + std::exp(-t)); }, "sigmoid" },
            { vadk::UOp::TANH, [](double t) { return std::tanh(t); }, "tanh" },
            { vadk::UOp::RELU, [](double t) { return t > 0 ? t : 0; }, "relu" },
        };
        for (auto & u : us) {
            k.unary(u.op, x.data(), y.data(), (int) x.size());
            double worst = 0;
            for (size_t i = 0; i < x.size(); ++i) worst = std::max(worst, std::fabs(u.ref(x[i]) - y[i]));
            CHECK(worst < 2e-7, "%s %s abs err %.3g", k.name, u.name, worst);
        }
        // exp / log / gelu: relative error over a wide range, plus log's special values
        {
            std::vector<float> xs, ys;
            for (int i = -1990; i <= 2000; ++i) xs.push_back(i * 0.0437f);   // exp clamps below -87
            ys.resize(xs.size());
            k.unary(vadk::UOp::EXP, xs.data(), ys.data(), (int) xs.size());
            double worst = 0;
            for (size_t i = 0; i < xs.size(); ++i) worst = std::max(worst, std::fabs(ys[i] / std::exp((double) xs[i]) - 1));
            CHECK(worst < 4e-7, "%s exp rel err %.3g", k.name, worst);
            k.unary(vadk::UOp::GELU_ERF, xs.data(), ys.data(), (int) xs.size());
            worst = 0;
            for (size_t i = 0; i < xs.size(); ++i) {
                const double t = xs[i];
                worst = std::max(worst, std::fabs(ys[i] - 0.5 * t * (1 + std::erf(t / std::sqrt(2.0)))) / (1 + std::fabs(t)));
            }
            CHECK(worst < 4e-7, "%s gelu_erf err %.3g", k.name, worst);
            std::vector<float> ls;
            for (float v = 1e-44f; v < 3e38f; v *= 1.37f) ls.push_back(v);   // denormals .. near FLT_MAX
            ls.push_back(1.0f); ls.push_back(0.0f); ls.push_back(-1.0f); ls.push_back(INFINITY);
            std::vector<float> lo(ls.size());
            k.unary(vadk::UOp::LOG, ls.data(), lo.data(), (int) ls.size());
            worst = 0;
            for (size_t i = 0; i + 4 < ls.size(); ++i) worst = std::max(worst, std::fabs(lo[i] - std::log((double) ls[i])) / std::max(1.0, std::fabs(std::log((double) ls[i]))));
            CHECK(worst < 3e-7, "%s log rel err %.3g", k.name, worst);
            const size_t e = ls.size();
            CHECK(lo[e - 4] == 0.0f && lo[e - 3] == -INFINITY && std::isnan(lo[e - 2]) && lo[e - 1] == INFINITY,
                  "%s log special values %g %g %g %g", k.name, lo[e - 4], lo[e - 3], lo[e - 2], lo[e - 1]);
        }
        // binary with broadcast
        std::vector<float> a(77), b(77), out(77);
        for (auto & t : a) t = nd(rng);
        for (auto & t : b) t = nd(rng);
        k.binary(vadk::BOp::MUL, a.data(), b.data(), 77, out.data(), 77);
        for (int i = 0; i < 77; ++i) CHECK(out[i] == a[i] * b[i], "%s mul", k.name);
        k.binary(vadk::BOp::ADD, a.data(), b.data(), 1, out.data(), 77);
        for (int i = 0; i < 77; ++i) CHECK(out[i] == a[i] + b[0], "%s add broadcast", k.name);
        fprintf(stderr, "variant %s checked\n", k.name);
    }
    return finish("test_kernels");
}
