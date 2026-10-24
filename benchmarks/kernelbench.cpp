// NNUE inference microbenchmark on the real network and real positions.
//
// Compiles the engine sources (minus main.cpp) into one translation unit and times
// eval_simd() plus the AVX2 layer kernels in isolation, together with a rough
// core-clock probe. It uses engine internals (simd.h kernels, NNUE members), so
// adjust it when those change. Build with: build_unity.ps1 -Out out\kernelbench.exe -Kernelbench
// Run: out\kernelbench.exe [path-to-weights]   (default: the engine's configured net)

#include <chrono>
#include <cstdio>

#ifndef ENGINE_SRC
#error "define ENGINE_SRC as the repository root"
#endif
#define TB_STR2(x) #x
#define TB_STR(x) TB_STR2(x)
#include TB_STR(ENGINE_SRC/src/board.cpp)
#include TB_STR(ENGINE_SRC/src/book.cpp)
#include TB_STR(ENGINE_SRC/src/engine.cpp)
#include TB_STR(ENGINE_SRC/src/gamestate.cpp)
#include TB_STR(ENGINE_SRC/src/helpers.cpp)
#include TB_STR(ENGINE_SRC/src/magics.cpp)
#include TB_STR(ENGINE_SRC/src/movegen.cpp)
#include TB_STR(ENGINE_SRC/src/nnue.cpp)
#include TB_STR(ENGINE_SRC/src/precomp.cpp)
#include TB_STR(ENGINE_SRC/src/searcher.cpp)
#include TB_STR(ENGINE_SRC/src/search_threads.cpp)
#include TB_STR(ENGINE_SRC/src/uci.cpp)
#include TB_STR(ENGINE_SRC/src/affinity.cpp)

static NNUE g_net;

// best of 5 timed batches after a warm-up (lets the core clock ramp up)
template <typename F>
static double time_ns(F&& f, int n) {
    for (int i = 0; i < n / 4; ++i) f();
    double best = 1e30;
    for (int rep = 0; rep < 5; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; ++i) f();
        auto t1 = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double, std::nano>(t1 - t0).count() / n);
    }
    return best;
}

// n dependent 3-cycle integer multiplies: ~3n cycles, so it estimates the core clock
static uint64_t dependent_multiplies(int n) {
    uint64_t x = 1, y = 3;
    for (int i = 0; i < n; ++i) asm volatile("imul %1, %0" : "+r"(x) : "r"(y));
    return x;
}

int main(int argc, char** argv) {
    PrecomputedMoveData::init();
#ifdef _WIN32
    Magics::initPEXTMagics();
#else
    Magics::initMagics();
#endif
    const std::string weights = argc > 1 ? argv[1] : EngineOptions{}.nnue_weight_path.string();
    if (!g_net.load(weights)) return 1;

    const char* fens[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bq1rk1/pp2b1pp/n1pp1n2/3P1p2/2P1p3/2N1P2N/PP2BPPP/R1BQ1RK1 b - - 2 10",
        "8/2p5/8/2kPKp1p/2p4P/2P5/3P4/8 w - - 0 1",
        "4rrk1/2p1b1p1/p1p3q1/4p3/2P2n1p/1P1NR2P/PB3PP1/3R1QK1 b - - 2 24",
    };
    volatile int sink = 0;
    for (const char* fen : fens) {
        Board b(fen);
        g_net.reset_stack(b);
        const U64 occ = b.colorBitboards[0] | b.colorBitboards[1];
        const double eval_ns = time_ns([&] { sink = sink + g_net.eval_simd(b.is_white_move, occ); }, 400000);
        const double scalar_ns = time_ns([&] { sink = sink + g_net.evaluate(b.is_white_move, occ); }, 20000);
        std::printf("%-26.26s eval_simd %7.1f ns  scalar %8.1f ns  (eval %d = %d)\n", fen, eval_ns, scalar_ns,
                    g_net.eval_simd(b.is_white_move, occ), g_net.evaluate(b.is_white_move, occ));
#if defined(__AVX2__)
        const int bucket = output_bucket(occ);
        const Accumulator& us = b.is_white_move ? g_net.acc_stm() : g_net.acc_ntm();
        const Accumulator& them = b.is_white_move ? g_net.acc_ntm() : g_net.acc_stm();
        alignas(32) int32_t sums[16];
        const double l1_ns = time_ns([&] {
            l1_sparse_avx2(us.vals, them.vals, g_net.l1w_pairs[bucket], sums);
            sink = sink + sums[3];
        }, 400000);
        alignas(32) int32_t x[16];
        for (int i = 0; i < 16; ++i) x[i] = (i * 977) % 8128;
        alignas(32) int64_t out[32];
        const double l2_ns = time_ns([&] { l2_avx2(x, g_net.l2w_by_input[bucket], out); sink = sink + (int)out[5]; }, 400000);
        std::printf("%-26s   L1 %7.1f ns  L2 %6.1f ns\n", "", l1_ns, l2_ns);
#endif
    }
    const double chain_ns = time_ns([&] { sink = sink + (int)dependent_multiplies(1000); }, 20000);
    std::printf("core clock estimate: ~%.2f GHz (3000 cycles in %.0f ns)\n", 3000.0 / chain_ns, chain_ns);
    return 0;
}
