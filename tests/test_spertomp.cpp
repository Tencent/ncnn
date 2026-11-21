// Copyright 2026 SpacemiT
// SPDX-License-Identifier: BSD-3-Clause

#include "cpu.h"
#include "datareader.h"
#include "net.h"
#include "spertomp.h"
#include "option.h"

#include <spert.hpp>
#include <algorithm>
#include <atomic>
#include <stdio.h>
#include <utility>
#include <vector>

// One CTest entry for compiler-generated OpenMP, Clang ABI and Net integration.
// Run: ctest --test-dir <build> -R '^test_spertomp$' --output-on-failure
// No model files or platform-specific CPU affinity are required.
static thread_local bool is_caller = false;

static int check(bool ok, const char* test, const char* message)
{
    if (!ok) fprintf(stderr, "test_spertomp %s failed: %s\n", test, message);
    return ok ? 0 : 1;
}

static int check_streams(size_t expected, const char* test)
{
    const size_t actual = spert::runtime_stats().active_streams;
    if (actual != expected)
        fprintf(stderr, "test_spertomp %s failed: active_streams=%zu expected=%zu\n", test, actual, expected);
    return actual == expected ? 0 : 1;
}

static bool on_backend()
{
    if (is_caller) return false;
#if NCNN_RVV
    return (size_t)ncnn::cpu_riscv_vlenb() == spert::backend_info().vlen;
#else
    return true;
#endif
}

struct Participant
{
    Participant()
        : calls(0), team(0), value(0), backend(false)
    {
    }
    int calls;
    int team;
    int value;
    bool backend;
};

static int check_participants(const std::vector<Participant>& results, int expected, int offset, const char* test)
{
    for (size_t i = 0; i < results.size(); i++)
    {
        const Participant& r = results[i];
        if ((int)i < expected ? (r.calls != 1 || r.team != expected || r.value != (int)i + offset || !r.backend) : r.calls != 0)
        {
            fprintf(stderr, "test_spertomp %s failed: tid=%zu calls=%d team=%d expected=%d value=%d backend=%d\n", test, i, r.calls, r.team, expected, r.value, r.backend);
            return 1;
        }
    }
    return 0;
}

// Check both compilers' generated ABI, including the one-thread launch.
static int test_parallel(int requested)
{
    const int expected = std::min(requested, omp_get_max_threads());
    const size_t streams = spert::runtime_stats().active_streams;
    std::vector<Participant> results(requested);
    std::atomic<int> invalid(0);
    #pragma omp parallel num_threads(requested)
    {
        const int tid = omp_get_thread_num();
        if (tid < 0 || tid >= requested)
            invalid.fetch_add(1);
        else
        {
            Participant& r = results[tid];
            r.calls++;
            r.team = omp_get_num_threads();
            r.value = tid;
            r.backend = on_backend();
        }
    }
    // Inspection after return verifies synchronous completion.
    return check(invalid == 0, "parallel", "invalid thread id")
           | check_participants(results, expected, 0, "parallel")
           | check_streams(streams, "parallel return");
}

template<typename Index>
static int test_static_loop(int threads, Index count)
{
    std::atomic<int> visits[17];
    for (int i = 0; i < 17; i++) visits[i] = 0;
    #pragma omp parallel for num_threads(threads)
    for (Index i = 0; i < count; i++)
        visits[i].fetch_add(1);

    for (int i = 0; i < 17; i++)
    {
        if (visits[i] != (i < (int)count ? 1 : 0))
        {
            fprintf(stderr, "test_spertomp static_loop failed: threads=%d index_bits=%zu count=%d i=%d visits=%d\n", threads, sizeof(Index) * 8, (int)count, i, visits[i].load());
            return 1;
        }
    }
    return check_streams(0, "static_loop return");
}

#ifdef __clang__
extern "C" {
    typedef void (*kmpc_micro)(int32_t*, int32_t*, ...);
    void __kmpc_push_num_threads(void*, int32_t, int32_t);
    void __kmpc_fork_call(void*, int32_t, kmpc_micro, ...);
    int32_t __kmpc_global_thread_num(void*);
}

static void record(int32_t* gtid, int32_t* tid, Participant* results)
{
    Participant& r = results[*tid];
    r.calls++;
    r.team = omp_get_num_threads();
    r.value = omp_get_thread_num();
    r.backend = on_backend() && *gtid == *tid && __kmpc_global_thread_num(0) == *tid;
}

template<size_t>
using IntArgument = int*;

template<size_t... I>
static void record_arguments(int32_t* gtid, int32_t* tid, Participant* results, IntArgument<I>... args)
{
    record(gtid, tid, results);
    // Distinct weights detect reordered arguments as well as missing ones.
    ((results[*tid].value += (int)(I + 1) * *args), ...);
}

template<size_t... I>
static int test_clang_abi(int requested, std::index_sequence<I...>)
{
    const int cores = omp_get_max_threads();
    const int expected = std::max(1, std::min(requested, cores));
    std::vector<Participant> results(std::max(cores, requested));
    int args[sizeof...(I) + 1];
    int offset = 0;
    for (size_t i = 0; i < sizeof...(I); i++)
    {
        args[i] = (int)i + 1;
        offset += ((int)i + 1) * args[i];
    }
    __kmpc_push_num_threads(0, 0, requested);
    __kmpc_fork_call(0, 1 + sizeof...(I), (kmpc_micro)record_arguments<I...>, results.data(), &args[I]...);
    char label[64];
    snprintf(label, sizeof(label), "clang_abi requested=%d argc=%zu", requested, 1 + sizeof...(I));
    return check_participants(results, expected, offset, label) | check_streams(0, "clang_abi return");
}

template<size_t Argc>
static int test_clang_abi(int requested)
{
    return test_clang_abi(requested, std::make_index_sequence < Argc - 1 > ());
}
#endif

static int test_nested_run()
{
    const size_t streams = spert::runtime_stats().active_streams;
    ncnn::Option opt;
    int inner_calls = 0;
    int bad = 0;
    const int ret = ncnn::spert_run([&]()
    {
        bad |= check(on_backend(), "nested_run", "outer callback ran on caller");
        bad |= check_streams(streams + 1, "outer callback");
        bad |= test_parallel(1);
        bad |= test_parallel(omp_get_max_threads());
        const int nested_ret = ncnn::spert_run([&]()
        {
            inner_calls++;
            bad |= check(on_backend(), "nested_run", "inner callback ran on caller");
            bad |= check_streams(streams + 1, "inner callback");
            return -7;
        },
        opt);
        bad |= check(nested_ret == -7 && inner_calls == 1, "nested_run", "lost callback result");
        bad |= check(omp_get_num_threads() == 1 && omp_get_thread_num() == 0, "nested_run", "parent team state not restored");
        return -9;
    },
    opt);
    return bad | check(ret == -9, "nested_run", "lost outer callback result")
           | check_streams(streams, "nested_run return");
}

static int test_stream_scope()
{
    int bad = check_streams(0, "before scope");
    {
        ncnn::SpertStreamScope outer;
        if (!outer.valid()) return check(false, "stream_scope", "invalid outer stream");
        bad |= check_streams(1, "outer scope");
        {
            ncnn::SpertStreamScope inner;
            bad |= check(inner.valid(), "stream_scope", "invalid inner stream");
            bad |= check_streams(1, "nested scope");
        }
        bad |= check_streams(1, "nested scope return");
    }
    bad |= check_streams(0, "outer scope return");
    bad |= test_nested_run();

    // Leave one available core: blocking sync inside a tile would deadlock.
    const unsigned cores = (unsigned)spert::backend_info().num_cores;
    if (cores > 1)
    {
        spert::Stream reserved(cores - 1);
        if (!reserved.valid()) return check(false, "stream_scope", "could not reserve cores");
        bad |= test_nested_run();
    }
    return bad | check_streams(0, "stream_scope return");
}

struct LayerState
{
    LayerState()
        : pipelines(0), forwards(0), bad(0), fail_pipeline(false), fail_forward(false)
    {
    }
    int pipelines;
    int forwards;
    int bad;
    bool fail_pipeline;
    bool fail_forward;
};

class SpertTestLayer : public ncnn::Layer
{
public:
    explicit SpertTestLayer(LayerState* state)
        : state(state)
    {
        one_blob_only = true;
    }

    virtual int create_pipeline(const ncnn::Option& opt)
    {
        state->pipelines++;
        probe(opt, "create_pipeline");
        return state->fail_pipeline ? -1 : 0;
    }

    virtual int forward(const ncnn::Mat& bottom, ncnn::Mat& top, const ncnn::Option& opt) const
    {
        state->forwards++;
        probe(opt, "forward");
        if (state->fail_forward) return -7;
        top.create(bottom.w, (size_t)4u, opt.blob_allocator);
        if (top.empty()) return -100;
        const float* in = bottom;
        float* out = top;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < bottom.w; i++) out[i] = in[i] + 1.f;
        return 0;
    }

private:
    void probe(const ncnn::Option& opt, const char* where) const
    {
        state->bad |= check(on_backend(), where, "CPU work ran on caller");
        state->bad |= check_streams(1, where);
        state->bad |= test_parallel(opt.num_threads);
    }
    LayerState* state;
};

static ncnn::Layer* create_test_layer(void* userdata)
{
    return new SpertTestLayer((LayerState*)userdata);
}

static ncnn::Layer* create_input_layer(void*)
{
    // Its output is supplied by Extractor::input(), so forward is never called.
    return new ncnn::Layer;
}

static int test_net(int threads, bool fail_pipeline, bool fail_forward)
{
    LayerState state;
    state.fail_pipeline = fail_pipeline;
    state.fail_forward = fail_forward;
    ncnn::Net net;
    net.opt.num_threads = threads;
    net.opt.use_packing_layout = false;
    net.opt.use_vulkan_compute = false;
    // Custom layers only, independent of optional built-in layer selection.
    net.register_custom_layer("TestInput", create_input_layer);
    net.register_custom_layer("SpertTest", create_test_layer, 0, &state);
    if (net.load_param_mem("7767517\n3 3\nTestInput input 0 1 in\nSpertTest first 1 1 in mid\nSpertTest second 1 1 mid out\n") != 0)
        return check(false, "net", "load_param failed");
    ncnn::DataReader dr;
    const int ret = net.load_model(dr);
    int bad = state.bad | check_streams(0, "load_model return");
    if (fail_pipeline)
        return bad | check(ret != 0, "net", "pipeline error was lost");
    if (ret != 0) return check(false, "net", "load_model failed");
    bad |= check(state.pipelines == 2, "net", "pipeline count differs");

    ncnn::Mat input(17);
    input.fill(2.f);
    for (int run = 0; run < 2; run++)
    {
        ncnn::Extractor ex = net.create_extractor();
        if (ex.input("in", input) != 0) return check(false, "net", "input failed");
        ncnn::Mat output;
        const int extract_ret = ex.extract("out", output);
        bad |= check_streams(0, "extract return");
        if (fail_forward)
        {
            bad |= check(extract_ret == -7, "net", "forward error was lost");
            continue;
        }
        if (extract_ret != 0 || output.empty() || output.dims != 1 || output.w != 17 || output.elempack != 1 || output.elembits() != 32)
        {
            fprintf(stderr, "test_spertomp net: ret=%d dims=%d w=%d total=%zu\n", extract_ret, output.dims, output.w, output.total());
            return check(false, "net", "invalid output");
        }
        for (int i = 0; i < 17; i++) bad |= check(output[i] == 4.f, "net", "incorrect output value");
        const int calls = state.forwards;
        bad |= check(ex.extract("out", output) == 0 && state.forwards == calls, "net", "cached output recomputed");
        bad |= check(ex.extract(-1, output) != 0, "net", "invalid output index accepted");
        bad |= check_streams(0, "cached/invalid extract return");
    }
    return bad | state.bad | check(state.forwards == (fail_forward ? 2 : 4), "net", "forward count differs");
}

int main()
{
    const int cores = (int)spert::backend_info().num_cores;
    if (check(cores > 0, "backend", "no compute cores")) return 1;
    is_caller = true;
    int failed = check_streams(0, "startup");
    const int requests[] = {1, 2, 4, 8, cores + 1};
    for (unsigned i = 0; i < sizeof(requests) / sizeof(requests[0]); i++)
    {
        const int threads = requests[i];
        failed |= test_parallel(threads);
        failed |= test_static_loop<int32_t>(threads, 17);
        failed |= test_static_loop<uint32_t>(threads, 3);
        failed |= test_static_loop<int64_t>(threads, 17);
        failed |= test_static_loop<uint64_t>(threads, 3);
#ifdef __clang__
        failed |= test_clang_abi<1>(threads);
        failed |= test_clang_abi<5>(threads);
        failed |= test_clang_abi<12>(threads);
        failed |= test_clang_abi<13>(threads);
        failed |= test_clang_abi<14>(threads);
        failed |= test_clang_abi<15>(threads);
        failed |= test_clang_abi<16>(threads);
#endif
        failed |= test_net(threads, false, false);
    }
#ifdef __clang__
    failed |= test_clang_abi<1>(0);
#endif
    failed |= test_stream_scope();
    failed |= test_net(1, true, false);
    failed |= test_net(1, false, true);
    failed |= check_streams(0, "shutdown");
    fprintf(stderr, "test_spertomp: %s (parallel, static_loop, stream_scope, net%s)\n", failed ? "FAILED" : "PASSED",
#ifdef __clang__
            ", clang_abi"
#else
            ""
#endif
           );
    return failed;
}
