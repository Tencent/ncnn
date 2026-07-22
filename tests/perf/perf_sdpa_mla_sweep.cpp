// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "allocator.h"
#include "benchmark.h"
#include "cpu.h"
#include "layer.h"
#include "mat.h"
#include "modelbin.h"
#include "option.h"
#include "paramdict.h"

#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

class TrackingAllocator : public ncnn::Allocator
{
public:
    TrackingAllocator() : peak_bytes(0), live_bytes(0), baseline_live_bytes(0), total_bytes(0), num_allocs(0) {}

    virtual void* fastMalloc(size_t size)
    {
        size_t total = size + 16;
        void* raw = ncnn::fastMalloc(total);
        ((size_t*)raw)[0] = size;
        live_bytes += size;
        total_bytes += size;
        num_allocs++;
        size_t relative_live_bytes = live_bytes > baseline_live_bytes ? live_bytes - baseline_live_bytes : 0;
        if (relative_live_bytes > peak_bytes)
            peak_bytes = relative_live_bytes;
        return (void*)((char*)raw + 16);
    }

    virtual void fastFree(void* ptr)
    {
        if (!ptr)
            return;

        void* raw = (void*)((char*)ptr - 16);
        size_t size = ((size_t*)raw)[0];
        if (size <= live_bytes)
            live_bytes -= size;
        else
            live_bytes = 0;
        ncnn::fastFree(raw);
    }

    void reset()
    {
        peak_bytes = 0;
        baseline_live_bytes = live_bytes;
        total_bytes = 0;
        num_allocs = 0;
    }

    size_t peak_bytes;
    size_t live_bytes;
    size_t baseline_live_bytes;
    size_t total_bytes;
    int num_allocs;
};

struct Report
{
    double median_ms;
    size_t ws_peak;
    size_t blob_peak;
    size_t op_peak;
    size_t ws_total;
    size_t blob_total;
    int ws_allocs;
    int blob_allocs;
};

static int env_int(const char* name, int default_value)
{
    const char* s = getenv(name);
    if (!s || !s[0])
        return default_value;
    return atoi(s);
}

static bool env_match_int(const char* name, int value)
{
    const char* s = getenv(name);
    if (!s || !s[0])
        return true;
    return atoi(s) == value;
}

static bool env_match_str(const char* name, const char* value)
{
    const char* s = getenv(name);
    if (!s || !s[0])
        return true;
    return strcmp(s, value) == 0;
}

static ncnn::Mat make_mat(int w, int h, int c, float v = 0.01f)
{
    ncnn::Mat m(w, h, c);
    m.fill(v);
    return m;
}

static ncnn::Mat make_causal_mask(int src_seqlen, int past_seqlen)
{
    const int total = src_seqlen + past_seqlen;
    ncnn::Mat mask(total, src_seqlen);
    mask.fill(0.f);

    for (int i = 0; i < src_seqlen; i++)
    {
        float* row = mask.row(i);
        for (int j = past_seqlen + i + 1; j < total; j++)
            row[j] = -1e38f;
    }

    return mask;
}

static void convert_input_layout(const ncnn::Mat& src, ncnn::Mat& dst, const ncnn::Option& opt, const ncnn::Layer* op)
{
    ncnn::Mat casted;

#if NCNN_ARM82
    if (opt.use_fp16_storage && ncnn::cpu_support_arm_asimdhp() && op->support_fp16_storage)
    {
        ncnn::cast_float32_to_float16(src, casted, opt);
    }
    else
#endif
    if (opt.use_fp16_storage && op->support_fp16_storage)
    {
        ncnn::cast_float32_to_float16(src, casted, opt);
    }
    else
    {
        casted = src;
    }

    if (opt.use_packing_layout && op->support_packing)
    {
        int elemcount = 0;
        if (casted.dims == 1) elemcount = casted.elempack * casted.w;
        if (casted.dims == 2) elemcount = casted.elempack * casted.h;
        if (casted.dims == 3 || casted.dims == 4) elemcount = casted.elempack * casted.c;

        int dst_elempack = 1;
        if (casted.elembits() == 16)
        {
#if NCNN_ARM82
            if (elemcount % 8 == 0 && ncnn::cpu_support_arm_asimdhp() && opt.use_fp16_arithmetic && op->support_fp16_storage)
                dst_elempack = 8;
            else if (elemcount % 4 == 0)
                dst_elempack = 4;
#else
            if (elemcount % 4 == 0)
                dst_elempack = 4;
#endif
        }
        else if (casted.elembits() == 32)
        {
            if (elemcount % 4 == 0)
                dst_elempack = 4;
        }
        else if (casted.elembits() == 8)
        {
            if (elemcount % 8 == 0)
                dst_elempack = 8;
        }

        ncnn::convert_packing(casted, dst, dst_elempack, opt);
    }
    else
    {
        dst = casted;
    }
}

static void convert_input_layout_persistent_view(const ncnn::Mat& src, ncnn::Mat& dst, const ncnn::Option& opt, const ncnn::Layer* op)
{
    ncnn::Mat src_full = src;
    const int capacity = src.w == 0 ? src.h : (int)(src.cstep / src.w);
    src_full.h = capacity;

    convert_input_layout(src_full, dst, opt, op);
    dst.h = src.h;
}

static Report run_sdpa(const char* phase, int kv_cache, int M, int past, int n_ctx, int threads)
{
    const int d_k = 192;
    const int d_v = 128;
    const int heads = 128;
    const int groups = 16;
    const int warmup_count = env_int("NCNN_MLA_SWEEP_WARMUP", 3);
    const int run_count = env_int("NCNN_MLA_SWEEP_RUNS", 9);

    ncnn::ParamDict pd;
    pd.set(5, strcmp(phase, "prefill") == 0 ? 1 : 0);
    pd.set(6, 0.f);
    pd.set(7, kv_cache);

    ncnn::Layer* op = ncnn::create_layer_cpu("SDPA");
    op->load_param(pd);
    std::vector<ncnn::Mat> weights(0);
    ncnn::ModelBinFromMatArray mb(weights.data());
    op->load_model(mb);

    TrackingAllocator ws_alloc;
    TrackingAllocator blob_alloc;

    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = threads;
    opt.use_packing_layout = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;
    opt.use_fp16_packed = true;
    opt.use_bf16_storage = false;
    opt.workspace_allocator = &ws_alloc;
    opt.blob_allocator = &blob_alloc;

    op->create_pipeline(opt);

    std::vector<ncnn::Mat> inputs;
    int output_count = 1;

    if (strcmp(phase, "prefill") == 0 && kv_cache == 0)
    {
        inputs.resize(4);
        inputs[0] = make_mat(d_k, M, heads);
        inputs[1] = make_mat(d_k, M, groups);
        inputs[2] = make_mat(d_v, M, groups);
        inputs[3] = make_causal_mask(M, 0);
    }
    else if (strcmp(phase, "prefill") == 0 && kv_cache == 2)
    {
        inputs.resize(6);
        inputs[0] = make_mat(d_k, M, heads);
        inputs[1] = make_mat(d_k, M, groups);
        inputs[2] = make_mat(d_v, M, groups);
        inputs[3] = make_causal_mask(M, 0);
        inputs[4] = make_mat(d_k, n_ctx, groups);
        inputs[5] = make_mat(d_v, n_ctx, groups);
        inputs[4].h = 0;
        inputs[5].h = 0;
        output_count = 3;
    }
    else if (strcmp(phase, "decode") == 0 && kv_cache == 1)
    {
        inputs.resize(5);
        inputs[0] = make_mat(d_k, 1, heads);
        inputs[1] = make_mat(d_k, 1, groups);
        inputs[2] = make_mat(d_v, 1, groups);
        inputs[3] = make_mat(d_k, past, groups);
        inputs[4] = make_mat(d_v, past, groups);
        output_count = 3;
    }
    else if (strcmp(phase, "decode") == 0 && kv_cache == 2)
    {
        inputs.resize(5);
        inputs[0] = make_mat(d_k, 1, heads);
        inputs[1] = make_mat(d_k, 1, groups);
        inputs[2] = make_mat(d_v, 1, groups);
        inputs[3] = make_mat(d_k, n_ctx, groups);
        inputs[4] = make_mat(d_v, n_ctx, groups);
        inputs[3].h = past;
        inputs[4].h = past;
        output_count = 3;
    }

    std::vector<ncnn::Mat> converted_inputs(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        const int cache_offset = strcmp(phase, "prefill") == 0 ? 4 : 3;
        if (kv_cache == 2 && (i == (size_t)cache_offset || i == (size_t)(cache_offset + 1)))
            convert_input_layout_persistent_view(inputs[i], converted_inputs[i], opt, op);
        else
            convert_input_layout(inputs[i], converted_inputs[i], opt, op);
    }

    for (int i = 0; i < warmup_count; i++)
    {
        std::vector<ncnn::Mat> outputs(output_count);
        op->forward(converted_inputs, outputs, opt);
    }

    std::vector<double> times;
    times.reserve(run_count);
    Report r = {0.0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < run_count; i++)
    {
        ws_alloc.reset();
        blob_alloc.reset();

        double t0 = ncnn::get_current_time();
        {
            std::vector<ncnn::Mat> outputs(output_count);
            op->forward(converted_inputs, outputs, opt);
        }
        double t1 = ncnn::get_current_time();

        times.push_back(t1 - t0);
        if (ws_alloc.peak_bytes + blob_alloc.peak_bytes > r.op_peak)
        {
            r.ws_peak = ws_alloc.peak_bytes;
            r.blob_peak = blob_alloc.peak_bytes;
            r.op_peak = ws_alloc.peak_bytes + blob_alloc.peak_bytes;
            r.ws_total = ws_alloc.total_bytes;
            r.blob_total = blob_alloc.total_bytes;
            r.ws_allocs = ws_alloc.num_allocs;
            r.blob_allocs = blob_alloc.num_allocs;
        }
    }

    std::sort(times.begin(), times.end());
    r.median_ms = times[times.size() / 2];

    op->destroy_pipeline(opt);
    delete op;

    return r;
}

static void print_report(const char* mode, const char* phase, int kv_cache, int len, int n_ctx, int threads, const Report& r)
{
    fprintf(stdout,
            "%-8s %-7s kv=%d len=%-5d ctx=%-5d t=%d median_ms=%9.4f ws_peak=%10zu blob_peak=%10zu op_peak=%10zu ws_total=%10zu blob_total=%10zu ws_allocs=%3d blob_allocs=%3d\n",
            mode, phase, kv_cache, len, n_ctx, threads, r.median_ms,
            r.ws_peak, r.blob_peak, r.op_peak, r.ws_total, r.blob_total, r.ws_allocs, r.blob_allocs);
}

int main()
{
    const int threads = env_int("NCNN_PERF_THREADS", 4);
    const int n_ctx = env_int("NCNN_PERF_SDPA_CTX", 4096);

    int prefill_lengths[] = {128, 256, 512, 1024};
    int decode_lengths[] = {128, 512, 1024, 2048};

    fprintf(stdout, "# Youtu MLA SDPA sweep: d_k=192 d_v=128 heads=128 groups=16 dtype=fp16psa\n");
    fprintf(stdout, "# env filters: NCNN_MLA_SWEEP_MODE=baseline|current NCNN_MLA_SWEEP_PHASE=prefill|decode NCNN_PERF_SDPA_M=... NCNN_PERF_SDPA_PAST=...\n");

    for (size_t i = 0; i < sizeof(prefill_lengths) / sizeof(prefill_lengths[0]); i++)
    {
        const int M = prefill_lengths[i];
        if (!env_match_str("NCNN_MLA_SWEEP_PHASE", "prefill") || !env_match_int("NCNN_PERF_SDPA_M", M))
            continue;

        if (env_match_str("NCNN_MLA_SWEEP_MODE", "baseline"))
            print_report("baseline", "prefill", 0, M, n_ctx, threads, run_sdpa("prefill", 0, M, 0, n_ctx, threads));
        if (env_match_str("NCNN_MLA_SWEEP_MODE", "current"))
            print_report("current", "prefill", 2, M, n_ctx, threads, run_sdpa("prefill", 2, M, 0, n_ctx, threads));
    }

    for (size_t i = 0; i < sizeof(decode_lengths) / sizeof(decode_lengths[0]); i++)
    {
        const int past = decode_lengths[i];
        if (!env_match_str("NCNN_MLA_SWEEP_PHASE", "decode") || !env_match_int("NCNN_PERF_SDPA_PAST", past))
            continue;

        if (env_match_str("NCNN_MLA_SWEEP_MODE", "baseline"))
            print_report("baseline", "decode", 1, past, n_ctx, threads, run_sdpa("decode", 1, 1, past, n_ctx, threads));
        if (env_match_str("NCNN_MLA_SWEEP_MODE", "current"))
            print_report("current", "decode", 2, past, n_ctx, threads, run_sdpa("decode", 2, 1, past, n_ctx, threads));
    }

    return 0;
}
