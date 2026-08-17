// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

#include "benchmark.h"
#include "modelbin.h"

#include <stdio.h>

#define SESSION_WARMUP_COUNT 2
#define SESSION_RUN_COUNT    5

class KVCachePerfAllocator : public ncnn::Allocator
{
public:
    KVCachePerfAllocator()
        : allocation_count(0)
    {
    }

    virtual void* fastMalloc(size_t size)
    {
        void* ptr = ncnn::fastMalloc(size);
        if (ptr)
            allocation_count++;

        return ptr;
    }

    virtual void fastFree(void* ptr)
    {
        ncnn::fastFree(ptr);
    }

    int allocation_count;
};

struct SessionResult
{
    double time;
    int relocation_count;
    int allocation_count;
};

static void print_session_result(const char* device, const char* cache_type, int embed_dim, int num_heads, int num_groups, int prefill_seqlen, int decode_steps, const SessionResult* results)
{
    double times[SESSION_RUN_COUNT];
    double time_avg = 0.0;
    for (int i = 0; i < SESSION_RUN_COUNT; i++)
    {
        times[i] = results[i].time / decode_steps;
        time_avg += times[i];
    }

    for (int i = 1; i < SESSION_RUN_COUNT; i++)
    {
        double key = times[i];
        int j = i - 1;
        while (j >= 0 && times[j] > key)
        {
            times[j + 1] = times[j];
            j--;
        }
        times[j + 1] = key;
    }
    time_avg /= SESSION_RUN_COUNT;

    fprintf(stdout, "SDPA %-3s %-12s embed=%d heads=%d groups=%d prefill=%d steps=%d  min=%.3f  max=%.3f  avg=%.3f  median=%.3f ms/step  reloc=%d",
            device, cache_type, embed_dim, num_heads, num_groups, prefill_seqlen, decode_steps,
            times[0], times[SESSION_RUN_COUNT - 1], time_avg, times[SESSION_RUN_COUNT / 2],
            results[SESSION_RUN_COUNT - 1].relocation_count);

    if (results[SESSION_RUN_COUNT - 1].allocation_count >= 0)
        fprintf(stdout, "  kvcache_alloc=%d", results[SESSION_RUN_COUNT - 1].allocation_count);

    fprintf(stdout, "\n");
}

static int run_sdpa_cpu_session(ncnn::Layer* op, const ncnn::Option& opt, KVCachePerfAllocator& allocator, int embed_dim, int num_heads, int num_groups, int prefill_seqlen, int decode_steps, SessionResult& result)
{
    const int allocation_count = allocator.allocation_count;

    std::vector<ncnn::Mat> prefill_bottoms(5);
    prefill_bottoms[0] = PerfMat(embed_dim, prefill_seqlen, num_heads);
    prefill_bottoms[1] = PerfMat(embed_dim, prefill_seqlen, num_groups);
    prefill_bottoms[2] = PerfMat(embed_dim, prefill_seqlen, num_groups);

    std::vector<ncnn::Mat> prefill_tops(3);
    int ret = op->forward(prefill_bottoms, prefill_tops, opt);
    if (ret != 0)
        return ret;

    ncnn::Mat key_cache = prefill_tops[1];
    ncnn::Mat value_cache = prefill_tops[2];
    prefill_tops[1].release();
    prefill_tops[2].release();

    ncnn::Mat query = PerfMat(embed_dim, 1, num_heads);
    ncnn::Mat current_key = PerfMat(embed_dim, 1, num_groups);
    ncnn::Mat current_value = PerfMat(embed_dim, 1, num_groups);

    result.relocation_count = 0;
    const double time_start = ncnn::get_current_time();
    for (int i = 0; i < decode_steps; i++)
    {
        void* key_data = key_cache.data;

        std::vector<ncnn::Mat> bottom_blobs(5);
        bottom_blobs[0] = query;
        bottom_blobs[1] = current_key;
        bottom_blobs[2] = current_value;
        bottom_blobs[3] = key_cache;
        bottom_blobs[4] = value_cache;
        key_cache.release();
        value_cache.release();

        std::vector<ncnn::Mat> top_blobs(3);
        ret = op->forward(bottom_blobs, top_blobs, opt);
        if (ret != 0)
            return ret;

        if (top_blobs[1].data != key_data)
            result.relocation_count++;

        key_cache = top_blobs[1];
        value_cache = top_blobs[2];
        top_blobs[1].release();
        top_blobs[2].release();
        bottom_blobs[3].release();
        bottom_blobs[4].release();
    }
    result.time = ncnn::get_current_time() - time_start;
    result.allocation_count = opt.kvcache_allocator ? allocator.allocation_count - allocation_count : -1;

    key_cache.release();
    value_cache.release();

    return 0;
}

static void perf_sdpa_kvcache_cpu(int embed_dim, int num_heads, int num_groups, int prefill_seqlen, int decode_steps, int max_seqlen_hint)
{
    KVCachePerfAllocator allocator;

    ncnn::Option opt;
    opt.kvcache_allocator = max_seqlen_hint >= 0 ? &allocator : 0;
    opt.kvcache_max_seqlen_hint = max_seqlen_hint > 0 ? max_seqlen_hint : 0;

    ncnn::Layer* op = ncnn::create_layer_cpu("SDPA");
    if (!op)
        return;

    ncnn::ParamDict pd;
    pd.set(7, 1);
    op->load_param(pd);
    op->load_model(ncnn::ModelBinFromMatArray(0));
    if (op->create_pipeline(opt) != 0)
    {
        delete op;
        return;
    }

    SessionResult result;
    int ret = 0;
    for (int i = 0; ret == 0 && i < SESSION_WARMUP_COUNT; i++)
    {
        ret = run_sdpa_cpu_session(op, opt, allocator, embed_dim, num_heads, num_groups, prefill_seqlen, decode_steps, result);
    }

    SessionResult results[SESSION_RUN_COUNT];
    int run_count = 0;
    for (; ret == 0 && run_count < SESSION_RUN_COUNT; run_count++)
    {
        ret = run_sdpa_cpu_session(op, opt, allocator, embed_dim, num_heads, num_groups, prefill_seqlen, decode_steps, results[run_count]);
    }

    if (run_count == SESSION_RUN_COUNT)
    {
        const char* cache_type = "legacy";
        if (opt.kvcache_allocator)
            cache_type = max_seqlen_hint > 0 ? "allocator-hint" : "allocator";
        print_session_result("cpu", cache_type, embed_dim, num_heads, num_groups, prefill_seqlen, decode_steps, results);
    }

    op->destroy_pipeline(opt);
    delete op;
}

static void perf_sdpa_kvcache(int embed_dim, int num_heads, int num_groups, int prefill_seqlen, int decode_steps)
{
    const int max_seqlen_hint = prefill_seqlen + decode_steps;

    perf_sdpa_kvcache_cpu(embed_dim, num_heads, num_groups, prefill_seqlen, decode_steps, -1);
    perf_sdpa_kvcache_cpu(embed_dim, num_heads, num_groups, prefill_seqlen, decode_steps, 0);
    perf_sdpa_kvcache_cpu(embed_dim, num_heads, num_groups, prefill_seqlen, decode_steps, max_seqlen_hint);
}

int main()
{
    perf_sdpa_kvcache(128, 4, 4, 128, 512);
    perf_sdpa_kvcache(512, 8, 1, 128, 512);

    return 0;
}
