// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause
//
// Memory comparison: SDPA Gemm path vs Flash path at MLA dimensions.
//
// Each path is run once with a tracking allocator that records:
//   - peak workspace bytes (qk_cross etc)
//   - peak blob bytes (top_blob, key/value copies)
//   - total bytes allocated cumulative over the call
//
// Flash's main theoretical advantage is avoiding the O(M*N*heads) qk_cross
// matrix. This tool quantifies that for our model dims.

#include "allocator.h"
#include "benchmark.h"
#include "layer.h"
#include "mat.h"
#include "modelbin.h"
#include "option.h"
#include "paramdict.h"

#include <stdio.h>
#include <string.h>
#include <vector>

// -----------------------------------------------------------------------------
// Tracking allocator: counts bytes through fastMalloc/fastFree.
// Reports peak (max live) and total (cumulative allocated) bytes.
// -----------------------------------------------------------------------------
class TrackingAllocator : public ncnn::Allocator
{
public:
    TrackingAllocator() : peak_bytes(0), live_bytes(0), total_bytes(0), num_allocs(0) {}

    virtual void* fastMalloc(size_t size)
    {
        // Allocate with header to remember size on free
        size_t total = size + 16;
        void* raw = ncnn::fastMalloc(total);
        ((size_t*)raw)[0] = size;
        live_bytes += size;
        total_bytes += size;
        num_allocs++;
        if (live_bytes > peak_bytes) peak_bytes = live_bytes;
        return (void*)((char*)raw + 16);
    }

    virtual void fastFree(void* ptr)
    {
        if (!ptr) return;
        void* raw = (void*)((char*)ptr - 16);
        size_t size = ((size_t*)raw)[0];
        live_bytes -= size;
        ncnn::fastFree(raw);
    }

    void reset()
    {
        peak_bytes = 0;
        live_bytes = 0;
        total_bytes = 0;
        num_allocs = 0;
    }

    size_t peak_bytes;
    size_t live_bytes;
    size_t total_bytes;
    int num_allocs;
};

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
static ncnn::Mat make_mat(int w, int h, int c, float v = 0.01f)
{
    ncnn::Mat m(w, h, c);
    m.fill(v);
    return m;
}

// -----------------------------------------------------------------------------
// Run SDPA forward with the given inputs, return tracked memory
// -----------------------------------------------------------------------------
struct MemReport
{
    size_t ws_peak;
    size_t ws_total;
    int    ws_allocs;
    size_t blob_peak;
    size_t blob_total;
    int    blob_allocs;
    double time_ms;
};

static MemReport run_sdpa(int kv_cache_mode,
                          int d_k, int d_v, int heads, int groups,
                          int M, int N, int n_ctx,
                          bool use_fp16)
{
    MemReport r = {0, 0, 0, 0, 0, 0, 0.0};

    ncnn::ParamDict pd;
    pd.set(5, 0);    // attn_mask
    pd.set(6, 0.f);  // scale (auto)
    pd.set(7, kv_cache_mode);

    ncnn::Layer* op = ncnn::create_layer_cpu("SDPA");
    op->load_param(pd);
    std::vector<ncnn::Mat> weights(0);
    ncnn::ModelBinFromMatArray mb(weights.data());
    op->load_model(mb);

    TrackingAllocator ws_alloc;
    TrackingAllocator blob_alloc;

    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 1;
    opt.use_packing_layout = false;
    opt.use_fp16_storage = use_fp16;
    opt.use_fp16_arithmetic = use_fp16;
    opt.use_fp16_packed = use_fp16;
    opt.use_bf16_storage = false;
    opt.workspace_allocator = &ws_alloc;
    opt.blob_allocator = &blob_alloc;

    op->create_pipeline(opt);

    // Build inputs (as fp32; they'll be cast inside SDPA forward)
    std::vector<ncnn::Mat> inputs;
    int n_outputs = 1;

    if (kv_cache_mode == 0)
    {
        // No kv_cache: just q, k, v
        inputs.resize(3);
        inputs[0] = make_mat(d_k, M, heads);
        inputs[1] = make_mat(d_k, N, groups);
        inputs[2] = make_mat(d_v, N, groups);
    }
    else if (kv_cache_mode == 2)
    {
        // kv_cache=2: q, cur_k, cur_v, past_k(view over preallocated cache), past_v(view over preallocated cache)
        inputs.resize(5);
        inputs[0] = make_mat(d_k, M, heads);
        inputs[1] = make_mat(d_k, N, groups);   // cur_k = N rows (past=0 prefill)
        inputs[2] = make_mat(d_v, N, groups);   // cur_v
        inputs[3] = make_mat(d_k, n_ctx, groups);
        inputs[4] = make_mat(d_v, n_ctx, groups);
        n_outputs = 3;
    }

    // Need to convert inputs for fp16 path
    if (use_fp16)
    {
        for (size_t i = 0; i < inputs.size(); i++)
        {
            ncnn::Mat casted;
            ncnn::cast_float32_to_float16(inputs[i], casted, opt);
            inputs[i] = casted;
        }
    }

    if (kv_cache_mode == 2)
    {
        inputs[3].h = 0;
        inputs[4].h = 0;
    }

    // Reset trackers (input creation went through them)
    ws_alloc.reset();
    blob_alloc.reset();

    // Warmup once (caches workspace pool)
    {
        std::vector<ncnn::Mat> outs(n_outputs);
        op->forward(inputs, outs, opt);
    }

    // Reset, then run a single timed call
    ws_alloc.reset();
    blob_alloc.reset();

    double t0 = ncnn::get_current_time();
    {
        std::vector<ncnn::Mat> outs(n_outputs);
        op->forward(inputs, outs, opt);
    }
    double t1 = ncnn::get_current_time();

    r.ws_peak = ws_alloc.peak_bytes;
    r.ws_total = ws_alloc.total_bytes;
    r.ws_allocs = ws_alloc.num_allocs;
    r.blob_peak = blob_alloc.peak_bytes;
    r.blob_total = blob_alloc.total_bytes;
    r.blob_allocs = blob_alloc.num_allocs;
    r.time_ms = t1 - t0;

    op->destroy_pipeline(opt);
    delete op;
    return r;
}

// -----------------------------------------------------------------------------
// Pretty printing
// -----------------------------------------------------------------------------
static const char* fmt_bytes(size_t b, char* buf)
{
    if (b >= (size_t)1024 * 1024 * 1024)
        snprintf(buf, 32, "%6.2f GB", b / (1024.0 * 1024.0 * 1024.0));
    else if (b >= (size_t)1024 * 1024)
        snprintf(buf, 32, "%6.2f MB", b / (1024.0 * 1024.0));
    else if (b >= 1024)
        snprintf(buf, 32, "%6.2f KB", b / 1024.0);
    else
        snprintf(buf, 32, "%5zu  B", b);
    return buf;
}

int main()
{
    const int d_k = 192;
    const int d_v = 128;
    const int heads = 128;
    const int groups = 16;
    const int n_ctx = 4096;

    int seqlens[] = {32, 64, 128, 256, 512, 1024, 2048};

    fprintf(stdout,
        "Per-call SDPA memory comparison (Youtu-LLM-2B dims: d_k=192 d_v=128 h=128 g=16, fp16)\n"
        "  GEMM = ncnn Gemm path (kv_cache=0)\n"
        "  FLASH = our flash kernel (kv_cache=2, past=0, n_ctx=%d)\n"
        "  workspace = qk_cross / K_packed / Q_packed (intermediate)\n"
        "  blob      = top_blob (output) + intermediate Mats (key/value copies)\n"
        "  total     = cumulative bytes allocated (incl. reused pool slots)\n\n",
        n_ctx);

    fprintf(stdout,
        "%-6s | %-10s | %-10s %-10s %4s | %-10s %-10s %4s | %s\n",
        "M=N", "PATH", "WS peak", "WS total", "WS#",
                       "Blob peak", "Blob total", "B#",
                       "time(ms)");
    fprintf(stdout, "%.110s\n", "------------------------------------------------------------------------------------------------------------");

    for (size_t i = 0; i < sizeof(seqlens) / sizeof(seqlens[0]); i++)
    {
        int s = seqlens[i];
        MemReport gemm  = run_sdpa(0, d_k, d_v, heads, groups, s, s, n_ctx, true);
        MemReport flash = run_sdpa(2, d_k, d_v, heads, groups, s, s, n_ctx, true);

        char b1[32], b2[32], b3[32], b4[32];
        fprintf(stdout, "%-6d | %-10s | %-10s %-10s %4d | %-10s %-10s %4d | %7.2f\n",
                s, "GEMM",
                fmt_bytes(gemm.ws_peak, b1), fmt_bytes(gemm.ws_total, b2), gemm.ws_allocs,
                fmt_bytes(gemm.blob_peak, b3), fmt_bytes(gemm.blob_total, b4), gemm.blob_allocs,
                gemm.time_ms);
        fprintf(stdout, "%-6s | %-10s | %-10s %-10s %4d | %-10s %-10s %4d | %7.2f\n",
                "", "FLASH",
                fmt_bytes(flash.ws_peak, b1), fmt_bytes(flash.ws_total, b2), flash.ws_allocs,
                fmt_bytes(flash.blob_peak, b3), fmt_bytes(flash.blob_total, b4), flash.blob_allocs,
                flash.time_ms);

        // Ratio summary
        double ws_ratio   = gemm.ws_peak  > 0 ? (double)gemm.ws_peak  / flash.ws_peak  : 0;
        double blob_ratio = gemm.blob_peak > 0 ? (double)gemm.blob_peak / (flash.blob_peak ? flash.blob_peak : 1) : 0;
        fprintf(stdout, "       |    SAVING | WS %.1fx %s     | Blob %.1fx     |\n\n",
                ws_ratio, ws_ratio > 1 ? "smaller" : "larger ",
                blob_ratio);
    }

    return 0;
}
