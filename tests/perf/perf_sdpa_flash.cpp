// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

// Micro-benchmark: flash attention (kv_cache=2) vs Gemm baseline (kv_cache=0)
// at the exact dimensions used by Youtu-LLM-2B (MLA, d_k=192, d_v=128).
//
// Both paths receive identical input shapes - the only difference is which
// internal code path SDPA_arm takes. Run side-by-side to see the algorithmic
// improvement of FlashAttention-2 vs ncnn's Gemm-based attention.

#include "perfutil.h"

static ncnn::Mat PerfCausalMask(int src_seqlen, int past_seqlen)
{
    const int total = past_seqlen + src_seqlen;
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

// Shared dimensions: matching Youtu-LLM-2B
//   d_k = 192   (MLA query/key head dim)
//   d_v = 128   (MLA value head dim)
//   heads = 128
//   groups = 16 (GQA 8:1)

// kv_cache=0: pure prefill via ncnn Gemm path
static void perf_gemm_prefill(int d_k, int d_v, int heads, int groups, int src_seqlen)
{
    if (!perf_match_env_int("NCNN_PERF_SDPA_M", src_seqlen))
        return;

    const bool causal = perf_env_int("NCNN_PERF_SDPA_CAUSAL", 0, 0) != 0;

    ncnn::ParamDict pd;
    pd.set(5, causal ? 1 : 0); // attn_mask
    pd.set(6, 0.f); // scale = 0 (default 1/sqrt(d_k))
    pd.set(7, 0);   // kv_cache = 0

    std::vector<ncnn::Mat> weights(0);
    std::vector<ncnn::Mat> inputs;
    inputs.resize(causal ? 4 : 3);
    inputs[0] = PerfMat(d_k, src_seqlen, heads);  // q
    inputs[1] = PerfMat(d_k, src_seqlen, groups); // k
    inputs[2] = PerfMat(d_v, src_seqlen, groups); // v
    if (causal)
        inputs[3] = PerfCausalMask(src_seqlen, 0);

    perf_layer("SDPA", pd, weights, inputs, 1,
               "GEMM   d_k=%d d_v=%d h=%d g=%d M=%d causal=%d",
               d_k, d_v, heads, groups, src_seqlen, causal ? 1 : 0);
}

// kv_cache=2 prefill: in-place append (past_seqlen=0) + flash prefill
static void perf_flash_prefill(int d_k, int d_v, int heads, int groups, int src_seqlen, int n_ctx)
{
    if (!perf_match_env_int("NCNN_PERF_SDPA_M", src_seqlen))
        return;

    const bool causal = perf_env_int("NCNN_PERF_SDPA_CAUSAL", 0, 0) != 0;

    ncnn::ParamDict pd;
    pd.set(5, causal ? 1 : 0); // attn_mask
    pd.set(6, 0.f); // scale
    pd.set(7, 2);   // kv_cache = 2 (in-place append + flash)

    std::vector<ncnn::Mat> weights(0);
    std::vector<ncnn::Mat> inputs;
    inputs.resize(causal ? 6 : 5);
    inputs[0] = PerfMat(d_k, src_seqlen, heads);  // q
    inputs[1] = PerfMat(d_k, src_seqlen, groups); // cur_k
    inputs[2] = PerfMat(d_v, src_seqlen, groups); // cur_v
    int offset = 3;
    if (causal)
        inputs[offset++] = PerfCausalMask(src_seqlen, 0);
    ncnn::Mat past_key = PerfMat(d_k, n_ctx, groups);
    ncnn::Mat past_value = PerfMat(d_v, n_ctx, groups);
    past_key.h = 0;
    past_value.h = 0;
    inputs[offset++] = past_key;   // past_k view (capacity in cstep)
    inputs[offset++] = past_value; // past_v view (capacity in cstep)

    perf_layer("SDPA", pd, weights, inputs, 3,
               "FLASH  d_k=%d d_v=%d h=%d g=%d M=%d ctx=%d causal=%d",
               d_k, d_v, heads, groups, src_seqlen, n_ctx, causal ? 1 : 0);
}

// kv_cache=2 decode: in-place append (past_seqlen>0) + flash decode
static void perf_flash_decode(int d_k, int d_v, int heads, int groups, int past_seqlen, int n_ctx)
{
    if (!perf_match_env_int("NCNN_PERF_SDPA_PAST", past_seqlen))
        return;

    ncnn::ParamDict pd;
    pd.set(5, 0);
    pd.set(6, 0.f);
    pd.set(7, 2);

    std::vector<ncnn::Mat> weights(0);
    std::vector<ncnn::Mat> inputs(5);
    inputs[0] = PerfMat(d_k, 1, heads);      // q (1 token)
    inputs[1] = PerfMat(d_k, 1, groups);     // cur_k (1 token)
    inputs[2] = PerfMat(d_v, 1, groups);     // cur_v (1 token)
    inputs[3] = PerfMat(d_k, n_ctx, groups); // past_k capacity
    inputs[4] = PerfMat(d_v, n_ctx, groups); // past_v capacity
    inputs[3].h = past_seqlen;
    inputs[4].h = past_seqlen;

    perf_layer("SDPA", pd, weights, inputs, 3,
               "FLASH  d_k=%d d_v=%d h=%d g=%d past=%d ctx=%d",
               d_k, d_v, heads, groups, past_seqlen, n_ctx);
}

int main()
{
    const int d_k = 192;
    const int d_v = 128;
    const int heads = 128;
    const int groups = 16;
    const int n_ctx = 4096;

    const bool run_prefill = !perf_has_env("NCNN_PERF_SDPA_DECODE_ONLY");
    const bool run_decode = !perf_has_env("NCNN_PERF_SDPA_PREFILL_ONLY");

    if (run_prefill)
    {
        fprintf(stdout, "=== Prefill: Gemm vs Flash (Youtu-LLM-2B dims) ===\n\n");

        int seqlens[] = {32, 64, 128, 256, 512, 1024};
        for (int i = 0; i < (int)(sizeof(seqlens) / sizeof(seqlens[0])); i++)
        {
            int M = seqlens[i];
            perf_gemm_prefill(d_k, d_v, heads, groups, M);
            perf_flash_prefill(d_k, d_v, heads, groups, M, n_ctx);
            fprintf(stdout, "\n");
        }
    }

    if (run_decode)
    {
        fprintf(stdout, "=== Decode: Flash with varying past_seqlen ===\n\n");

        int pasts[] = {32, 128, 256, 512, 1024, 2048};
        for (int i = 0; i < (int)(sizeof(pasts) / sizeof(pasts[0])); i++)
        {
            perf_flash_decode(d_k, d_v, heads, groups, pasts[i], n_ctx);
        }
    }

    return 0;
}
