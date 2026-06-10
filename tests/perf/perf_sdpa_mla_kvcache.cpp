// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

static bool should_run_case(int kv_cache, int past_seqlen, int threads)
{
    return perf_match_env_int("NCNN_PERF_SDPA_KVCACHE", kv_cache)
           && perf_match_env_int("NCNN_PERF_SDPA_PAST", past_seqlen)
           && perf_match_env_int("NCNN_PERF_THREADS", threads);
}

static void perf_sdpa_mla_decode(int kv_cache, int past_seqlen)
{
    const int d_k = perf_env_int("NCNN_PERF_SDPA_DK", 192, 1);
    const int d_v = perf_env_int("NCNN_PERF_SDPA_DV", 128, 1);
    const int num_heads = perf_env_int("NCNN_PERF_SDPA_HEADS", 128, 1);
    const int num_groups = perf_env_int("NCNN_PERF_SDPA_GROUPS", 16, 1);
    const int n_ctx_default = past_seqlen + 1 > 4096 ? past_seqlen + 1 : 4096;
    const int n_ctx = perf_env_int("NCNN_PERF_SDPA_CTX", n_ctx_default, past_seqlen + 1);
    const int threads = perf_env_int("NCNN_PERF_THREADS", 1, 1);

    if (!should_run_case(kv_cache, past_seqlen, threads))
        return;

    ncnn::ParamDict pd;
    pd.set(5, 0);   // attn_mask = 0
    pd.set(6, 0.f); // scale = 0 (default 1/sqrt(d_k))
    pd.set(7, kv_cache);

    std::vector<ncnn::Mat> weights(0);
    std::vector<ncnn::Mat> inputs;

    if (kv_cache == 1)
    {
        inputs.resize(5);
        inputs[0] = PerfMat(d_k, 1, num_heads); // q
        inputs[1] = PerfMat(d_k, 1, num_groups); // cur_k
        inputs[2] = PerfMat(d_v, 1, num_groups); // cur_v
        inputs[3] = PerfMat(d_k, past_seqlen, num_groups); // past_k
        inputs[4] = PerfMat(d_v, past_seqlen, num_groups); // past_v
    }
    else
    {
        inputs.resize(5);
        inputs[0] = PerfMat(d_k, 1, num_heads);  // q
        inputs[1] = PerfMat(d_k, 1, num_groups); // cur_k
        inputs[2] = PerfMat(d_v, 1, num_groups); // cur_v
        inputs[3] = PerfMat(d_k, n_ctx, num_groups); // preallocated past_k
        inputs[4] = PerfMat(d_v, n_ctx, num_groups); // preallocated past_v
        inputs[3].h = past_seqlen;
        inputs[4].h = past_seqlen;
    }

    perf_layer("SDPA", pd, weights, inputs, 3,
               "MLA kv_cache=%d d_k=%d d_v=%d h=%d g=%d past=%d ctx=%d t=%d",
               kv_cache, d_k, d_v, num_heads, num_groups, past_seqlen, n_ctx, threads);
}

int main()
{
    int pasts[] = {0, 128, 512, 1024, 2048, 4096};
    for (int i = 0; i < (int)(sizeof(pasts) / sizeof(pasts[0])); i++)
    {
        perf_sdpa_mla_decode(1, pasts[i]);
        perf_sdpa_mla_decode(2, pasts[i]);
    }

    return 0;
}
