// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

static void perf_sdpa_prefill(int head_dim, int value_dim, int num_heads, int num_kv_heads, int src_seqlen)
{
    ncnn::ParamDict pd;
    pd.set(5, 0);   // attn_mask = 0
    pd.set(6, 0.f); // scale = 0

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> inputs(3);
    inputs[0] = PerfMat(head_dim, src_seqlen, num_heads);
    inputs[1] = PerfMat(head_dim, src_seqlen, num_kv_heads);
    inputs[2] = PerfMat(value_dim, src_seqlen, num_kv_heads);

    perf_layer("SDPA", pd, weights, inputs, 1,
               "head_dim=%d value_dim=%d heads=%d kv_heads=%d seqlen=%d",
               head_dim, value_dim, num_heads, num_kv_heads, src_seqlen);
}

int main()
{
    perf_sdpa_prefill(64, 64, 8, 8, 16);
    perf_sdpa_prefill(64, 64, 8, 8, 64);
    perf_sdpa_prefill(64, 64, 8, 8, 128);
    perf_sdpa_prefill(64, 64, 8, 8, 256);
    perf_sdpa_prefill(128, 128, 8, 8, 1024);

    perf_sdpa_prefill(128, 128, 32, 32, 16);
    perf_sdpa_prefill(128, 128, 32, 32, 64);
    perf_sdpa_prefill(128, 128, 32, 32, 128);
    perf_sdpa_prefill(128, 128, 32, 32, 256);
    perf_sdpa_prefill(128, 128, 32, 32, 512);

    perf_sdpa_prefill(128, 128, 32, 8, 64);
    perf_sdpa_prefill(128, 128, 32, 8, 128);
    perf_sdpa_prefill(128, 128, 32, 8, 256);
    perf_sdpa_prefill(128, 128, 32, 8, 512);
    perf_sdpa_prefill(64, 64, 8, 2, 2048);

    perf_sdpa_prefill(128, 128, 32, 1, 64);
    perf_sdpa_prefill(128, 128, 32, 1, 256);

    perf_sdpa_prefill(80, 96, 14, 2, 128);
    perf_sdpa_prefill(96, 80, 16, 4, 512);
    perf_sdpa_prefill(192, 128, 16, 16, 128);
    perf_sdpa_prefill(256, 192, 8, 8, 64);

    return 0;
}
