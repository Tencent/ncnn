// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

static void perf_sdpa_decode(int head_dim, int value_dim, int num_heads, int num_kv_heads, int key_seqlen)
{
    const int src_seqlen = 1;

    ncnn::ParamDict pd;
    pd.set(5, 0);   // attn_mask = 0
    pd.set(6, 0.f); // scale = 0

    std::vector<ncnn::Mat> weights(0);

    std::vector<ncnn::Mat> inputs(3);
    inputs[0] = PerfMat(head_dim, src_seqlen, num_heads);
    inputs[1] = PerfMat(head_dim, key_seqlen, num_kv_heads);
    inputs[2] = PerfMat(value_dim, key_seqlen, num_kv_heads);

    perf_layer("SDPA", pd, weights, inputs, 1,
               "head_dim=%d value_dim=%d heads=%d kv_heads=%d seqlen=%d",
               head_dim, value_dim, num_heads, num_kv_heads, key_seqlen);
}

int main()
{
    perf_sdpa_decode(64, 64, 8, 8, 128);
    perf_sdpa_decode(64, 64, 8, 8, 512);
    perf_sdpa_decode(64, 64, 8, 8, 2048);

    perf_sdpa_decode(128, 128, 32, 32, 128);
    perf_sdpa_decode(128, 128, 32, 32, 512);
    perf_sdpa_decode(128, 128, 32, 32, 2048);
    perf_sdpa_decode(128, 128, 32, 32, 8192);

    perf_sdpa_decode(128, 128, 32, 8, 512);
    perf_sdpa_decode(128, 128, 32, 8, 2048);
    perf_sdpa_decode(128, 128, 32, 8, 8192);

    perf_sdpa_decode(128, 128, 32, 1, 512);
    perf_sdpa_decode(128, 128, 32, 1, 2048);
    perf_sdpa_decode(128, 128, 32, 1, 8192);

    perf_sdpa_decode(80, 96, 14, 2, 512);
    perf_sdpa_decode(96, 80, 16, 4, 2048);
    perf_sdpa_decode(192, 128, 16, 16, 512);
    perf_sdpa_decode(256, 192, 8, 8, 128);

    return 0;
}
