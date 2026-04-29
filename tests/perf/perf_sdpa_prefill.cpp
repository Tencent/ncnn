// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

// prefill phase: larger src_seqlen, no kv_cache (past_seqlen=0)
static void perf_sdpa_prefill(int embed_dim, int num_heads, int num_groups, int src_seqlen)
{
    const int cur_seqlen = src_seqlen; // in prefill, cur_seqlen == src_seqlen
    const int out_embed_dim = embed_dim;

    ncnn::ParamDict pd;
    pd.set(5, 0);   // attn_mask = 0
    pd.set(6, 0.f); // scale = 0 (default 1/sqrt(embed_dim))
    pd.set(7, 0);   // kv_cache = 0 (no cache in prefill)

    std::vector<ncnn::Mat> weights(0);

    // inputs: q, k, v
    std::vector<ncnn::Mat> inputs(3);
    inputs[0] = PerfMat(embed_dim, src_seqlen, num_heads);      // q
    inputs[1] = PerfMat(embed_dim, cur_seqlen, num_groups);     // k
    inputs[2] = PerfMat(out_embed_dim, cur_seqlen, num_groups); // v

    perf_layer("SDPA", pd, weights, inputs, 1,
               "embed=%d heads=%d groups=%d seqlen=%d",
               embed_dim, num_heads, num_groups, src_seqlen);

    // int8 variant
    ncnn::ParamDict pd_int8;
    pd_int8.set(5, 0);    // attn_mask = 0
    pd_int8.set(6, 0.f);  // scale = 0
    pd_int8.set(7, 0);    // kv_cache = 0
    pd_int8.set(18, 2);   // int8_scale_term
    perf_layer_int8("SDPA", pd_int8, weights, inputs, 1,
                    "embed=%d heads=%d groups=%d seqlen=%d",
                    embed_dim, num_heads, num_groups, src_seqlen);
}

int main()
{
    // typical LLM configurations for prefill phase
    // format: (embed_dim, num_heads, num_groups, src_seqlen)

    // small model, various sequence lengths
    perf_sdpa_prefill(128, 4, 4, 16);
    perf_sdpa_prefill(128, 4, 4, 32);
    perf_sdpa_prefill(128, 4, 4, 64);
    perf_sdpa_prefill(128, 4, 4, 128);
    perf_sdpa_prefill(128, 4, 4, 256);
    perf_sdpa_prefill(128, 4, 4, 512);

    // medium model
    perf_sdpa_prefill(512, 8, 8, 16);
    perf_sdpa_prefill(512, 8, 8, 32);
    perf_sdpa_prefill(512, 8, 8, 64);
    perf_sdpa_prefill(512, 8, 8, 128);
    perf_sdpa_prefill(512, 8, 8, 256);
    perf_sdpa_prefill(512, 8, 8, 512);
    perf_sdpa_prefill(512, 8, 8, 1024);

    // larger model (e.g., 7B scale)
    perf_sdpa_prefill(4096, 32, 32, 16);
    perf_sdpa_prefill(4096, 32, 32, 32);
    perf_sdpa_prefill(4096, 32, 32, 64);

    // GQA/MQA configurations
    // GQA: num_groups < num_heads
    perf_sdpa_prefill(4096, 32, 4, 16);
    perf_sdpa_prefill(4096, 32, 4, 32);
    perf_sdpa_prefill(4096, 32, 4, 64);

    // MQA: num_groups = 1
    perf_sdpa_prefill(4096, 32, 1, 16);
    perf_sdpa_prefill(4096, 32, 1, 32);
    perf_sdpa_prefill(4096, 32, 1, 64);

    return 0;
}
