// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

// decode phase: src_seqlen=1, with kv_cache and various past_seqlen
static void perf_sdpa_decode(int embed_dim, int num_heads, int num_groups, int past_seqlen)
{
    const int src_seqlen = 1;
    const int cur_seqlen = 1;
    const int out_embed_dim = embed_dim;
    const int dst_seqlen = past_seqlen + cur_seqlen;

    ncnn::ParamDict pd;
    pd.set(5, 0);   // attn_mask = 0
    pd.set(6, 0.f); // scale = 0 (default 1/sqrt(embed_dim))
    pd.set(7, 1);   // kv_cache = 1

    std::vector<ncnn::Mat> weights(0);

    // inputs: q, k, v, past_k, past_v
    std::vector<ncnn::Mat> inputs(5);
    inputs[0] = PerfMat(embed_dim, src_seqlen, num_heads);       // q
    inputs[1] = PerfMat(embed_dim, cur_seqlen, num_groups);      // cur_k
    inputs[2] = PerfMat(out_embed_dim, cur_seqlen, num_groups);  // cur_v
    inputs[3] = PerfMat(embed_dim, past_seqlen, num_groups);     // past_k
    inputs[4] = PerfMat(out_embed_dim, past_seqlen, num_groups); // past_v

    perf_layer("SDPA", pd, weights, inputs, 3,
               "embed=%d heads=%d groups=%d past=%d",
               embed_dim, num_heads, num_groups, past_seqlen);
}

int main()
{
    // typical LLM configurations for decode phase
    // format: (embed_dim, num_heads, num_groups, past_seqlen)

    // small model, various cache lengths
    perf_sdpa_decode(128, 4, 4, 0);
    perf_sdpa_decode(128, 4, 4, 128);
    perf_sdpa_decode(128, 4, 4, 512);
    perf_sdpa_decode(128, 4, 4, 1024);
    perf_sdpa_decode(128, 4, 4, 2048);

    // medium model
    perf_sdpa_decode(512, 8, 8, 0);
    perf_sdpa_decode(512, 8, 8, 128);
    perf_sdpa_decode(512, 8, 8, 512);
    perf_sdpa_decode(512, 8, 8, 1024);
    perf_sdpa_decode(512, 8, 8, 2048);

    // larger model (e.g., 7B scale)
    perf_sdpa_decode(4096, 32, 32, 0);
    perf_sdpa_decode(4096, 32, 32, 128);
    perf_sdpa_decode(4096, 32, 32, 512);
    perf_sdpa_decode(4096, 32, 32, 1024);
    perf_sdpa_decode(4096, 32, 32, 2048);
    perf_sdpa_decode(4096, 32, 32, 4096);
    perf_sdpa_decode(4096, 32, 32, 8192);

    // GQA/MQA configurations
    // GQA: num_groups < num_heads
    perf_sdpa_decode(4096, 32, 4, 128);
    perf_sdpa_decode(4096, 32, 4, 512);
    perf_sdpa_decode(4096, 32, 4, 1024);
    perf_sdpa_decode(4096, 32, 4, 2048);
    perf_sdpa_decode(4096, 32, 4, 4096);

    // MQA: num_groups = 1
    perf_sdpa_decode(4096, 32, 1, 128);
    perf_sdpa_decode(4096, 32, 1, 512);
    perf_sdpa_decode(4096, 32, 1, 1024);
    perf_sdpa_decode(4096, 32, 1, 2048);
    perf_sdpa_decode(4096, 32, 1, 4096);

    // very large context lengths
    perf_sdpa_decode(4096, 32, 32, 16384);
    perf_sdpa_decode(4096, 32, 32, 32768);
    perf_sdpa_decode(4096, 32, 4, 16384);
    perf_sdpa_decode(4096, 32, 4, 32768);

    return 0;
}
