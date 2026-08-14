// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "perfutil.h"

#include <stdlib.h>
#include <string.h>

static bool match_env_int(const char* name, int value)
{
    const char* s = getenv(name);
    if (!s || !s[0])
        return true;

    return atoi(s) == value;
}

static bool has_env_int(const char* name)
{
    const char* s = getenv(name);
    return s && s[0];
}

static bool match_env_string(const char* name, const char* value)
{
    const char* s = getenv(name);
    if (!s || !s[0])
        return true;

    return strcmp(s, value) == 0;
}

static bool should_run_fp_prefill()
{
    return match_env_string("NCNN_PERF_DTYPE", "fp32")
           || match_env_string("NCNN_PERF_DTYPE", "fp16ps")
           || match_env_string("NCNN_PERF_DTYPE", "fp16psa")
           || match_env_string("NCNN_PERF_DTYPE", "bf16ps");
}

static bool should_run_int8_prefill()
{
    return match_env_string("NCNN_PERF_DTYPE", "int8");
}

static bool should_run_prefill(int embed_dim, int num_heads, int num_groups, int src_seqlen)
{
    return match_env_int("NCNN_PERF_SDPA_EMBED", embed_dim)
           && match_env_int("NCNN_PERF_SDPA_HEADS", num_heads)
           && match_env_int("NCNN_PERF_SDPA_GROUPS", num_groups)
           && match_env_int("NCNN_PERF_SDPA_SEQLEN", src_seqlen);
}

static bool should_run_extended_prefill()
{
    return has_env_int("NCNN_PERF_SDPA_EMBED")
           && has_env_int("NCNN_PERF_SDPA_HEADS")
           && has_env_int("NCNN_PERF_SDPA_GROUPS")
           && has_env_int("NCNN_PERF_SDPA_SEQLEN");
}

// prefill phase: larger src_seqlen, no kv_cache (past_seqlen=0)
static void perf_sdpa_prefill(int embed_dim, int num_heads, int num_groups, int src_seqlen)
{
    if (!should_run_prefill(embed_dim, num_heads, num_groups, src_seqlen))
        return;

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

    if (should_run_fp_prefill())
    {
        perf_layer("SDPA", pd, weights, inputs, 1,
                   "embed=%d heads=%d groups=%d seqlen=%d",
                   embed_dim, num_heads, num_groups, src_seqlen);
    }

    // int8 variant
    ncnn::ParamDict pd_int8;
    pd_int8.set(5, 0);   // attn_mask = 0
    pd_int8.set(6, 0.f); // scale = 0
    pd_int8.set(7, 0);   // kv_cache = 0
    pd_int8.set(18, 2);  // int8_scale_term
    if (should_run_int8_prefill())
    {
        perf_layer_int8("SDPA", pd_int8, weights, inputs, 1,
                        "embed=%d heads=%d groups=%d seqlen=%d",
                        embed_dim, num_heads, num_groups, src_seqlen);
    }
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
    perf_sdpa_prefill(4096, 32, 16, 16);
    perf_sdpa_prefill(4096, 32, 16, 32);
    perf_sdpa_prefill(4096, 32, 16, 64);
    perf_sdpa_prefill(4096, 32, 8, 16);
    perf_sdpa_prefill(4096, 32, 8, 32);
    perf_sdpa_prefill(4096, 32, 8, 64);
    perf_sdpa_prefill(4096, 32, 4, 16);
    perf_sdpa_prefill(4096, 32, 4, 32);
    perf_sdpa_prefill(4096, 32, 4, 64);

    // MQA: num_groups = 1
    perf_sdpa_prefill(4096, 32, 1, 16);
    perf_sdpa_prefill(4096, 32, 1, 32);
    perf_sdpa_prefill(4096, 32, 1, 64);

    // Longer large-model cases are opt-in through NCNN_PERF_SDPA_* filters.
    if (should_run_extended_prefill())
    {
        perf_sdpa_prefill(4096, 32, 32, 128);
        perf_sdpa_prefill(4096, 32, 32, 256);
        perf_sdpa_prefill(4096, 32, 16, 128);
        perf_sdpa_prefill(4096, 32, 16, 256);
        perf_sdpa_prefill(4096, 32, 8, 128);
        perf_sdpa_prefill(4096, 32, 8, 256);
        perf_sdpa_prefill(4096, 32, 4, 128);
        perf_sdpa_prefill(4096, 32, 4, 256);
        perf_sdpa_prefill(4096, 32, 1, 128);
        perf_sdpa_prefill(4096, 32, 1, 256);
    }

    return 0;
}
