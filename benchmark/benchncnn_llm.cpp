// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "benchmark.h"
#include "cpu.h"
#include "datareader.h"
#include "gpu.h"
#include "layer.h"
#include "layer_type.h"
#include "net.h"

#include "benchncnn_llm_param_data.h"

#ifndef NCNN_SIMPLESTL
#include <algorithm>
#include <vector>
#endif

class DataReaderFromEmpty : public ncnn::DataReader
{
public:
    virtual int scan(const char* /*format*/, void* /*p*/) const
    {
        return 0;
    }

    virtual size_t read(void* buf, size_t size) const
    {
        memset(buf, 0, size);
        return size;
    }
};

namespace qwen35 {

static void l2norm(const float* x, float* out, int n, int dim, float eps)
{
    for (int i = 0; i < n; i++)
    {
        const float* row_in = x + i * dim;
        float* row_out = out + i * dim;

        float sum = 0.f;
        for (int j = 0; j < dim; j++)
        {
            sum += row_in[j] * row_in[j];
        }

        const float inv_norm = 1.f / sqrtf(sum + eps);
        for (int j = 0; j < dim; j++)
        {
            row_out[j] = row_in[j] * inv_norm;
        }
    }
}

static float sigmoidf(float x)
{
    return 1.f / (1.f + expf(-x));
}

static float softplusf(float x)
{
    if (x > 20.f)
        return x;
    if (x < -20.f)
        return expf(x);
    return logf(1.f + expf(x));
}

static void torch_recurrent_gated_delta_rule(const float* query, const float* key, const float* value, const float* g, const float* beta, float* core_attn_out, float* last_recurrent_state, int batch_size, int num_heads, int seq_len, int k_head_dim, int v_head_dim, bool use_qk_l2norm_in_kernel)
{
    std::vector<float> query_norm;
    std::vector<float> key_norm;

    const float* q_ptr = query;
    const float* k_ptr = key;

    const int qk_size = batch_size * num_heads * seq_len * k_head_dim;
    if (use_qk_l2norm_in_kernel)
    {
        query_norm.resize(qk_size);
        key_norm.resize(qk_size);

        l2norm(query, query_norm.data(), batch_size * num_heads * seq_len, k_head_dim, 1e-6f);
        l2norm(key, key_norm.data(), batch_size * num_heads * seq_len, k_head_dim, 1e-6f);

        q_ptr = query_norm.data();
        k_ptr = key_norm.data();
    }

    const float scale = 1.f / sqrtf((float)k_head_dim);

    memset(core_attn_out, 0, (size_t)batch_size * num_heads * seq_len * v_head_dim * sizeof(float));

    for (int t = 0; t < seq_len; t++)
    {
        for (int b = 0; b < batch_size; b++)
        {
            for (int h = 0; h < num_heads; h++)
            {
                const float* q_t = q_ptr + ((b * num_heads + h) * seq_len + t) * k_head_dim;
                const float* k_t = k_ptr + ((b * num_heads + h) * seq_len + t) * k_head_dim;
                const float* v_t = value + ((b * num_heads + h) * seq_len + t) * v_head_dim;

                const float g_t = g[(b * num_heads + h) * seq_len + t];
                const float beta_t = beta[(b * num_heads + h) * seq_len + t];

                float* state = last_recurrent_state + (b * num_heads + h) * k_head_dim * v_head_dim;
                float* out_t = core_attn_out + ((b * num_heads + h) * seq_len + t) * v_head_dim;

                const float g_t_exp = expf(g_t);

                for (int i = 0; i < k_head_dim * v_head_dim; i++)
                {
                    state[i] *= g_t_exp;
                }

                std::vector<float> kv_mem(v_head_dim, 0.f);
                for (int dv = 0; dv < v_head_dim; dv++)
                {
                    for (int dk = 0; dk < k_head_dim; dk++)
                    {
                        kv_mem[dv] += state[dk * v_head_dim + dv] * k_t[dk];
                    }
                }

                std::vector<float> delta(v_head_dim);
                for (int dv = 0; dv < v_head_dim; dv++)
                {
                    delta[dv] = (v_t[dv] - kv_mem[dv]) * beta_t;
                }

                for (int dk = 0; dk < k_head_dim; dk++)
                {
                    for (int dv = 0; dv < v_head_dim; dv++)
                    {
                        state[dk * v_head_dim + dv] += k_t[dk] * delta[dv];
                    }
                }

                for (int dv = 0; dv < v_head_dim; dv++)
                {
                    float sum = 0.f;
                    for (int dk = 0; dk < k_head_dim; dk++)
                    {
                        sum += state[dk * v_head_dim + dv] * q_t[dk] * scale;
                    }
                    out_t[dv] = sum;
                }
            }
        }
    }
}

class GatedDeltaRule : public ncnn::Layer
{
public:
    GatedDeltaRule()
    {
        one_blob_only = false;
        support_inplace = false;

        num_k_heads = 128;
        num_v_heads = 128;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& a_log = bottom_blobs[0];
        const ncnn::Mat& dt_bias = bottom_blobs[1];
        const ncnn::Mat& b = bottom_blobs[2];
        const ncnn::Mat& a = bottom_blobs[3];
        const ncnn::Mat& query = bottom_blobs[4];
        const ncnn::Mat& key = bottom_blobs[5];
        const ncnn::Mat& value = bottom_blobs[6];
        const ncnn::Mat& initial_state = bottom_blobs[7];

        int num_heads = query.h;
        int seq_len = query.c;
        int k_head_dim = query.w;
        int v_head_dim = value.w;

        const bool use_qk_l2norm_in_kernel = true;

        ncnn::Mat& top_blob = top_blobs[0];
        top_blob.create(k_head_dim, num_heads, seq_len, 4u, opt.blob_allocator);

        ncnn::Mat& state_out = top_blobs[1];
        state_out.create(v_head_dim, k_head_dim, num_heads, 4u, opt.blob_allocator);

        if (top_blob.empty() || state_out.empty())
            return -100;

        const float* query_data = (const float*)query.data;
        const float* key_data = (const float*)key.data;
        const float* value_data = (const float*)value.data;

        std::vector<float> query_t(num_heads * seq_len * k_head_dim);
        std::vector<float> key_t(num_heads * seq_len * k_head_dim);
        std::vector<float> value_t(num_heads * seq_len * v_head_dim);

        for (int t = 0; t < seq_len; t++)
        {
            for (int h = 0; h < num_heads; h++)
            {
                for (int d = 0; d < k_head_dim; d++)
                {
                    int src_idx = (t * num_heads + h) * k_head_dim + d;
                    int dst_idx = (h * seq_len + t) * k_head_dim + d;
                    query_t[dst_idx] = query_data[src_idx];
                    key_t[dst_idx] = key_data[src_idx];
                }

                for (int d = 0; d < v_head_dim; d++)
                {
                    int src_idx = (t * num_heads + h) * v_head_dim + d;
                    int dst_idx = (h * seq_len + t) * v_head_dim + d;
                    value_t[dst_idx] = value_data[src_idx];
                }
            }
        }

        const float* b_data = (const float*)b.data;
        const float* a_data = (const float*)a.data;
        const float* a_log_data = (const float*)a_log.data;
        const float* dt_bias_data = (const float*)dt_bias.data;

        std::vector<float> beta(num_heads * seq_len);
        std::vector<float> g(num_heads * seq_len);

        for (int h = 0; h < num_heads; h++)
        {
            for (int t = 0; t < seq_len; t++)
            {
                const float b_val = b_data[t * num_heads + h];
                beta[h * seq_len + t] = sigmoidf(b_val);
            }
        }

        for (int h = 0; h < num_heads; h++)
        {
            const float a_log_val = a_log_data[h];
            const float dt_bias_val = dt_bias_data[h];
            const float exp_a = expf(a_log_val);

            for (int t = 0; t < seq_len; t++)
            {
                const float a_val = a_data[t * num_heads + h];
                const float sp_val = softplusf(a_val + dt_bias_val);
                g[h * seq_len + t] = -exp_a * sp_val;
            }
        }

        const int batch_size = 1;

        std::vector<float> core_attn_out(num_heads * seq_len * v_head_dim);
        std::vector<float> last_recurrent_state(num_heads * k_head_dim * v_head_dim);

        if (!initial_state.empty())
            memcpy(last_recurrent_state.data(), initial_state.data, (size_t)num_heads * k_head_dim * v_head_dim * sizeof(float));
        else
            memset(last_recurrent_state.data(), 0, (size_t)num_heads * k_head_dim * v_head_dim * sizeof(float));

        torch_recurrent_gated_delta_rule(query_t.data(), key_t.data(), value_t.data(), g.data(), beta.data(), core_attn_out.data(), last_recurrent_state.data(), batch_size, num_heads, seq_len, k_head_dim, v_head_dim, use_qk_l2norm_in_kernel);

        float* top_data = (float*)top_blob.data;
        for (int h = 0; h < num_heads; h++)
        {
            for (int t = 0; t < seq_len; t++)
            {
                for (int d = 0; d < v_head_dim; d++)
                {
                    int src_idx = (h * seq_len + t) * v_head_dim + d;
                    int dst_idx = (t * num_heads + h) * v_head_dim + d;
                    top_data[dst_idx] = core_attn_out[src_idx];
                }
            }
        }

        memcpy((float*)state_out.data, last_recurrent_state.data(), (size_t)num_heads * k_head_dim * v_head_dim * sizeof(float));

        return 0;
    }

public:
    int num_k_heads;
    int num_v_heads;
};

class ShortConv : public ncnn::Layer
{
public:
    ShortConv()
    {
        one_blob_only = false;
        support_inplace = false;
    }

    virtual int forward(const std::vector<ncnn::Mat>& bottom_blobs, std::vector<ncnn::Mat>& top_blobs, const ncnn::Option& opt) const
    {
        const ncnn::Mat& weight_mat = bottom_blobs[0];
        const ncnn::Mat& mixed_qkv = bottom_blobs[1];
        const ncnn::Mat& conv_state = bottom_blobs[2];

        const int seq_len = mixed_qkv.h;
        const int groups = mixed_qkv.w;
        const int kernel_size = weight_mat.w;

        ncnn::Mat stated_mixed_qkv;
        if (conv_state.empty())
        {
            stated_mixed_qkv.create(groups, kernel_size - 1 + seq_len, 4u, opt.blob_allocator);
            if (stated_mixed_qkv.empty())
                return -100;

            memset(stated_mixed_qkv.row(0), 0, (size_t)(kernel_size - 1) * groups * sizeof(float));
            memcpy(stated_mixed_qkv.row(kernel_size - 1), mixed_qkv, (size_t)mixed_qkv.h * groups * sizeof(float));
        }
        else
        {
            stated_mixed_qkv.create(groups, conv_state.h + seq_len, 4u, opt.blob_allocator);
            if (stated_mixed_qkv.empty())
                return -100;

            memcpy(stated_mixed_qkv.row(0), conv_state, (size_t)conv_state.h * groups * sizeof(float));
            memcpy(stated_mixed_qkv.row(conv_state.h), mixed_qkv, (size_t)mixed_qkv.h * groups * sizeof(float));
        }

        const int state_len = kernel_size;
        const int total_len = conv_state.empty() ? (kernel_size - 1 + seq_len) : (conv_state.h + seq_len);

        ncnn::Mat last_conv_state(groups, state_len, 4u, opt.blob_allocator);
        if (last_conv_state.empty())
            return -100;

        memcpy(last_conv_state, stated_mixed_qkv.row(total_len - state_len), (size_t)state_len * groups * sizeof(float));

        ncnn::Mat& top_blob = top_blobs[0];
        top_blob.create(groups, seq_len, 4u, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int g = 0; g < groups; g++)
        {
            const float* w_ptr = weight_mat.channel(g);

            for (int i = 0; i < seq_len; i++)
            {
                float sum = 0.f;

                const int prefix_len = conv_state.empty() ? (kernel_size - 1) : conv_state.h;
                const int base = prefix_len + i;

                for (int k = 0; k < kernel_size; k++)
                {
                    const int src_i = base - (kernel_size - 1) + k;
                    sum += stated_mixed_qkv.row(src_i)[g] * w_ptr[k];
                }

                top_blob.row(i)[g] = sum * (1.f / (1.f + expf(-sum)));
            }
        }

        top_blobs[1] = last_conv_state;

        return 0;
    }
};

static ncnn::Layer* GatedDeltaRule_creator(void*)
{
    return new GatedDeltaRule;
}

static void GatedDeltaRule_destroyer(ncnn::Layer* layer, void*)
{
    delete layer;
}

static ncnn::Layer* ShortConv_creator(void*)
{
    return new ShortConv;
}

static void ShortConv_destroyer(ncnn::Layer* layer, void*)
{
    delete layer;
}

static void register_custom_layers(ncnn::Net& net)
{
    net.register_custom_layer("GatedDeltaRule", GatedDeltaRule_creator, GatedDeltaRule_destroyer);
    net.register_custom_layer("ShortConv", ShortConv_creator, ShortConv_destroyer);
}

} // namespace qwen35

static int g_warmup_loop_count = 8;
static int g_loop_count = 4;
static bool g_enable_cooling_down = true;

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

#if NCNN_VULKAN
static ncnn::VulkanDevice* g_vkdev = 0;
static ncnn::VkAllocator* g_blob_vkallocator = 0;
static ncnn::VkAllocator* g_staging_vkallocator = 0;
#endif // NCNN_VULKAN

struct CacheIndexes
{
    std::vector<int> input_indexes;
    std::vector<int> output_indexes;
};

struct ModelConfig
{
    const char* name;
    const char* param_data;
    int hidden_size;
    int rope_half_dim;
    void (*register_custom_layers)(ncnn::Net&);
};

namespace minicpm4 {

static const ModelConfig model = {"minicpm4_0.5b", minicpm4_0_5b_decoder_ncnn_param_data, 1024, 32, 0};

} // namespace minicpm4

namespace qwen35 {

static const ModelConfig model = {"qwen3.5_0.8b", qwen3_5_0_8b_decoder_ncnn_param_data, 1024, 32, register_custom_layers};

} // namespace qwen35

static void resolve_cache_indexes(const ncnn::Net& net, CacheIndexes& cache_indexes)
{
    const std::vector<ncnn::Layer*>& layers = net.layers();
    for (size_t i = 0; i < layers.size(); i++)
    {
        const ncnn::Layer* op = layers[i];

        if (op->typeindex == ncnn::LayerType::SDPA && op->tops.size() == 3)
        {
            cache_indexes.input_indexes.push_back(op->bottoms[op->bottoms.size() - 2]);
            cache_indexes.input_indexes.push_back(op->bottoms[op->bottoms.size() - 1]);
            cache_indexes.output_indexes.push_back(op->tops[op->tops.size() - 2]);
            cache_indexes.output_indexes.push_back(op->tops[op->tops.size() - 1]);
        }

        if ((op->type == "ShortConv" || op->type == "GatedDeltaRule") && op->tops.size() == 2)
        {
            cache_indexes.input_indexes.push_back(op->bottoms[op->bottoms.size() - 1]);
            cache_indexes.output_indexes.push_back(op->tops[op->tops.size() - 1]);
        }
    }
}

static void make_attention_mask(int cur_seqlen, int past_seqlen, ncnn::Mat& attention_mask)
{
    const int dst_seqlen = past_seqlen + cur_seqlen;

    attention_mask.create(dst_seqlen, cur_seqlen);
    attention_mask.fill(0.f);

    for (int i = 0; i < cur_seqlen; i++)
    {
        float* row = attention_mask.row(i);
        for (int j = past_seqlen + i + 1; j < dst_seqlen; j++)
        {
            row[j] = -INFINITY;
        }
    }
}

static void make_rope_cache(int half_dim, int seqlen, ncnn::Mat& cos_cache, ncnn::Mat& sin_cache)
{
    cos_cache.create(half_dim, seqlen);
    sin_cache.create(half_dim, seqlen);

    cos_cache.fill(1.f);
    sin_cache.fill(0.f);
}

static int run_decoder_once(ncnn::Net& net, const CacheIndexes& cache_indexes, int hidden_size, int rope_half_dim, int cur_seqlen, int past_seqlen, const std::vector<ncnn::Mat>& cache, std::vector<ncnn::Mat>& out_cache)
{
    ncnn::Mat token_embeds(hidden_size, cur_seqlen);
    token_embeds.fill(0.01f);

    ncnn::Mat attention_mask;
    make_attention_mask(cur_seqlen, past_seqlen, attention_mask);

    ncnn::Mat cos_cache;
    ncnn::Mat sin_cache;
    make_rope_cache(rope_half_dim, cur_seqlen, cos_cache, sin_cache);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", token_embeds);
    ex.input("in1", attention_mask);
    ex.input("in2", cos_cache);
    ex.input("in3", sin_cache);

    for (size_t i = 0; i < cache.size(); i++)
    {
        ex.input(cache_indexes.input_indexes[i], cache[i]);
    }

    out_cache.resize(cache_indexes.output_indexes.size());
    for (size_t i = 0; i < cache_indexes.output_indexes.size(); i++)
    {
        ex.extract(cache_indexes.output_indexes[i], out_cache[i], 1);
    }

    ncnn::Mat out;
    return ex.extract("out0", out);
}

static void benchmark_case(const char* name, ncnn::Net& net, const CacheIndexes& cache_indexes, int hidden_size, int rope_half_dim, int cur_seqlen, int past_seqlen, const std::vector<ncnn::Mat>& cache, double rate_scale)
{
    std::vector<ncnn::Mat> out_cache;

    for (int i = 0; i < g_warmup_loop_count; i++)
    {
        run_decoder_once(net, cache_indexes, hidden_size, rope_half_dim, cur_seqlen, past_seqlen, cache, out_cache);
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    for (int i = 0; i < g_loop_count; i++)
    {
        double start = ncnn::get_current_time();
        run_decoder_once(net, cache_indexes, hidden_size, rope_half_dim, cur_seqlen, past_seqlen, cache, out_cache);
        double end = ncnn::get_current_time();

        double time = end - start;
        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;
    }

    time_avg /= g_loop_count;

    const double tokens_per_second = rate_scale * 1000.0 / time_avg;
    fprintf(stderr, "%30s  min = %7.2f  max = %7.2f  avg = %7.2f  tokens/s = %7.2f\n", name, time_min, time_max, time_avg, tokens_per_second);
}

static int benchmark_model(const ModelConfig& config, const ncnn::Option& opt)
{
    g_blob_pool_allocator.clear();
    g_workspace_pool_allocator.clear();

#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        g_blob_vkallocator->clear();
        g_staging_vkallocator->clear();
    }
#endif // NCNN_VULKAN

    ncnn::Net net;
    net.opt = opt;

#if NCNN_VULKAN
    if (net.opt.use_vulkan_compute)
    {
        net.set_vulkan_device(g_vkdev);
    }
#endif // NCNN_VULKAN

    if (config.register_custom_layers)
    {
        config.register_custom_layers(net);
    }

    net.load_param_mem(config.param_data);

    DataReaderFromEmpty dr;
    net.load_model(dr);

    CacheIndexes cache_indexes;
    resolve_cache_indexes(net, cache_indexes);

    if (g_enable_cooling_down)
    {
        ncnn::sleep(10 * 1000);
    }

    std::vector<ncnn::Mat> empty_cache;
    std::vector<ncnn::Mat> past_cache;
    run_decoder_once(net, cache_indexes, config.hidden_size, config.rope_half_dim, 1024, 0, empty_cache, past_cache);

    char prefill_name[256];
    snprintf(prefill_name, 256, "%s_1k_prefill", config.name);

    benchmark_case(prefill_name, net, cache_indexes, config.hidden_size, config.rope_half_dim, 1024, 0, empty_cache, 1024.0);

    char decode_name[256];
    snprintf(decode_name, 256, "%s_1k_decode", config.name);

    benchmark_case(decode_name, net, cache_indexes, config.hidden_size, config.rope_half_dim, 1, 1024, past_cache, 1.0);

    return 0;
}

static void show_usage()
{
    fprintf(stderr, "Usage: benchncnn_llm [loop count] [num threads] [powersave] [gpu device] [cooling down]\n");
}

int main(int argc, char** argv)
{
    int loop_count = 4;
    int num_threads = ncnn::get_physical_big_cpu_count();
    int powersave = 2;
    int gpu_device = -1;
    int cooling_down = 1;

    for (int i = 1; i < argc; i++)
    {
        if (argv[i][0] == '-' && argv[i][1] == 'h')
        {
            show_usage();
            return -1;
        }

        if (strcmp(argv[i], "--help") == 0)
        {
            show_usage();
            return -1;
        }
    }

    if (argc >= 2)
    {
        loop_count = atoi(argv[1]);
    }
    if (argc >= 3)
    {
        num_threads = atoi(argv[2]);
    }
    if (argc >= 4)
    {
        powersave = atoi(argv[3]);
    }
    if (argc >= 5)
    {
        gpu_device = atoi(argv[4]);
    }
    if (argc >= 6)
    {
        cooling_down = atoi(argv[5]);
    }

    const bool use_vulkan_compute = gpu_device != -1;

    g_enable_cooling_down = cooling_down != 0;
    g_loop_count = loop_count;

    g_blob_pool_allocator.set_size_compare_ratio(0.f);
    g_workspace_pool_allocator.set_size_compare_ratio(0.f);

#if NCNN_VULKAN
    if (use_vulkan_compute)
    {
        g_warmup_loop_count = 10;

        g_vkdev = ncnn::get_gpu_device(gpu_device);

        g_blob_vkallocator = new ncnn::VkBlobAllocator(g_vkdev);
        g_staging_vkallocator = new ncnn::VkStagingAllocator(g_vkdev);
    }
#endif // NCNN_VULKAN

    ncnn::set_cpu_powersave(powersave);

    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(num_threads);

    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = num_threads;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
#if NCNN_VULKAN
    opt.blob_vkallocator = g_blob_vkallocator;
    opt.workspace_vkallocator = g_blob_vkallocator;
    opt.staging_vkallocator = g_staging_vkallocator;
#endif // NCNN_VULKAN
    opt.use_winograd_convolution = true;
    opt.use_sgemm_convolution = true;
    opt.use_int8_inference = true;
    opt.use_vulkan_compute = use_vulkan_compute;
    opt.use_bf16_packed = true;
    opt.use_bf16_storage = true;
    opt.use_fp16_packed = true;
    opt.use_fp16_storage = true;
    opt.use_fp16_arithmetic = true;
    opt.use_int8_storage = true;
    opt.use_int8_arithmetic = true;
    opt.use_packing_layout = true;

    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    fprintf(stderr, "gpu_device = %d\n", gpu_device);
    fprintf(stderr, "cooling_down = %d\n", (int)g_enable_cooling_down);

    const ModelConfig* models[] = {
        &minicpm4::model,
        &qwen35::model,
    };

    for (size_t i = 0; i < sizeof(models) / sizeof(models[0]); i++)
    {
        benchmark_model(*models[i], opt);
    }

#if NCNN_VULKAN
    delete g_blob_vkallocator;
    delete g_staging_vkallocator;
#endif // NCNN_VULKAN

    return 0;
}
