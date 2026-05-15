// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include <float.h>
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
    const char* decoder_param_data;
    const char* proj_out_param_data;
    int hidden_size;
    int rope_half_dim;
};

namespace minicpm4 {

static const ModelConfig model = {"minicpm4_0.5b", minicpm4_0_5b_decoder_ncnn_param_data, minicpm4_0_5b_proj_out_ncnn_param_data, 1024, 32};

} // namespace minicpm4

namespace qwen25 {

static const ModelConfig model = {"qwen2.5_0.5b", qwen2_5_0_5b_decoder_ncnn_param_data, qwen2_5_0_5b_proj_out_ncnn_param_data, 896, 32};

} // namespace qwen25

namespace qwen3 {

static const ModelConfig model = {"qwen3_0.6b", qwen3_0_6b_decoder_ncnn_param_data, qwen3_0_6b_proj_out_ncnn_param_data, 1024, 64};

} // namespace qwen3

namespace hunyuan {

static const ModelConfig model = {"hunyuan_0.5b", hunyuan_0_5b_instruct_decoder_ncnn_param_data, hunyuan_0_5b_instruct_proj_out_ncnn_param_data, 1024, 64};

} // namespace hunyuan

namespace tinyllama {

static const ModelConfig model = {"tinyllama_1.1b", tinyllama_1_1b_decoder_ncnn_param_data, tinyllama_1_1b_proj_out_ncnn_param_data, 2048, 32};

} // namespace tinyllama

namespace llama32 {

static const ModelConfig model = {"llama3.2_1b", llama3_2_1b_decoder_ncnn_param_data, llama3_2_1b_proj_out_ncnn_param_data, 2048, 32};

} // namespace llama32

namespace youtu_llm {

static const ModelConfig model = {"youtu_llm_2b", youtu_llm_2b_decoder_ncnn_param_data, youtu_llm_2b_proj_out_ncnn_param_data, 2048, 64};

} // namespace youtu_llm

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

static int run_decoder_once(ncnn::Net& decoder, ncnn::Net& proj_out, const CacheIndexes& cache_indexes, int hidden_size, int rope_half_dim, int cur_seqlen, int past_seqlen, const std::vector<ncnn::Mat>& cache, std::vector<ncnn::Mat>& out_cache)
{
    ncnn::Mat token_embeds(hidden_size, cur_seqlen);
    token_embeds.fill(0.01f);

    ncnn::Mat attention_mask;
    make_attention_mask(cur_seqlen, past_seqlen, attention_mask);

    ncnn::Mat cos_cache;
    ncnn::Mat sin_cache;
    make_rope_cache(rope_half_dim, cur_seqlen, cos_cache, sin_cache);

    ncnn::Extractor ex = decoder.create_extractor();
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

    ncnn::Mat hidden;
    int ret = ex.extract("out0", hidden);
    if (ret != 0)
        return ret;

    ncnn::Mat last_hidden = hidden;
    if (cur_seqlen > 1)
    {
        last_hidden = hidden.row_range(cur_seqlen - 1, 1).clone();
    }

    ncnn::Extractor ex2 = proj_out.create_extractor();
    ex2.input("in0", last_hidden);

    ncnn::Mat logits;
    return ex2.extract("out0", logits);
}

static void benchmark_case(const char* name, ncnn::Net& decoder, ncnn::Net& proj_out, const CacheIndexes& cache_indexes, int hidden_size, int rope_half_dim, int cur_seqlen, int past_seqlen, const std::vector<ncnn::Mat>& cache, double rate_scale)
{
    std::vector<ncnn::Mat> out_cache;

    for (int i = 0; i < g_warmup_loop_count; i++)
    {
        run_decoder_once(decoder, proj_out, cache_indexes, hidden_size, rope_half_dim, cur_seqlen, past_seqlen, cache, out_cache);
    }

    double time_min = DBL_MAX;
    double time_max = -DBL_MAX;
    double time_avg = 0;

    for (int i = 0; i < g_loop_count; i++)
    {
        double start = ncnn::get_current_time();
        run_decoder_once(decoder, proj_out, cache_indexes, hidden_size, rope_half_dim, cur_seqlen, past_seqlen, cache, out_cache);
        double end = ncnn::get_current_time();

        double time = end - start;
        time_min = std::min(time_min, time);
        time_max = std::max(time_max, time);
        time_avg += time;
    }

    time_avg /= g_loop_count;

    const double tokens_per_second = rate_scale * 1000.0 / time_avg;
    fprintf(stderr, "%30s  min = %7.2f  max = %7.2f  avg = %7.2f  tps = %7.2f\n", name, time_min, time_max, time_avg, tokens_per_second);
}

static int load_net(ncnn::Net& net, const char* param_data, const ncnn::Option& opt)
{
    net.opt = opt;

#if NCNN_VULKAN
    if (net.opt.use_vulkan_compute)
    {
        net.set_vulkan_device(g_vkdev);
    }
#endif // NCNN_VULKAN

    int ret = net.load_param_mem(param_data);
    if (ret != 0)
        return ret;

    DataReaderFromEmpty dr;
    return net.load_model(dr);
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

    ncnn::Net decoder;
    int ret = load_net(decoder, config.decoder_param_data, opt);
    if (ret != 0)
        return ret;

    ncnn::Net proj_out;
    ret = load_net(proj_out, config.proj_out_param_data, opt);
    if (ret != 0)
        return ret;

    CacheIndexes cache_indexes;
    resolve_cache_indexes(decoder, cache_indexes);

    if (g_enable_cooling_down)
    {
        ncnn::sleep(10 * 1000);
    }

    std::vector<ncnn::Mat> empty_cache;
    std::vector<ncnn::Mat> past_cache;
    run_decoder_once(decoder, proj_out, cache_indexes, config.hidden_size, config.rope_half_dim, 1024, 0, empty_cache, past_cache);

    char prefill_name[256];
    snprintf(prefill_name, 256, "%s_1k_prefill", config.name);

    benchmark_case(prefill_name, decoder, proj_out, cache_indexes, config.hidden_size, config.rope_half_dim, 1024, 0, empty_cache, 1024.0);

    char decode_name[256];
    snprintf(decode_name, 256, "%s_1k_decode", config.name);

    benchmark_case(decode_name, decoder, proj_out, cache_indexes, config.hidden_size, config.rope_half_dim, 1, 1024, past_cache, 1.0);

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
        &hunyuan::model,
        &minicpm4::model,
        &qwen25::model,
        &qwen3::model,
        &llama32::model,
        &tinyllama::model,
        &youtu_llm::model,
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
