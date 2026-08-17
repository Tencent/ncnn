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

#if NCNN_VULKAN
#include "command.h"
#endif // NCNN_VULKAN

#include "benchncnn_llm_param_data.h"

#ifndef NCNN_SIMPLESTL
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
    int quantize_term;
};

#if NCNN_WEIGHT_QUANT
static void make_weight_block_quantize_param(const char* param_data, int quantize_term, std::vector<char>& param_data_wq)
{
    const int weight_bits = quantize_term / 100;
    const int format_code = quantize_term % 100 / 10;
    const int block_size_code = quantize_term % 10;
    const bool valid_weight_block_quantize = (weight_bits == 4 || weight_bits == 6 || weight_bits == 8) && (format_code == 0 || format_code == 1) && block_size_code >= 0 && block_size_code <= 2;
    char quantize_term_param[16];
    const int quantize_term_param_length = sprintf(quantize_term_param, "18=%d", quantize_term);

    const char* line = param_data;

    while (*line)
    {
        const char* line_end = strchr(line, '\n');
        if (!line_end)
            line_end = line + strlen(line);

        bool use_weight_block_quantize = false;
        const char* quantize_term_ptr = 0;
        const char* quantize_term_end = 0;

        if (line_end - line > 5 && memcmp(line, "Gemm ", 5) == 0)
        {
            int transA = 0;
            int transB = 0;
            int constantA = 0;
            int constantB = 0;
            int constantN = 0;
            int constantK = 0;
            int output_N1M = 0;
            int output_elempack = 0;
            int output_elemtype = 0;
            int output_transpose = 0;
            int gemm_quantize_term = 0;
            int bottom_count = 0;
            int top_count = 0;
            int param_index = -1;
            int token_index = 0;
            const char* ptr = line;

            while (ptr < line_end)
            {
                const char* token = ptr;
                while (ptr < line_end && *ptr != ' ')
                    ptr++;

                if (token_index == 2)
                    sscanf(token, "%d", &bottom_count);
                if (token_index == 3)
                {
                    sscanf(token, "%d", &top_count);
                    param_index = 4 + bottom_count + top_count;
                }

                if (param_index != -1 && token_index >= param_index)
                {
                    int id;
                    int value;
                    if (sscanf(token, "%d=%d", &id, &value) == 2)
                    {
                        if (id == 2) transA = value;
                        if (id == 3) transB = value;
                        if (id == 4) constantA = value;
                        if (id == 5) constantB = value;
                        if (id == 8) constantN = value;
                        if (id == 9) constantK = value;
                        if (id == 11) output_N1M = value;
                        if (id == 12) output_elempack = value;
                        if (id == 13) output_elemtype = value;
                        if (id == 14) output_transpose = value;
                        if (id == 18)
                        {
                            gemm_quantize_term = value;
                            quantize_term_ptr = token;
                            quantize_term_end = ptr;
                        }
                    }
                }

                while (ptr < line_end && *ptr == ' ')
                    ptr++;
                token_index++;
            }

            use_weight_block_quantize = valid_weight_block_quantize && constantA == 0 && constantB == 1 && transB == 1 && (transA == 0 || (weight_bits == 8 && transA == 1)) && constantN > 0 && constantK > 0 && (weight_bits == 8 || output_N1M == 0) && output_elempack >= 0 && (weight_bits == 8 || output_elempack == 0) && (output_elemtype == 0 || output_elemtype == 1) && (output_transpose == 0 || (weight_bits == 8 && output_transpose == 1)) && gemm_quantize_term == 0;
        }

        if (use_weight_block_quantize && quantize_term_ptr)
        {
            size_t old_size = param_data_wq.size();
            param_data_wq.resize(old_size + quantize_term_ptr - line + quantize_term_param_length + line_end - quantize_term_end);
            char* outptr = param_data_wq.data() + old_size;
            memcpy(outptr, line, quantize_term_ptr - line);
            outptr += quantize_term_ptr - line;
            memcpy(outptr, quantize_term_param, quantize_term_param_length);
            outptr += quantize_term_param_length;
            memcpy(outptr, quantize_term_end, line_end - quantize_term_end);
        }
        else
        {
            size_t old_size = param_data_wq.size();
            param_data_wq.resize(old_size + line_end - line + (use_weight_block_quantize ? quantize_term_param_length + 1 : 0));
            char* outptr = param_data_wq.data() + old_size;
            memcpy(outptr, line, line_end - line);
            if (use_weight_block_quantize)
            {
                outptr[line_end - line] = ' ';
                memcpy(outptr + (line_end - line) + 1, quantize_term_param, quantize_term_param_length);
            }
        }

        if (*line_end == '\n')
        {
            param_data_wq.push_back('\n');
            line = line_end + 1;
        }
        else
        {
            line = line_end;
        }
    }

    param_data_wq.push_back('\0');
}
#endif // NCNN_WEIGHT_QUANT

namespace minicpm4 {

static const ModelConfig model = {"minicpm4_0.5b", minicpm4_0_5b_decoder_ncnn_param_data, minicpm4_0_5b_proj_out_ncnn_param_data, 1024, 32, 0};
#if NCNN_WEIGHT_QUANT
static const ModelConfig model_int8g32 = {"minicpm4_0.5b_int8g32", minicpm4_0_5b_decoder_ncnn_param_data, minicpm4_0_5b_proj_out_ncnn_param_data, 1024, 32, 800};
static const ModelConfig model_int8g128 = {"minicpm4_0.5b_int8g128", minicpm4_0_5b_decoder_ncnn_param_data, minicpm4_0_5b_proj_out_ncnn_param_data, 1024, 32, 802};
#endif

} // namespace minicpm4

namespace qwen25 {

static const ModelConfig model = {"qwen2.5_0.5b", qwen2_5_0_5b_decoder_ncnn_param_data, qwen2_5_0_5b_proj_out_ncnn_param_data, 896, 32, 0};
#if NCNN_WEIGHT_QUANT
static const ModelConfig model_int8g32 = {"qwen2.5_0.5b_int8g32", qwen2_5_0_5b_decoder_ncnn_param_data, qwen2_5_0_5b_proj_out_ncnn_param_data, 896, 32, 800};
static const ModelConfig model_int8g128 = {"qwen2.5_0.5b_int8g128", qwen2_5_0_5b_decoder_ncnn_param_data, qwen2_5_0_5b_proj_out_ncnn_param_data, 896, 32, 802};
#endif

} // namespace qwen25

namespace qwen3 {

static const ModelConfig model = {"qwen3_0.6b", qwen3_0_6b_decoder_ncnn_param_data, qwen3_0_6b_proj_out_ncnn_param_data, 1024, 64, 0};
#if NCNN_WEIGHT_QUANT
static const ModelConfig model_int8g32 = {"qwen3_0.6b_int8g32", qwen3_0_6b_decoder_ncnn_param_data, qwen3_0_6b_proj_out_ncnn_param_data, 1024, 64, 800};
static const ModelConfig model_int8g128 = {"qwen3_0.6b_int8g128", qwen3_0_6b_decoder_ncnn_param_data, qwen3_0_6b_proj_out_ncnn_param_data, 1024, 64, 802};
#endif

} // namespace qwen3

namespace hunyuan {

static const ModelConfig model = {"hunyuan_0.5b", hunyuan_0_5b_instruct_decoder_ncnn_param_data, hunyuan_0_5b_instruct_proj_out_ncnn_param_data, 1024, 64, 0};
#if NCNN_WEIGHT_QUANT
static const ModelConfig model_int8g32 = {"hunyuan_0.5b_int8g32", hunyuan_0_5b_instruct_decoder_ncnn_param_data, hunyuan_0_5b_instruct_proj_out_ncnn_param_data, 1024, 64, 800};
static const ModelConfig model_int8g128 = {"hunyuan_0.5b_int8g128", hunyuan_0_5b_instruct_decoder_ncnn_param_data, hunyuan_0_5b_instruct_proj_out_ncnn_param_data, 1024, 64, 802};
#endif

} // namespace hunyuan

namespace tinyllama {

static const ModelConfig model = {"tinyllama_1.1b", tinyllama_1_1b_decoder_ncnn_param_data, tinyllama_1_1b_proj_out_ncnn_param_data, 2048, 32, 0};
#if NCNN_WEIGHT_QUANT
static const ModelConfig model_int8g32 = {"tinyllama_1.1b_int8g32", tinyllama_1_1b_decoder_ncnn_param_data, tinyllama_1_1b_proj_out_ncnn_param_data, 2048, 32, 800};
static const ModelConfig model_int8g128 = {"tinyllama_1.1b_int8g128", tinyllama_1_1b_decoder_ncnn_param_data, tinyllama_1_1b_proj_out_ncnn_param_data, 2048, 32, 802};
#endif

} // namespace tinyllama

namespace llama32 {

static const ModelConfig model = {"llama3.2_1b", llama3_2_1b_decoder_ncnn_param_data, llama3_2_1b_proj_out_ncnn_param_data, 2048, 32, 0};
#if NCNN_WEIGHT_QUANT
static const ModelConfig model_int8g32 = {"llama3.2_1b_int8g32", llama3_2_1b_decoder_ncnn_param_data, llama3_2_1b_proj_out_ncnn_param_data, 2048, 32, 800};
static const ModelConfig model_int8g128 = {"llama3.2_1b_int8g128", llama3_2_1b_decoder_ncnn_param_data, llama3_2_1b_proj_out_ncnn_param_data, 2048, 32, 802};
#endif

} // namespace llama32

namespace youtu_llm {

static const ModelConfig model = {"youtu_llm_2b", youtu_llm_2b_decoder_ncnn_param_data, youtu_llm_2b_proj_out_ncnn_param_data, 2048, 64, 0};
#if NCNN_WEIGHT_QUANT
static const ModelConfig model_int8g32 = {"youtu_llm_2b_int8g32", youtu_llm_2b_decoder_ncnn_param_data, youtu_llm_2b_proj_out_ncnn_param_data, 2048, 64, 800};
static const ModelConfig model_int8g128 = {"youtu_llm_2b_int8g128", youtu_llm_2b_decoder_ncnn_param_data, youtu_llm_2b_proj_out_ncnn_param_data, 2048, 64, 802};
#endif

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

static int run_decoder_once(ncnn::Net& decoder, ncnn::Net& proj_out, const CacheIndexes& cache_indexes, const ncnn::Mat& token_embeds, const ncnn::Mat& attention_mask, const ncnn::Mat& cos_cache, const ncnn::Mat& sin_cache, std::vector<ncnn::Mat>& cache, ncnn::Allocator* kvcache_allocator, int kvcache_max_seqlen_hint)
{
    const int cur_seqlen = token_embeds.h;

    ncnn::Extractor ex = decoder.create_extractor();
    ex.set_kvcache_allocator(kvcache_allocator);
    ex.set_kvcache_max_seqlen_hint(kvcache_max_seqlen_hint);
    ex.input("in0", token_embeds);
    ex.input("in1", attention_mask);
    ex.input("in2", cos_cache);
    ex.input("in3", sin_cache);

    for (size_t i = 0; i < cache.size(); i++)
    {
        ex.input(cache_indexes.input_indexes[i], cache[i]);
        cache[i].release();
    }

    cache.resize(cache_indexes.output_indexes.size());
    for (size_t i = 0; i < cache_indexes.output_indexes.size(); i++)
    {
        int ret = ex.extract(cache_indexes.output_indexes[i], cache[i], 1);
        if (ret != 0)
            return ret;
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

#if NCNN_VULKAN
static int run_decoder_once_vulkan(ncnn::Net& decoder, ncnn::Net& proj_out, const CacheIndexes& cache_indexes, const ncnn::Mat& token_embeds, const ncnn::Mat& attention_mask, const ncnn::Mat& cos_cache, const ncnn::Mat& sin_cache, std::vector<ncnn::VkMat>& cache, ncnn::VkAllocator* kvcache_vkallocator, int kvcache_max_seqlen_hint)
{
    const int cur_seqlen = token_embeds.h;

    ncnn::Extractor ex = decoder.create_extractor();
    ex.set_kvcache_vkallocator(kvcache_vkallocator);
    ex.set_kvcache_max_seqlen_hint(kvcache_max_seqlen_hint);
    ex.input("in0", token_embeds);
    ex.input("in1", attention_mask);
    ex.input("in2", cos_cache);
    ex.input("in3", sin_cache);

    for (size_t i = 0; i < cache.size(); i++)
    {
        ex.input(cache_indexes.input_indexes[i], cache[i]);
        cache[i].release();
    }

    ncnn::Mat hidden;
    int ret = ex.extract("out0", hidden);
    if (ret != 0)
        return ret;

    ncnn::VkCompute cmd(g_vkdev);
    cache.resize(cache_indexes.output_indexes.size());
    for (size_t i = 0; i < cache_indexes.output_indexes.size(); i++)
    {
        ret = ex.extract(cache_indexes.output_indexes[i], cache[i], cmd);
        if (ret != 0)
            return ret;
    }

    ret = cmd.submit_and_wait();
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
#endif // NCNN_VULKAN

static int benchmark_cpu(ncnn::Net& decoder, ncnn::Net& proj_out, const CacheIndexes& cache_indexes, const ncnn::Mat& prefill_embeddings, const ncnn::Mat& prefill_attention_mask, const ncnn::Mat& prefill_cos_cache, const ncnn::Mat& prefill_sin_cache, const ncnn::Mat& decode_embedding, const ncnn::Mat& decode_attention_mask, const ncnn::Mat& decode_cos_cache, const ncnn::Mat& decode_sin_cache, ncnn::Allocator* kvcache_allocator, int kvcache_max_seqlen_hint, double& prefill_tps, double& decode_tps)
{
    double time_min = DBL_MAX;

    for (int i = 0; i < g_warmup_loop_count + g_loop_count; i++)
    {
        std::vector<ncnn::Mat> cache;

        double start = ncnn::get_current_time();
        int ret = run_decoder_once(decoder, proj_out, cache_indexes, prefill_embeddings, prefill_attention_mask, prefill_cos_cache, prefill_sin_cache, cache, kvcache_allocator, kvcache_max_seqlen_hint);
        double end = ncnn::get_current_time();

        for (size_t j = 0; j < cache.size(); j++)
            cache[j].release();

        if (ret != 0)
            return ret;

        if (i >= g_warmup_loop_count)
        {
            const double time = end - start;
            if (time < time_min)
                time_min = time;
        }
    }

    prefill_tps = prefill_embeddings.h * 1000.0 / time_min;

    time_min = DBL_MAX;

    for (int i = 0; i < g_warmup_loop_count + g_loop_count; i++)
    {
        std::vector<ncnn::Mat> cache;

        int ret = run_decoder_once(decoder, proj_out, cache_indexes, prefill_embeddings, prefill_attention_mask, prefill_cos_cache, prefill_sin_cache, cache, kvcache_allocator, kvcache_max_seqlen_hint);
        if (ret != 0)
        {
            for (size_t j = 0; j < cache.size(); j++)
                cache[j].release();
            return ret;
        }

        double start = ncnn::get_current_time();
        ret = run_decoder_once(decoder, proj_out, cache_indexes, decode_embedding, decode_attention_mask, decode_cos_cache, decode_sin_cache, cache, kvcache_allocator, kvcache_max_seqlen_hint);
        double end = ncnn::get_current_time();

        for (size_t j = 0; j < cache.size(); j++)
            cache[j].release();

        if (ret != 0)
            return ret;

        if (i >= g_warmup_loop_count)
        {
            const double time = end - start;
            if (time < time_min)
                time_min = time;
        }
    }

    decode_tps = 1000.0 / time_min;

    return 0;
}

#if NCNN_VULKAN
static int benchmark_vulkan(ncnn::Net& decoder, ncnn::Net& proj_out, const CacheIndexes& cache_indexes, const ncnn::Mat& prefill_embeddings, const ncnn::Mat& prefill_attention_mask, const ncnn::Mat& prefill_cos_cache, const ncnn::Mat& prefill_sin_cache, const ncnn::Mat& decode_embedding, const ncnn::Mat& decode_attention_mask, const ncnn::Mat& decode_cos_cache, const ncnn::Mat& decode_sin_cache, ncnn::VkAllocator* kvcache_vkallocator, int kvcache_max_seqlen_hint, double& prefill_tps, double& decode_tps)
{
    double time_min = DBL_MAX;

    for (int i = 0; i < g_warmup_loop_count + g_loop_count; i++)
    {
        std::vector<ncnn::VkMat> cache;

        double start = ncnn::get_current_time();
        int ret = run_decoder_once_vulkan(decoder, proj_out, cache_indexes, prefill_embeddings, prefill_attention_mask, prefill_cos_cache, prefill_sin_cache, cache, kvcache_vkallocator, kvcache_max_seqlen_hint);
        double end = ncnn::get_current_time();

        for (size_t j = 0; j < cache.size(); j++)
            cache[j].release();

        if (ret != 0)
            return ret;

        if (i >= g_warmup_loop_count)
        {
            const double time = end - start;
            if (time < time_min)
                time_min = time;
        }
    }

    prefill_tps = prefill_embeddings.h * 1000.0 / time_min;

    time_min = DBL_MAX;

    for (int i = 0; i < g_warmup_loop_count + g_loop_count; i++)
    {
        std::vector<ncnn::VkMat> cache;

        int ret = run_decoder_once_vulkan(decoder, proj_out, cache_indexes, prefill_embeddings, prefill_attention_mask, prefill_cos_cache, prefill_sin_cache, cache, kvcache_vkallocator, kvcache_max_seqlen_hint);
        if (ret != 0)
        {
            for (size_t j = 0; j < cache.size(); j++)
                cache[j].release();
            return ret;
        }

        double start = ncnn::get_current_time();
        ret = run_decoder_once_vulkan(decoder, proj_out, cache_indexes, decode_embedding, decode_attention_mask, decode_cos_cache, decode_sin_cache, cache, kvcache_vkallocator, kvcache_max_seqlen_hint);
        double end = ncnn::get_current_time();

        for (size_t j = 0; j < cache.size(); j++)
            cache[j].release();

        if (ret != 0)
            return ret;

        if (i >= g_warmup_loop_count)
        {
            const double time = end - start;
            if (time < time_min)
                time_min = time;
        }
    }

    decode_tps = 1000.0 / time_min;

    return 0;
}
#endif // NCNN_VULKAN

static int load_net(ncnn::Net& net, const char* param_data, int quantize_term, const ncnn::Option& opt)
{
    net.opt = opt;

#if NCNN_VULKAN
    if (net.opt.use_vulkan_compute)
    {
        net.set_vulkan_device(g_vkdev);
    }
#endif // NCNN_VULKAN

#if NCNN_WEIGHT_QUANT
    std::vector<char> param_data_wq;
    if (quantize_term)
    {
        make_weight_block_quantize_param(param_data, quantize_term, param_data_wq);
        param_data = param_data_wq.data();
    }
#else
    (void)quantize_term;
#endif

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
    int ret = load_net(decoder, config.decoder_param_data, config.quantize_term, opt);
    if (ret != 0)
        return ret;

    ncnn::Net proj_out;
    ret = load_net(proj_out, config.proj_out_param_data, config.quantize_term, opt);
    if (ret != 0)
        return ret;

    CacheIndexes cache_indexes;
    resolve_cache_indexes(decoder, cache_indexes);

    const int prefill_len = 256;

    ncnn::Mat prefill_embeddings(config.hidden_size, prefill_len);
    ncnn::Mat decode_embedding(config.hidden_size, 1);
    prefill_embeddings.fill(0.01f);
    decode_embedding.fill(0.01f);

    ncnn::Mat prefill_attention_mask;
    make_attention_mask(prefill_len, 0, prefill_attention_mask);
    ncnn::Mat prefill_cos_cache;
    ncnn::Mat prefill_sin_cache;
    make_rope_cache(config.rope_half_dim, prefill_len, prefill_cos_cache, prefill_sin_cache);

    ncnn::Mat decode_attention_mask;
    make_attention_mask(1, prefill_len, decode_attention_mask);
    ncnn::Mat decode_cos_cache;
    ncnn::Mat decode_sin_cache;
    make_rope_cache(config.rope_half_dim, 1, decode_cos_cache, decode_sin_cache);

    if (g_enable_cooling_down)
    {
        ncnn::sleep(10 * 1000);
    }

    double prefill_tps;
    double decode_tps;

#if NCNN_VULKAN
    if (opt.use_vulkan_compute)
    {
        ncnn::VkBlobAllocator kvcache_vkallocator(g_vkdev);
        ret = benchmark_vulkan(decoder, proj_out, cache_indexes, prefill_embeddings, prefill_attention_mask, prefill_cos_cache, prefill_sin_cache, decode_embedding, decode_attention_mask, decode_cos_cache, decode_sin_cache, &kvcache_vkallocator, prefill_len + 1, prefill_tps, decode_tps);
    }
    else
#endif // NCNN_VULKAN
    {
        ncnn::UnlockedPoolAllocator kvcache_allocator;
        kvcache_allocator.set_size_compare_ratio(0.f);
        ret = benchmark_cpu(decoder, proj_out, cache_indexes, prefill_embeddings, prefill_attention_mask, prefill_cos_cache, prefill_sin_cache, decode_embedding, decode_attention_mask, decode_cos_cache, decode_sin_cache, &kvcache_allocator, prefill_len + 1, prefill_tps, decode_tps);
    }

    if (ret != 0)
        return ret;

    fprintf(stderr, "%30s  %12.2f  %12.2f\n", config.name, prefill_tps, decode_tps);

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
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_int8_storage = true;
    opt.use_int8_arithmetic = true;
    opt.use_packing_layout = true;

    fprintf(stderr, "loop_count = %d\n", g_loop_count);
    fprintf(stderr, "num_threads = %d\n", num_threads);
    fprintf(stderr, "powersave = %d\n", ncnn::get_cpu_powersave());
    fprintf(stderr, "gpu_device = %d\n", gpu_device);
    fprintf(stderr, "cooling_down = %d\n", (int)g_enable_cooling_down);
    fprintf(stderr, "%30s  %12s  %12s\n", "model", "prefill tps", "decode tps");

    const ModelConfig* models[] = {
        &hunyuan::model,
#if NCNN_WEIGHT_QUANT
        &hunyuan::model_int8g32,
        &hunyuan::model_int8g128,
#endif
        &minicpm4::model,
#if NCNN_WEIGHT_QUANT
        &minicpm4::model_int8g32,
        &minicpm4::model_int8g128,
#endif
        &qwen25::model,
#if NCNN_WEIGHT_QUANT
        &qwen25::model_int8g32,
        &qwen25::model_int8g128,
#endif
        &qwen3::model,
#if NCNN_WEIGHT_QUANT
        &qwen3::model_int8g32,
        &qwen3::model_int8g128,
#endif
        &llama32::model,
#if NCNN_WEIGHT_QUANT
        &llama32::model_int8g32,
        &llama32::model_int8g128,
#endif
        &tinyllama::model,
#if NCNN_WEIGHT_QUANT
        &tinyllama::model_int8g32,
        &tinyllama::model_int8g128,
#endif
        &youtu_llm::model,
#if NCNN_WEIGHT_QUANT
        &youtu_llm::model_int8g32,
        &youtu_llm::model_int8g128,
#endif
    };

    for (size_t i = 0; i < sizeof(models) / sizeof(models[0]); i++)
    {
        if (use_vulkan_compute && models[i]->quantize_term)
            continue;

        int ret = benchmark_model(*models[i], opt);
        if (ret != 0)
        {
            fprintf(stderr, "benchmark %s failed %d\n", models[i]->name, ret);
            return ret;
        }
    }

#if NCNN_VULKAN
    delete g_blob_vkallocator;
    delete g_staging_vkallocator;
#endif // NCNN_VULKAN

    return 0;
}
