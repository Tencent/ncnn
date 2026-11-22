// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef NCNNDIT_QUANT_H
#define NCNNDIT_QUANT_H

#include "../modelwriter.h"

#include "cpu.h"
#include "ncnnllm_quant.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>

struct DiTQuantizeParams
{
    DiTQuantizeParams()
        : bits(8), block_size(0), rotation(0), method(LLM_QUANT_METHOD_MINMAX)
    {
    }

    int bits;
    int block_size;
    int rotation;
    int method;
};

struct DiTQuantizeOptions
{
    DiTQuantizeParams params;
    std::set<std::string> include;
    std::set<std::string> exclude;
};

static bool read_dit_layer_list(const char* path, std::set<std::string>& layers)
{
    if (!path)
        return true;

    std::ifstream file(path);
    if (!file)
    {
        fprintf(stderr, "cannot open layer list %s\n", path);
        return false;
    }

    std::string name;
    while (file >> name)
    {
        if (!layers.insert(name).second)
        {
            fprintf(stderr, "duplicate layer %s in %s\n", name.c_str(), path);
            return false;
        }
    }

    if (layers.empty() || !file.eof())
    {
        fprintf(stderr, "invalid layer list %s\n", path);
        return false;
    }

    return true;
}

static bool parse_dit_int(const char* arg, const char* key, int& value)
{
    const size_t len = strlen(key);
    if (strncmp(arg, key, len) != 0 || arg[len] < '0' || arg[len] > '9')
        return false;

    char* end;
    errno = 0;
    const long v = strtol(arg + len, &end, 10);
    if (errno || *end || v < 0 || v > INT_MAX)
        return false;

    value = (int)v;
    return true;
}

static bool is_dit_rotation_group(int group)
{
    if (group < 4)
        return false;

    while (group % 4 == 0)
        group /= 4;

    return group == 1;
}

static bool is_dit_quantize_params(const DiTQuantizeParams& params)
{
    if (params.method != LLM_QUANT_METHOD_MINMAX && params.method != LLM_QUANT_METHOD_MSECLIP)
        return false;

    if (params.bits == 8)
        return params.block_size == 0 && (params.rotation == 0 || is_dit_rotation_group(params.rotation));

    return (params.bits == 4 || params.bits == 6) && params.rotation == 0
           && llm_weight_block_quantize_term(params.bits, params.block_size) != 0;
}

static bool parse_dit_quantize_options(int argc, char** argv, int start, DiTQuantizeOptions* options, int* storage_type)
{
    const char* include_path = 0;
    const char* exclude_path = 0;

    for (int i = start; i < argc; i++)
    {
        const char* arg = argv[i];
        if (storage_type && strcmp(arg, "storage=fp32") == 0)
        {
            *storage_type = 0;
            continue;
        }
        if (storage_type && strcmp(arg, "storage=fp16") == 0)
        {
            *storage_type = 1;
            continue;
        }
        if (storage_type && strcmp(arg, "storage=bf16") == 0)
        {
            *storage_type = 2;
            continue;
        }

        if (!options)
        {
            fprintf(stderr, "unsupported argument %s with a DiT quantization table\n", arg);
            return false;
        }

        DiTQuantizeParams& params = options->params;
        if (strncmp(arg, "include=", 8) == 0)
            include_path = arg + 8;
        else if (strncmp(arg, "exclude=", 8) == 0)
            exclude_path = arg + 8;
        else if (strcmp(arg, "convrot=auto") == 0)
            params.rotation = -1;
        else if (strncmp(arg, "method=", 7) == 0)
            params.method = llm_quant_method_from_string(arg + 7);
        else if (!parse_dit_int(arg, "bits=", params.bits)
                 && !parse_dit_int(arg, "block=", params.block_size)
                 && !parse_dit_int(arg, "convrot=", params.rotation))
        {
            fprintf(stderr, "unsupported argument %s\n", arg);
            return false;
        }
    }

    if (!options)
        return true;

    DiTQuantizeParams layout = options->params;
    if (layout.bits != 8 && layout.block_size == 0)
        layout.block_size = 64;
    if (layout.rotation == -1)
        layout.rotation = 256;
    if (!is_dit_quantize_params(layout))
    {
        fprintf(stderr, "invalid DiT quantization options (bits, block, convrot or method)\n");
        return false;
    }
    options->params.block_size = layout.block_size;

    return read_dit_layer_list(include_path, options->include) && read_dit_layer_list(exclude_path, options->exclude);
}

static bool is_supported_dit_gemm(const ncnn::Gemm* op, int rotation)
{
    const int N = op->constantN;
    const int K = op->constantK;
    const int weight_w = op->transB ? K : N;
    const int weight_h = op->transB ? N : K;
    return op->quantize_term == 0 && !op->constantA && op->constantB == 1 && (op->transB == 0 || op->transB == 1)
           && (op->transA == 0 || op->transA == 1) && (op->output_N1M == 0 || op->output_N1M == 1)
           && (op->output_transpose == 0 || op->output_transpose == 1)
           && N > 0 && K > 0 && N <= INT_MAX / K
           && (rotation == 0 || (rotation > 0 && K % rotation == 0 && K <= INT_MAX / (128 * 128)))
           && op->B_data.dims == 2 && op->B_data.w == weight_w && op->B_data.h == weight_h
           && op->B_data.elemsize == 4u && op->B_data.elempack == 1 && (op->output_elemtype == 0 || op->output_elemtype == 1);
}

static int dit_rotation_group(int K)
{
    for (int group = 256; group >= 4; group /= 4)
    {
        if (K % group == 0)
            return group;
    }

    return -1;
}

static void create_dit_weight_scales(ncnn::Mat& scales, int K, int N, const DiTQuantizeParams& params)
{
    if (params.bits == 8)
    {
        scales.create(N);
        return;
    }

    const int block_count = (K - 1) / params.block_size + 1;
    scales.create(block_count, N);
}

static inline void rotate_convrot_row(float* ptr, int K, int group)
{
    // Use the same normalized regular 4x4 Hadamard transform as Gemm.
    for (int k = 0; k < K; k += group)
    {
        float* g = ptr + k;
        for (int stride = 1; stride < group; stride *= 4)
        {
            for (int i = 0; i < group; i += stride * 4)
            {
                for (int j = 0; j < stride; j++)
                {
                    float* v = g + i + j;
                    const float a = v[0];
                    const float b = v[stride];
                    const float c = v[stride * 2];
                    const float d = v[stride * 3];
                    const float sum = (a + b + c + d) * 0.5f;
                    v[0] = sum - d;
                    v[stride] = sum - c;
                    v[stride * 2] = sum - b;
                    v[stride * 3] = sum - a;
                }
            }
        }
    }
}

static int quantize_dit_gemm_int8(const ncnn::Gemm* op, const DiTQuantizeParams& params, ncnn::Mat& scales, ncnn::Mat* weight_data_quantized, int num_threads)
{
    const int N = op->constantN;
    const int K = op->constantK;
    const bool compute_scales = scales.empty();
    if (compute_scales)
    {
        scales.create(N);
        if (scales.empty())
            return -100;
    }
    if (weight_data_quantized)
    {
        weight_data_quantized->create(K, N, (size_t)1u);
        if (weight_data_quantized->empty())
            return -100;
    }

    num_threads = std::min(num_threads, N);
    ncnn::Mat rows;
    if (!op->transB || params.rotation)
    {
        rows.create(K, num_threads);
        if (rows.empty())
            return -100;
    }

    const float* B = op->B_data;
    int invalid_row = N;
    #pragma omp parallel for num_threads(num_threads)
    for (int n = 0; n < N; n++)
    {
        const float* ptr = B + (size_t)n * K;
        if (!rows.empty())
        {
            float* row = rows.row(ncnn::get_omp_thread_num());
            if (op->transB)
                memcpy(row, ptr, K * sizeof(float));
            else
                for (int k = 0; k < K; k++)
                    row[k] = B[(size_t)k * N + n];

            if (params.rotation)
                rotate_convrot_row(row, K, params.rotation);
            ptr = row;
        }

        float absmax = 0.f;
        for (int k = 0; k < K; k++)
        {
            if (!std::isfinite(ptr[k]))
            {
                absmax = -1.f;
                break;
            }
            absmax = std::max(absmax, fabsf(ptr[k]));
        }

        if (absmax < 0.f || (!compute_scales && scales[n] == 0.f && absmax != 0.f))
        {
            #pragma omp critical
            {
                invalid_row = std::min(invalid_row, n);
            }
            continue;
        }

        if (compute_scales)
        {
            float scale = absmax / 127.f;
            if (scale == 0.f)
                scale = absmax;
            if (absmax != 0.f && params.method == LLM_QUANT_METHOD_MSECLIP)
                scale = 1.f / choose_weight_scale(ptr, K, 8, params.method);
            scales[n] = scale;
        }

        if (!weight_data_quantized)
            continue;

        signed char* outptr = weight_data_quantized->row<signed char>(n);
        const float scale = scales[n];
        if (scale == 0.f)
        {
            memset(outptr, 0, K);
            continue;
        }

        const float inv_scale = 1.f / scale;
        if (std::isfinite(inv_scale))
        {
            for (int k = 0; k < K; k++)
                outptr[k] = (signed char)float2int_weight(ptr[k] * inv_scale, 8);
        }
        else
        {
            for (int k = 0; k < K; k++)
                outptr[k] = (signed char)float2int_weight(ptr[k] / scale, 8);
        }
    }

    if (invalid_row < N)
    {
        fprintf(stderr, "invalid INT8 weights or zero scale in %s row %d\n", op->name.c_str(), invalid_row);
        return -1;
    }

    return 0;
}

static int quantize_dit_gemm_blocks(const ncnn::Gemm* op, const DiTQuantizeParams& params, ncnn::Mat& scales, ncnn::Mat* weight_data_quantized, int num_threads)
{
    const int K = op->constantK;
    const int N = op->constantN;
    const float* weights = op->B_data;
    ncnn::Mat BT = op->B_data;
    if (!op->transB)
    {
        BT.create(K, N);
        if (BT.empty())
            return -100;
    }

    const int stride = op->transB ? 1 : N;
    int invalid_row = N;
    #pragma omp parallel for num_threads(num_threads)
    for (int n = 0; n < N; n++)
    {
        const float* ptr = op->transB ? weights + (size_t)n * K : weights + n;
        float* outptr = BT.row(n);
        for (int k = 0; k < K; k++)
        {
            const float v = ptr[(size_t)k * stride];
            if (!std::isfinite(v))
            {
                #pragma omp critical
                {
                    invalid_row = std::min(invalid_row, n);
                }
                break;
            }
            if (!op->transB)
                outptr[k] = v;
        }
    }
    if (invalid_row < N)
    {
        fprintf(stderr, "non-finite weight in %s row %d\n", op->name.c_str(), invalid_row);
        return -1;
    }

    if (scales.empty())
    {
        const int ret = make_weight_scales(BT, params.block_size, params.bits, params.method, scales, num_threads);
        if (ret != 0)
            return ret;
    }

    if (weight_data_quantized)
        return pack_weight_data(BT, scales, params.block_size, params.bits, *weight_data_quantized, num_threads);

    return 0;
}

static int quantize_dit_weight(const ncnn::Gemm* op, const DiTQuantizeParams& params, ncnn::Mat& scales, ncnn::Mat* weight_data_quantized, int num_threads)
{
    if (params.bits == 8)
        return quantize_dit_gemm_int8(op, params, scales, weight_data_quantized, num_threads);

    return quantize_dit_gemm_blocks(op, params, scales, weight_data_quantized, num_threads);
}

#endif // NCNNDIT_QUANT_H
