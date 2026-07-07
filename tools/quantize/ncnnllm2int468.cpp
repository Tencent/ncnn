// Copyright 2019 BUG1989
// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ncnn public header
#include "datareader.h"
#include "layer.h"
#include "layer_type.h"
#include "net.h"

// ncnn private header
#include "../modelwriter.h"

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

class NetQuantize : public ModelWriter
{
public:
    NetQuantize();

public:
    int quantize_gemm(int block_size, int weight_bits, int method);
};

NetQuantize::NetQuantize()
    : ModelWriter()
{
}

static void print_usage(const char* argv0)
{
    fprintf(stderr, "Usage: %s [inparam] [inbin] [outparam] [outbin] [(key=value)...]\n", argv0);
    fprintf(stderr, "  method=minmax/mseclip\n");
    fprintf(stderr, "  bits=4/6/8\n");
    fprintf(stderr, "  block=32/64/128\n");
    fprintf(stderr, "Sample usage:\n");
    fprintf(stderr, "  %s model.param model.bin model-int4.param model-int4.bin method=mseclip bits=4 block=64\n", argv0);
}

enum
{
    METHOD_MINMAX = 0,
    METHOD_MSECLIP = 1
};

static inline int float2int_weight(float v, int weight_bits)
{
    const int qmax = (1 << (weight_bits - 1)) - 1;
    int q = static_cast<int>(round(v));
    if (q > qmax) q = qmax;
    if (q < -qmax) q = -qmax;
    return q;
}

static inline void pack_signed_weight(unsigned char* ptr, int k, int weight_bits, int q)
{
    const unsigned int mask = (1u << weight_bits) - 1u;
    const unsigned int v = (unsigned int)q & mask;
    const int bit_offset = k * weight_bits;

    for (int b = 0; b < weight_bits; b++)
    {
        if (v & (1u << b))
        {
            const int out_bit = bit_offset + b;
            ptr[out_bit / 8] |= (unsigned char)(1u << (out_bit % 8));
        }
    }
}

static const char* gemm_name(const ncnn::Gemm* gemm)
{
#if NCNN_STRING
    return gemm->name.c_str();
#else
    (void)gemm;
    return "";
#endif
}

static void print_skip_gemm(const ncnn::Gemm* gemm, const char* reason)
{
    fprintf(stderr, "skip_gemm %s %s\n", gemm_name(gemm), reason);
}

static float choose_weight_scale(const float* ptr, int size, int weight_bits, int method)
{
    float absmax = 0.f;
    for (int i = 0; i < size; i++)
    {
        absmax = std::max(absmax, (float)fabs(ptr[i]));
    }

    if (absmax == 0.f)
        return 1.f;

    const int qmax = (1 << (weight_bits - 1)) - 1;
    if (method == METHOD_MINMAX)
        return (float)qmax / absmax;

    // Calibration-free weight clipping search. This keeps the runtime format
    // symmetric scale-only while improving block reconstruction MSE.
    float best_scale = (float)qmax / absmax;
    float best_error = 0.f;
    for (int i = 0; i < size; i++)
    {
        const int q = float2int_weight(ptr[i] * best_scale, weight_bits);
        const float deq = q / best_scale;
        const float diff = ptr[i] - deq;
        best_error += diff * diff;
    }

    const int search_steps = 20;
    for (int s = 1; s <= search_steps; s++)
    {
        const float shrink = 1.f - 0.5f * s / search_steps;
        const float scale = (float)qmax / (absmax * shrink);

        float error = 0.f;
        for (int i = 0; i < size; i++)
        {
            const int q = float2int_weight(ptr[i] * scale, weight_bits);
            const float deq = q / scale;
            const float diff = ptr[i] - deq;
            error += diff * diff;
        }

        if (error < best_error)
        {
            best_error = error;
            best_scale = scale;
        }
    }

    return best_scale;
}

int NetQuantize::quantize_gemm(int block_size, int weight_bits, int method)
{
    const int quantize_term = ncnn::gemm_weight_block_quantize_term(weight_bits, block_size);
    if (quantize_term == 0)
    {
        fprintf(stderr, "unsupported bits=%d or block=%d\n", weight_bits, block_size);
        return -1;
    }

    int quantized_count = 0;

    for (size_t i = 0; i < layers.size(); i++)
    {
        if (layers[i]->type != "Gemm")
            continue;

        ncnn::Gemm* gemm = (ncnn::Gemm*)layers[i];

        if (gemm->alpha != 1.f || gemm->beta != 1.f)
        {
            print_skip_gemm(gemm, "alpha/beta is not 1");
            continue;
        }

        if (gemm->transA != 0 || gemm->transB != 1)
        {
            print_skip_gemm(gemm, "requires transA=0 transB=1");
            continue;
        }

        if (gemm->constantA != 0 || gemm->constantB != 1 || gemm->constantC != 1 || gemm->constant_broadcast_type_C != -1)
        {
            print_skip_gemm(gemm, "requires constantA=0 constantB=1 constantC=1 broadcastC=-1");
            continue;
        }

        if (gemm->constantM != 0)
        {
            print_skip_gemm(gemm, "requires dynamic M");
            continue;
        }

        if (gemm->output_N1M != 0 || gemm->output_elempack != 0 || gemm->output_elemtype != 0 || gemm->output_transpose != 0)
        {
            print_skip_gemm(gemm, "unsupported output layout");
            continue;
        }

        if (gemm->quantize_term != 0)
        {
            print_skip_gemm(gemm, "already quantized");
            continue;
        }

        if (gemm->constant_TILE_M != 0 || gemm->constant_TILE_N != 0 || gemm->constant_TILE_K != 0)
        {
            print_skip_gemm(gemm, "tiled Gemm is not supported");
            continue;
        }

        const int constantN = gemm->constantN;
        const int constantK = gemm->constantK;

        if (constantN <= 0 || constantK <= 0 || gemm->B_data.w != constantK || gemm->B_data.h != constantN || gemm->B_data.elemsize != 4u)
        {
            print_skip_gemm(gemm, "B weight shape or storage is unsupported");
            continue;
        }

        fprintf(stderr, "quantize_gemm bits=%d block_size=%d term=%d %s\n", weight_bits, block_size, quantize_term, gemm_name(gemm));

        const int packed_k_bytes = ncnn::gemm_weight_quantize_packed_k_bytes(constantK, weight_bits);
        const int block_count = (constantK + block_size - 1) / block_size;

        ncnn::Mat B_data_quantized(packed_k_bytes, constantN, (size_t)1u);
        ncnn::Mat B_data_quantize_scales(block_count, constantN);
        if (B_data_quantized.empty() || B_data_quantize_scales.empty())
            return -100;

        memset(B_data_quantized.data, 0, B_data_quantized.total() * B_data_quantized.elemsize);

        for (int n = 0; n < constantN; n++)
        {
            const float* ptr = gemm->B_data.row(n);
            unsigned char* qptr = B_data_quantized.row<unsigned char>(n);
            float* scale_ptr = B_data_quantize_scales.row(n);

            for (int b = 0; b < block_count; b++)
            {
                const int k0 = b * block_size;
                const int max_kk = std::min(block_size, constantK - k0);

                const float scale = choose_weight_scale(ptr + k0, max_kk, weight_bits, method);
                scale_ptr[b] = scale;

                for (int k = 0; k < max_kk; k++)
                {
                    const int q = float2int_weight(ptr[k0 + k] * scale, weight_bits);
                    pack_signed_weight(qptr, k0 + k, weight_bits, q);
                }
            }
        }

        gemm->B_data = B_data_quantized;
        gemm->B_data_quantize_scales = B_data_quantize_scales;
        gemm->quantize_term = quantize_term;

        quantized_count++;
    }

    fprintf(stderr, "quantized %d Gemm layers\n", quantized_count);

    return 0;
}

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        print_usage(argv[0]);
        return -1;
    }

    for (int i = 1; i < argc; i++)
    {
        if (argv[i][0] == '-')
        {
            print_usage(argv[0]);
            return -1;
        }
    }

    const char* inparam = argv[1];
    const char* inbin = argv[2];
    const char* outparam = argv[3];
    const char* outbin = argv[4];

    int weight_bits = 6;
    int block_size = 64;
    const char* method = "minmax";

    for (int i = 5; i < argc; i++)
    {
        // key=value
        char* kv = argv[i];
        char* eqs = strchr(kv, '=');
        if (eqs == NULL)
        {
            fprintf(stderr, "unrecognized arg %s\n", kv);
            return -1;
        }

        eqs[0] = '\0';
        const char* key = kv;
        const char* value = eqs + 1;

        if (strcmp(key, "method") == 0)
            method = value;
        else if (strcmp(key, "bits") == 0)
            weight_bits = atoi(value);
        else if (strcmp(key, "block") == 0)
            block_size = atoi(value);
        else
        {
            fprintf(stderr, "unrecognized arg %s\n", key);
            return -1;
        }
    }

    int quantize_method = METHOD_MINMAX;
    if (strcmp(method, "minmax") == 0)
    {
        quantize_method = METHOD_MINMAX;
    }
    else if (strcmp(method, "mseclip") == 0)
    {
        quantize_method = METHOD_MSECLIP;
    }
    else if (strcmp(method, "gptq-import") == 0)
    {
        fprintf(stderr, "unsupported method=gptq-import, explicit qweight/scales import format is not implemented\n");
        return -1;
    }
    else
    {
        fprintf(stderr, "unsupported method=%s\n", method);
        return -1;
    }

    if (ncnn::gemm_weight_block_quantize_term(weight_bits, block_size) == 0)
    {
        print_usage(argv[0]);
        return -1;
    }

    NetQuantize quantizer;
    quantizer.storage_type = 1; // keep existing prototype behavior for unrelated fp32 weights

    if (quantizer.load_param(inparam) != 0)
        return -1;

    if (strcmp(inbin, "null") == 0)
    {
        DataReaderFromEmpty dr;
        if (quantizer.load_model(dr) != 0)
            return -1;
    }
    else
    {
        if (quantizer.load_model(inbin) != 0)
            return -1;
    }

    if (quantizer.quantize_gemm(block_size, weight_bits, quantize_method) != 0)
        return -1;

    quantizer.save(outparam, outbin);

    return 0;
}
