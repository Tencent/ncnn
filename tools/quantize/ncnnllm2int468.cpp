// Copyright 2019 BUG1989
// SPDX-License-Identifier: BSD-3-Clause

#ifdef _MSC_VER
#define _CRT_SECURE_NO_DEPRECATE
#endif

#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <vector>

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
    virtual int scan(const char* format, void* p) const
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

    std::map<std::string, ncnn::Mat> blob_int8scale_table;
    std::map<std::string, ncnn::Mat> weight_int8scale_table;

public:
    int quantize_gemm(int block_size, int nbits);
};

NetQuantize::NetQuantize()
    : ModelWriter()
{
}

static inline signed char float2int8(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 127) return 127;
    if (int32 < -127) return -127;
    return (signed char)int32;
}

static inline signed char float2int6(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 31) return 31;
    if (int32 < -31) return -31;
    return (signed char)int32;
}

static inline signed char float2int4(float v)
{
    int int32 = static_cast<int>(round(v));
    if (int32 > 7) return 7;
    if (int32 < -7) return -7;
    return (signed char)int32;
}

int NetQuantize::quantize_gemm(int block_size, int nbits)
{
    for (size_t i = 0; i < layers.size(); i++)
    {
        if (layers[i]->type != "Gemm")
            continue;

        // Gemm - quantize weight from fp32 to int8
        ncnn::Gemm* gemm = (ncnn::Gemm*)layers[i];

        // quantize gemm layers in llm decoder
        // 2=0 3=1 4=0 5=1 6=1 7=0 8=N 9=K 10=-1
        if (gemm->alpha != 1.f || gemm->beta != 1.f)
            continue;

        if (gemm->transA != 0 || gemm->transB != 1)
            continue;

        if (gemm->constantA != 0 || gemm->constantB != 1 || gemm->constantC != 1 || gemm->constant_broadcast_type_C != -1)
            continue;

        if (gemm->constantM != 0)
            continue;

        if (gemm->output_N1M != 0 || gemm->output_elempack != 0 || gemm->output_elemtype != 0 || gemm->output_transpose != 0)
            continue;

        if (gemm->int8_scale_term != 0)
            continue;

        if (gemm->constant_TILE_M != 0 || gemm->constant_TILE_N != 0 || gemm->constant_TILE_K != 0)
            continue;

        fprintf(stderr, "quantize_gemm block_size=%d nbits=%d %s\n", block_size, nbits, gemm->name.c_str());

        // TODO move to ncnn2table

        const int constantN = gemm->constantN;
        const int constantK = gemm->constantK;

        // assert gemm->B_data.w == constantK
        // assert gemm->B_data.h == constantN

        const int block_count = (constantK + block_size - 1) / block_size;

        if (nbits == 8)
        {
            ncnn::Mat B_data_int8(constantK, constantN, (size_t)1u);
            ncnn::Mat B_data_int8_scales(block_count, constantN);

            for (int i = 0; i < constantN; i++)
            {
                const float* ptr = gemm->B_data.row(i);
                signed char* i8ptr = B_data_int8.row<signed char>(i);
                float* scale_ptr = B_data_int8_scales.row(i);

                for (int j = 0; j < block_count; j++)
                {
                    // block quantize
                    const float* ptr1 = ptr + j * block_size;
                    signed char* i8ptr1 = i8ptr + j * block_size;
                    const int block_size1 = std::min(block_size, constantK - j * block_size);

                    float absmax = 0.f;
                    for (int k = 0; k < block_size1; k++)
                    {
                        absmax = std::max(absmax, (float)fabs(ptr1[k]));
                    }

                    const float scale = absmax == 0.f ? 1.f : 127 / absmax;

                    for (int k = 0; k < block_size1; k++)
                    {
                        i8ptr1[k] = float2int8(ptr1[k] * scale);
                    }

                    scale_ptr[j] = scale;
                }
            }

            gemm->B_data = B_data_int8;
            gemm->B_data_quantize_scales = B_data_int8_scales;

            gemm->int8_scale_term = 4;
        }
        if (nbits == 6)
        {
            ncnn::Mat B_data_int6((constantK + 3) / 4 * 3, constantN, (size_t)1u);
            ncnn::Mat B_data_int6_scales(block_count, constantN);

            union i6x4_t
            {
                signed char i6[3];
                struct
                {
                    signed char i6_a : 6;
                    signed char i6_b : 6;
                    signed char i6_c : 6;
                    signed char i6_d : 6;
                } __attribute__((packed));
            };

            for (int i = 0; i < constantN; i++)
            {
                const float* ptr = gemm->B_data.row(i);
                i6x4_t* i6ptr = B_data_int6.row<i6x4_t>(i);
                float* scale_ptr = B_data_int6_scales.row(i);

                for (int j = 0; j < block_count; j++)
                {
                    // block quantize
                    const float* ptr1 = ptr + j * block_size;
                    i6x4_t* i6ptr1 = i6ptr + j * block_size / 4;
                    const int block_size1 = std::min(block_size, constantK - j * block_size);

                    float absmax = 0.f;
                    for (int k = 0; k < block_size1; k++)
                    {
                        absmax = std::max(absmax, (float)fabs(ptr1[k]));
                    }

                    const float scale = absmax == 0.f ? 1.f : 31 / absmax;

                    int k = 0;
                    for (; k + 3 < block_size1; k += 4)
                    {
                        i6ptr1[k / 4].i6_a = float2int6(ptr1[k] * scale);
                        i6ptr1[k / 4].i6_b = float2int6(ptr1[k + 1] * scale);
                        i6ptr1[k / 4].i6_c = float2int6(ptr1[k + 2] * scale);
                        i6ptr1[k / 4].i6_d = float2int6(ptr1[k + 3] * scale);
                    }
                    for (; k + 2 < block_size1; k += 3)
                    {
                        i6ptr1[k / 4].i6_a = float2int6(ptr1[k] * scale);
                        i6ptr1[k / 4].i6_b = float2int6(ptr1[k + 1] * scale);
                        i6ptr1[k / 4].i6_c = float2int6(ptr1[k + 2] * scale);
                    }
                    for (; k + 1 < block_size1; k += 2)
                    {
                        i6ptr1[k / 4].i6_a = float2int6(ptr1[k] * scale);
                        i6ptr1[k / 4].i6_b = float2int6(ptr1[k + 1] * scale);
                    }
                    for (; k < block_size1; k++)
                    {
                        i6ptr1[k / 4].i6_a = float2int6(ptr1[k] * scale);
                    }

                    scale_ptr[j] = scale;
                }
            }

            gemm->B_data = B_data_int6;
            gemm->B_data_quantize_scales = B_data_int6_scales;

            gemm->int8_scale_term = 5;
        }
        if (nbits == 4)
        {
            ncnn::Mat B_data_int4((constantK + 1) / 2, constantN, (size_t)1u);
            ncnn::Mat B_data_int4_scales(block_count, constantN);

            union i4x2_t
            {
                signed char i4;
                struct
                {
                    signed char i4_low : 4;
                    signed char i4_high : 4;
                } __attribute__((packed));
            };

            for (int i = 0; i < constantN; i++)
            {
                const float* ptr = gemm->B_data.row(i);
                i4x2_t* i4ptr = B_data_int4.row<i4x2_t>(i);
                float* scale_ptr = B_data_int4_scales.row(i);

                for (int j = 0; j < block_count; j++)
                {
                    // block quantize
                    const float* ptr1 = ptr + j * block_size;
                    i4x2_t* i4ptr1 = i4ptr + j * block_size / 2;
                    const int block_size1 = std::min(block_size, constantK - j * block_size);

                    float absmax = 0.f;
                    for (int k = 0; k < block_size1; k++)
                    {
                        absmax = std::max(absmax, (float)fabs(ptr1[k]));
                    }

                    const float scale = absmax == 0.f ? 1.f : 7 / absmax;

                    int k = 0;
                    for (; k + 1 < block_size1; k += 2)
                    {
                        i4ptr1[k / 2].i4_low = float2int4(ptr1[k] * scale);
                        i4ptr1[k / 2].i4_high = float2int4(ptr1[k + 1] * scale);
                    }
                    for (; k < block_size1; k++)
                    {
                        i4ptr1[k / 2].i4_low = float2int4(ptr1[k] * scale);
                    }

                    scale_ptr[j] = scale;
                }
            }

            gemm->B_data = B_data_int4;
            gemm->B_data_quantize_scales = B_data_int4_scales;

            gemm->int8_scale_term = 6;
        }
    }

    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 5)
    {
        fprintf(stderr, "usage: %s [inparam] [inbin] [outparam] [outbin]\n", argv[0]);
        return -1;
    }

    const char* inparam = argv[1];
    const char* inbin = argv[2];
    const char* outparam = argv[3];
    const char* outbin = argv[4];

    const int block_size = 64; // FIXME hardcode
    // const int nbits = 8; // FIXME hardcode
    const int nbits = 6; // FIXME hardcode
    // const int nbits = 4; // FIXME hardcode

    NetQuantize quantizer;
    quantizer.storage_type = 1; // use fp16 where int8 not applied

    quantizer.load_param(inparam);
    if (strcmp(inbin, "null") == 0)
    {
        DataReaderFromEmpty dr;
        quantizer.load_model(dr);
        quantizer.gen_random_weight = true;
    }
    else
        quantizer.load_model(inbin);

    quantizer.quantize_gemm(block_size, nbits);

    quantizer.save(outparam, outbin);

    return 0;
}
