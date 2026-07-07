// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GEMM_H
#define LAYER_GEMM_H

#include "layer.h"

namespace ncnn {

static inline bool gemm_is_weight_block_quantize(int quantize_term)
{
    return quantize_term == 400 || quantize_term == 401 || quantize_term == 402
           || quantize_term == 600 || quantize_term == 601 || quantize_term == 602
           || quantize_term == 800 || quantize_term == 801 || quantize_term == 802;
}

static inline int gemm_weight_quantize_bits(int quantize_term)
{
    return gemm_is_weight_block_quantize(quantize_term) ? quantize_term / 100 : 0;
}

static inline int gemm_weight_quantize_block_size(int quantize_term)
{
    if (!gemm_is_weight_block_quantize(quantize_term))
        return 0;

    const int block_size_code = quantize_term % 100;
    return block_size_code == 0 ? 32 : block_size_code == 1 ? 64 : 128;
}

static inline int gemm_weight_block_quantize_term(int weight_bits, int block_size)
{
    const int block_size_code = block_size == 32 ? 0 : block_size == 64 ? 1 : block_size == 128 ? 2 : -1;
    if ((weight_bits != 4 && weight_bits != 6 && weight_bits != 8) || block_size_code < 0)
        return 0;

    return weight_bits * 100 + block_size_code;
}

static inline int gemm_weight_quantize_packed_k_bytes(int constantK, int weight_bits)
{
    return (constantK * weight_bits + 7) / 8;
}

class Gemm : public Layer
{
public:
    Gemm();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
#if NCNN_WEIGHT_QUANT
    int forward_weight_block_quantize(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif

#if NCNN_INT8
    int forward_int8(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif

public:
    float alpha;
    float beta;
    int transA;
    int transB;

    int constantA;
    int constantB;
    int constantC;
    int constantM;
    int constantN;
    int constantK;
    int constant_broadcast_type_C;
    int output_N1M;
    int output_elempack;
    int output_elemtype; // 0=auto 1=fp32
    int output_transpose;

    union
    {
        int quantize_term;
        int int8_scale_term;
    };

    int constant_TILE_M;
    int constant_TILE_N;
    int constant_TILE_K;

    // constant A / B / C
    Mat A_data;
    Mat B_data;
    Mat C_data;

#if NCNN_INT8
    Mat A_data_int8_scales;
    float B_data_int8_scale;
#endif
    Mat B_data_quantize_scales;
};

} // namespace ncnn

#endif // LAYER_GEMM_H
