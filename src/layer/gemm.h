// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GEMM_H
#define LAYER_GEMM_H

#include "layer.h"

namespace ncnn {

class Gemm : public Layer
{
public:
    Gemm();

    virtual int load_param(const ParamDict& pd);

    virtual int load_model(const ModelBin& mb);

    virtual int forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
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

    int int8_scale_term;

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
};

} // namespace ncnn

#endif // LAYER_GEMM_H
