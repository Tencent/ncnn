// Copyright 2022 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GEMM_ARM_H
#define LAYER_GEMM_ARM_H

#include "gemm.h"

namespace ncnn {

class Gemm_arm : public Gemm
{
public:
    Gemm_arm();

    virtual int create_pipeline(const Option& opt);

    virtual int forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;

protected:
#if NCNN_VFPV4
    int create_pipeline_fp16s(const Option& opt);
    int forward_fp16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#if NCNN_ARM82
    int create_pipeline_fp16sa(const Option& opt);
    int forward_fp16sa(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif
#endif
#if NCNN_BF16
    int create_pipeline_bf16s(const Option& opt);
    int forward_bf16s(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif
#if NCNN_INT8
    int create_pipeline_int8(const Option& opt);
    int forward_int8(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& opt) const;
#endif

public:
    int nT;
    Mat AT_data;
    Mat BT_data;
    Mat CT_data;

    int input_elemtype; // 0=auto 1=fp32 2=fp16 3=bf16
};

} // namespace ncnn

#endif // LAYER_GEMM_ARM_H
