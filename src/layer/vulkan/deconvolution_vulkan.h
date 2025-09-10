// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DECONVOLUTION_VULKAN_H
#define LAYER_DECONVOLUTION_VULKAN_H

#include "deconvolution.h"

namespace ncnn {

class Deconvolution_vulkan : public Deconvolution
{
public:
    Deconvolution_vulkan();

    virtual int load_param(const ParamDict& pd);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using Deconvolution::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Mat weight_data_packed;

    VkMat weight_data_gpu;
    VkMat bias_data_gpu;

    ncnn::Layer* crop;
    ncnn::Layer* output_crop;

    Pipeline* pipeline_deconvolution;

    Pipeline* pipeline_deconvolution_gemm;
    Pipeline* pipeline_deconvolution_col2im;

    // cooperative matrix
    bool use_cooperative_matrix;
    int coopmat_M;
    int coopmat_N;
    int coopmat_K;
    int UNROLL_SG_M;
    int UNROLL_SG_N;
    int UNROLL_SG_K;
    int UNROLL_WG_M;
    int UNROLL_WG_N;
};

} // namespace ncnn

#endif // LAYER_DECONVOLUTION_VULKAN_H
