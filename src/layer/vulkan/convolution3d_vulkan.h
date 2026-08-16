// Copyright 2026 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_CONVOLUTION3D_VULKAN_H
#define LAYER_CONVOLUTION3D_VULKAN_H

#include "convolution3d.h"

namespace ncnn {

class Convolution3D_vulkan : public Convolution3D
{
public:
    Convolution3D_vulkan();

    virtual int load_param(const ParamDict& pd);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using Convolution3D::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    ncnn::Layer* padding;

    Mat weight_data_packed;
    Mat weight_winograd222_data_packed;

    VkMat weight_data_gpu;
    VkMat bias_data_gpu;

    Pipeline* pipeline_convolution3d;
    Pipeline* pipeline_convolution3d_1x1x1;
    Pipeline* pipeline_convolution3d_gemm;

    // winograd222
    VkMat weight_data_gpu_tm_winograd222;
    Pipeline* pipeline_convolution3d_3x3x3_winograd222_transform_input;
    Pipeline* pipeline_convolution3d_3x3x3_winograd222_gemm;
    Pipeline* pipeline_convolution3d_3x3x3_winograd222_transform_output;

    // cooperative matrix
    bool use_cooperative_matrix;
    int coopmat_M;
    int coopmat_N;
    int coopmat_K;
    int coopmat_subgroup_size;
    int UNROLL_SG_M;
    int UNROLL_SG_N;
    int UNROLL_SG_K;
    int UNROLL_WG_M;
    int UNROLL_WG_N;
};

} // namespace ncnn

#endif // LAYER_CONVOLUTION3D_VULKAN_H


