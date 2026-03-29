// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_DEFORMABLECONV2D_VULKAN_H
#define LAYER_DEFORMABLECONV2D_VULKAN_H

#include "deformableconv2d.h"

namespace ncnn {

class DeformableConv2D_vulkan : public DeformableConv2D
{
public:
    DeformableConv2D_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using DeformableConv2D::forward;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

public:
    ncnn::Layer* padding;

    Mat weight_data_packed;
    Mat weight_data_cm_packed;

    VkMat weight_data_gpu;
    VkMat weight_data_cm_gpu;
    VkMat bias_data_gpu;

    Pipeline* pipeline_deformableconv2d_packed;
    Pipeline* pipeline_deformableconv2d_packed_mask;
    Pipeline* pipeline_deformableconv2d_packed_gemm;
    Pipeline* pipeline_deformableconv2d_packed_gemm_mask;
    Pipeline* pipeline_deformableconv2d_gemm_cm;
    Pipeline* pipeline_deformableconv2d_gemm_cm_mask;

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

#endif // LAYER_DEFORMABLECONV2D_VULKAN_H
