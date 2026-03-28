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

    VkMat weight_data_gpu;
    VkMat bias_data_gpu;

    Pipeline* pipeline_deformableconv2d_packed;
    Pipeline* pipeline_deformableconv2d_packed_mask;
    Pipeline* pipeline_deformableconv2d_packed_gemm;
    Pipeline* pipeline_deformableconv2d_packed_gemm_mask;
};

} // namespace ncnn

#endif // LAYER_DEFORMABLECONV2D_VULKAN_H
