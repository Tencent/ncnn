// Copyright 2025 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_LAYERNORM_VULKAN_H
#define LAYER_LAYERNORM_VULKAN_H

#include "layernorm.h"

namespace ncnn {

class LayerNorm_vulkan : public LayerNorm
{
public:
    LayerNorm_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using LayerNorm::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    // GPU-side model data
    VkMat gamma_data_gpu;
    VkMat beta_data_gpu;

    // Vulkan pipelines for the normalization process
    Pipeline* pipeline_layernorm_reduce_sum4_fp16_to_fp32;
    Pipeline* pipeline_layernorm_reduce_sum4_fp32[2];
    Pipeline* pipeline_layernorm_reduce_mean;
    Pipeline* pipeline_layernorm_sub_mean_square;
    Pipeline* pipeline_layernorm_coeffs;
    Pipeline* pipeline_layernorm_norm;
};

} // namespace ncnn

#endif // LAYER_LAYERNORM_VULKAN_H