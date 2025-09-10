// Copyright 2025 Futz12 <pchar.cn>
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
    // a = gamma / sqrt(var + eps)
    // b = -mean * a + beta
    VkMat gamma_data_gpu;
    VkMat beta_data_gpu;

    // pack1 pipelines
    Pipeline* pipeline_layernorm_reduce_sum4_fp16_to_fp32;
    Pipeline* pipeline_layernorm_reduce_sum4_fp32[2];
    Pipeline* pipeline_layernorm_reduce_mean;
    Pipeline* pipeline_layernorm_sub_mean_square;
    Pipeline* pipeline_layernorm_coeffs;
    Pipeline* pipeline_layernorm_norm;

    // pack4 pipelines
    Pipeline* pipeline_layernorm_reduce_sum4_fp16_to_fp32_pack4;
    Pipeline* pipeline_layernorm_reduce_sum4_fp32_pack4[2];
    Pipeline* pipeline_layernorm_reduce_mean_pack4;
    Pipeline* pipeline_layernorm_sub_mean_square_pack4;
    Pipeline* pipeline_layernorm_coeffs_pack4;
    Pipeline* pipeline_layernorm_norm_pack4;
};

} // namespace ncnn

#endif // LAYER_LAYERNORM_VULKAN_H
