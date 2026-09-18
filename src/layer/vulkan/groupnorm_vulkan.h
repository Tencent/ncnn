// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_GROUPNORM_VULKAN_H
#define LAYER_GROUPNORM_VULKAN_H

#include "groupnorm.h"

namespace ncnn {

class GroupNorm_vulkan : public GroupNorm
{
public:
    GroupNorm_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using GroupNorm::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    VkMat gamma_data_gpu;
    VkMat beta_data_gpu;

    Pipeline* pipeline_groupnorm_reduce_sum4_fp16_to_fp32;
    Pipeline* pipeline_groupnorm_reduce_sum4_fp32[2];
    Pipeline* pipeline_groupnorm_reduce_mean;
    Pipeline* pipeline_groupnorm_sub_mean_square;
    Pipeline* pipeline_groupnorm_coeffs;
    Pipeline* pipeline_groupnorm_norm;

    Pipeline* pipeline_groupnorm_reduce_sum4_fp16_to_fp32_pack4;
    Pipeline* pipeline_groupnorm_reduce_sum4_fp32_pack4[2];
    Pipeline* pipeline_groupnorm_reduce_mean_pack4;
    Pipeline* pipeline_groupnorm_sub_mean_square_pack4;
    Pipeline* pipeline_groupnorm_coeffs_pack4;
    Pipeline* pipeline_groupnorm_norm_pack4;
};

} // namespace ncnn

#endif // LAYER_GROUPNORM_VULKAN_H
