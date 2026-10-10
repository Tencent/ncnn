// Copyright 2025 Futz12 <pchar.cn>
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_RMSNORM_VULKAN_H
#define LAYER_RMSNORM_VULKAN_H

#include "rmsnorm.h"

namespace ncnn {

class RMSNorm_vulkan : public RMSNorm
{
public:
    RMSNorm_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using RMSNorm::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    VkMat gamma_data_gpu;

    // pack1
    Pipeline* pipeline_rmsnorm_square; // x -> x^2
    Pipeline* pipeline_rmsnorm_reduce_sum4_fp16_to_fp32;
    Pipeline* pipeline_rmsnorm_reduce_sum4_fp32[2];
    Pipeline* pipeline_rmsnorm_reduce_mean;
    Pipeline* pipeline_rmsnorm_coeffs;
    Pipeline* pipeline_rmsnorm_norm;

    // pack4
    Pipeline* pipeline_rmsnorm_square_pack4;
    Pipeline* pipeline_rmsnorm_reduce_sum4_fp16_to_fp32_pack4;
    Pipeline* pipeline_rmsnorm_reduce_sum4_fp32_pack4[2];
    Pipeline* pipeline_rmsnorm_reduce_mean_pack4;
    Pipeline* pipeline_rmsnorm_coeffs_pack4;
    Pipeline* pipeline_rmsnorm_norm_pack4;
};

} // namespace ncnn
#endif // LAYER_RMSNORM_VULKAN_H
