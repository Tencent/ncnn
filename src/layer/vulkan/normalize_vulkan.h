// Copyright 2019 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef LAYER_NORMALIZE_VULKAN_H
#define LAYER_NORMALIZE_VULKAN_H

#include "normalize.h"

namespace ncnn {

class Normalize_vulkan : public Normalize
{
public:
    Normalize_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using Normalize::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    VkMat scale_data_gpu;

    Pipeline* pipeline_normalize_reduce_sum4_fp16_to_fp32;
    Pipeline* pipeline_normalize_reduce_sum4_fp32[2];
    Pipeline* pipeline_normalize_coeffs;
    Pipeline* pipeline_normalize_norm;

    Pipeline* pipeline_normalize_reduce_sum4_fp16_to_fp32_pack4;
    Pipeline* pipeline_normalize_reduce_sum4_fp32_pack4[2];
    Pipeline* pipeline_normalize_coeffs_pack4;
    Pipeline* pipeline_normalize_norm_pack4;
};

} // namespace ncnn

#endif // LAYER_NORMALIZE_VULKAN_H
