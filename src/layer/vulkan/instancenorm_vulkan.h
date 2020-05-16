// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef LAYER_INSTANCENORM_VULKAN_H
#define LAYER_INSTANCENORM_VULKAN_H

#include "instancenorm.h"

namespace ncnn {

class InstanceNorm_vulkan : virtual public InstanceNorm
{
public:
    InstanceNorm_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using InstanceNorm::forward_inplace;
    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;
    virtual int forward_inplace(VkImageMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    VkMat gamma_data_gpu;
    VkMat beta_data_gpu;
    VkImageMat gamma_data_gpu_image;
    VkImageMat beta_data_gpu_image;

    Pipeline* pipeline_instancenorm_reduce_sum4_fp16_to_fp32;
    Pipeline* pipeline_instancenorm_reduce_sum4_fp32[2];
    Pipeline* pipeline_instancenorm_reduce_mean;
    Pipeline* pipeline_instancenorm_sub_mean_square;
    Pipeline* pipeline_instancenorm_coeffs;
    Pipeline* pipeline_instancenorm_norm;

    Pipeline* pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack4;
    Pipeline* pipeline_instancenorm_reduce_sum4_fp32_pack4[2];
    Pipeline* pipeline_instancenorm_reduce_mean_pack4;
    Pipeline* pipeline_instancenorm_sub_mean_square_pack4;
    Pipeline* pipeline_instancenorm_coeffs_pack4;
    Pipeline* pipeline_instancenorm_norm_pack4;

    Pipeline* pipeline_instancenorm_reduce_sum4_fp16_to_fp32_pack8;
    Pipeline* pipeline_instancenorm_reduce_sum4_fp32_pack8[2];
    Pipeline* pipeline_instancenorm_reduce_mean_pack8;
    Pipeline* pipeline_instancenorm_sub_mean_square_pack8;
    Pipeline* pipeline_instancenorm_coeffs_pack8;
    Pipeline* pipeline_instancenorm_norm_pack8;
};

} // namespace ncnn

#endif // LAYER_INSTANCENORM_VULKAN_H
