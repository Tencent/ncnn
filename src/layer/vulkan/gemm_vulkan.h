// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2023 THL A29 Limited, a Tencent company. All rights reserved.
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

#ifndef LAYER_GEMM_VULKAN_H
#define LAYER_GEMM_VULKAN_H

#include "gemm.h"

namespace ncnn {

class Gemm_vulkan : public Gemm
{
public:
    Gemm_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using Gemm::forward;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;
    virtual int forward(const std::vector<VkImageMat>& bottom_blobs, std::vector<VkImageMat>& top_blobs, VkCompute& cmd, const Option& opt) const;
    virtual int forward(const VkImageMat& bottom_blob, VkImageMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Mat A_data_packed;
    Mat B_data_packed;
    Mat C_data_packed;

    VkMat A_data_gpu;
    VkMat B_data_gpu;
    VkMat C_data_gpu;

    VkImageMat A_data_gpu_image;
    VkImageMat B_data_gpu_image;
    VkImageMat C_data_gpu_image;

    Pipeline* pipeline_gemm;
};

} // namespace ncnn

#endif // LAYER_GEMM_VULKAN_H
