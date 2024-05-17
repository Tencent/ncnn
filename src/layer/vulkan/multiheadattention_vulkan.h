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

#ifndef LAYER_MULTIHEADATTENTION_VULKAN_H
#define LAYER_MULTIHEADATTENTION_VULKAN_H

#include "multiheadattention.h"

namespace ncnn {

class MultiHeadAttention_vulkan : public MultiHeadAttention
{
public:
    MultiHeadAttention_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    using MultiHeadAttention::forward;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;
    virtual int forward(const std::vector<VkImageMat>& bottom_blobs, std::vector<VkImageMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

public:
    Layer* q_gemm;
    Layer* k_gemm;
    Layer* v_gemm;
    Layer* o_gemm;

    Layer* qk_softmax;

    Pipeline* pipeline_multiheadattention_qk_cross;
    Pipeline* pipeline_multiheadattention_qk_cross_pack4;
    Pipeline* pipeline_multiheadattention_qk_cross_pack1to4;
    Pipeline* pipeline_multiheadattention_qk_cross_pack4to1;

    Pipeline* pipeline_multiheadattention_qkv_cross;
    Pipeline* pipeline_multiheadattention_qkv_cross_pack4;
    Pipeline* pipeline_multiheadattention_qkv_cross_pack1to4;
    Pipeline* pipeline_multiheadattention_qkv_cross_pack4to1;
};

} // namespace ncnn

#endif // LAYER_MULTIHEADATTENTION_VULKAN_H
