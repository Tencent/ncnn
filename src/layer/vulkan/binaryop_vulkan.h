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

#ifndef LAYER_BINARYOP_VULKAN_H
#define LAYER_BINARYOP_VULKAN_H

#include "binaryop.h"

namespace ncnn {

class BinaryOp_vulkan : public BinaryOp
{
public:
    BinaryOp_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using BinaryOp::forward;
    using BinaryOp::forward_inplace;
    virtual int forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& cmd, const Option& opt) const;

    virtual int forward_inplace(VkMat& bottom_top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_binaryop;
    Pipeline* pipeline_binaryop_pack4;
    Pipeline* pipeline_binaryop_pack8;

    // broadcast
    Pipeline* pipeline_binaryop_broadcast[2];
    Pipeline* pipeline_binaryop_broadcast_pack4[2];
    Pipeline* pipeline_binaryop_broadcast_pack1to4[2];
    Pipeline* pipeline_binaryop_broadcast_pack8[2];
    Pipeline* pipeline_binaryop_broadcast_pack1to8[2];
};

} // namespace ncnn

#endif // LAYER_BINARYOP_VULKAN_H
