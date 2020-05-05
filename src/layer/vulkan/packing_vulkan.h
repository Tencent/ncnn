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

#ifndef LAYER_PACKING_VULKAN_H
#define LAYER_PACKING_VULKAN_H

#include "packing.h"

namespace ncnn {

class Packing_vulkan : virtual public Packing
{
public:
    Packing_vulkan();

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    using Packing::forward;
    virtual int forward(const VkMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;
    virtual int forward(const VkImageMat& bottom_blob, VkImageMat& top_blob, VkCompute& cmd, const Option& opt) const;
    virtual int forward(const VkMat& bottom_blob, VkImageMat& top_blob, VkCompute& cmd, const Option& opt) const;
    virtual int forward(const VkImageMat& bottom_blob, VkMat& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    Pipeline* pipeline_packing;
    Pipeline* pipeline_packing_pack4;
    Pipeline* pipeline_packing_pack8;
    Pipeline* pipeline_packing_pack1to4;
    Pipeline* pipeline_packing_pack4to1;
    Pipeline* pipeline_packing_pack1to8;
    Pipeline* pipeline_packing_pack4to8;
    Pipeline* pipeline_packing_pack8to4;
    Pipeline* pipeline_packing_pack8to1;
};

} // namespace ncnn

#endif // LAYER_PACKING_VULKAN_H
