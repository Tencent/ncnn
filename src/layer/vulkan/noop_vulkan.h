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

#ifndef LAYER_NOOP_VULKAN_H
#define LAYER_NOOP_VULKAN_H

#include "noop.h"

namespace ncnn {

class Noop_vulkan : public Noop
{
public:
    Noop_vulkan();

    using Noop::forward;
    virtual int forward_inplace(std::vector<VkMat>& bottom_top_blobs, VkCompute& cmd, const Option& opt) const;
    virtual int forward_inplace(std::vector<VkImageMat>& bottom_top_blobs, VkCompute& cmd, const Option& opt) const;
};

} // namespace ncnn

#endif // LAYER_NOOP_VULKAN_H
