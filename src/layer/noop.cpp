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

#include "noop.h"
#include "cpu.h"

namespace ncnn {

Noop::Noop()
{
    support_inplace = true;
    support_vulkan = true;
    support_packing = true;
    support_fp16_storage = cpu_support_arm_asimdhp();
    support_bf16_storage = true;
    support_image_storage = true;
}

int Noop::forward_inplace(std::vector<Mat>& /*bottom_top_blobs*/, const Option& /*opt*/) const
{
    return 0;
}

#if NCNN_VULKAN
int Noop::forward_inplace(std::vector<VkMat>& /*bottom_top_blobs*/, VkCompute& /*cmd*/, const Option& /*opt*/) const
{
    return 0;
}

int Noop::forward_inplace(std::vector<VkImageMat>& /*bottom_top_blobs*/, VkCompute& /*cmd*/, const Option& /*opt*/) const
{
    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
