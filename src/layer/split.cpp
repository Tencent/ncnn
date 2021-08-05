// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "split.h"
#include "cpu.h"

namespace ncnn {

Split::Split()
{
    one_blob_only = false;
    support_inplace = false;
    support_vulkan = true;
    support_packing = true;
    support_fp16_storage = cpu_support_arm_asimdhp() || cpu_support_riscv_zfh();
    support_bf16_storage = true;
    support_image_storage = true;
}

int Split::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs, const Option& /*opt*/) const
{
    const Mat& bottom_blob = bottom_blobs[0];
    for (size_t i = 0; i < top_blobs.size(); i++)
    {
        top_blobs[i] = bottom_blob;
    }

    return 0;
}

#if NCNN_VULKAN
int Split::forward(const std::vector<VkMat>& bottom_blobs, std::vector<VkMat>& top_blobs, VkCompute& /*cmd*/, const Option& /*opt*/) const
{
    const VkMat& bottom_blob = bottom_blobs[0];
    for (size_t i = 0; i < top_blobs.size(); i++)
    {
        top_blobs[i] = bottom_blob;
    }

    return 0;
}

int Split::forward(const std::vector<VkImageMat>& bottom_blobs, std::vector<VkImageMat>& top_blobs, VkCompute& /*cmd*/, const Option& /*opt*/) const
{
    const VkImageMat& bottom_blob = bottom_blobs[0];
    for (size_t i = 0; i < top_blobs.size(); i++)
    {
        top_blobs[i] = bottom_blob;
    }

    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
