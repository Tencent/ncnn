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

#include "input.h"

namespace ncnn {

Input::Input()
{
    one_blob_only = true;
    support_inplace = true;
    support_vulkan = true;
    support_packing = true;
    support_bf16_storage = true;
    support_image_storage = true;
}

int Input::load_param(const ParamDict& pd)
{
    w = pd.get(0, 0);
    h = pd.get(1, 0);
    c = pd.get(2, 0);

    return 0;
}

int Input::forward_inplace(Mat& /*bottom_top_blob*/, const Option& /*opt*/) const
{
    return 0;
}

#if NCNN_VULKAN
int Input::forward_inplace(VkMat& /*bottom_top_blob*/, VkCompute& /*cmd*/, const Option& /*opt*/) const
{
    return 0;
}

int Input::forward_inplace(VkImageMat& /*bottom_top_blob*/, VkCompute& /*cmd*/, const Option& /*opt*/) const
{
    return 0;
}
#endif // NCNN_VULKAN

} // namespace ncnn
