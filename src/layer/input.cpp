// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "input.h"

namespace ncnn {

Input::Input()
{
    one_blob_only = true;
    support_inplace = true;
    support_vulkan = true;
    support_packing = true;
    support_bf16_storage = true;
}

int Input::load_param(const ParamDict& pd)
{
    w = pd.get(0, 0);
    h = pd.get(1, 0);
    d = pd.get(11, 0);
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
#endif // NCNN_VULKAN

} // namespace ncnn
