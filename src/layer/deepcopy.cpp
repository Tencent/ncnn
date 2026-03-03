// Copyright 2020 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "deepcopy.h"

namespace ncnn {

DeepCopy::DeepCopy()
{
    one_blob_only = true;
    support_inplace = false;
    support_packing = true;
}

int DeepCopy::forward(const Mat& bottom_blob, Mat& top_blob, const Option& /*opt*/) const
{
    top_blob = bottom_blob.clone();
    if (top_blob.empty())
        return -100;

    return 0;
}

} // namespace ncnn
