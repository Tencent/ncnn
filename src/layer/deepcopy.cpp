// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
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
