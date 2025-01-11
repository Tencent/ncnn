// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2022 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "flip.h"

namespace ncnn {

Flip::Flip()
{
    one_blob_only = true;
}

int Flip::load_param(const ParamDict& pd)
{
    axis = pd.get(0, Mat());
    // 打印
    const int* axis_ptr = axis;
    printf("axis_len = %d", axis.w);
    printf("axis[0] = %d", axis_ptr[0]);
    printf("axis[1] = %d", axis_ptr[1]);
    return 0;
}

int Flip::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // wip
    return 0;
}

} // namespace ncnn
