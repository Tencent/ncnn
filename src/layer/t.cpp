// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "t.h"

#include <float.h>

namespace ncnn {

T::T()
{
    one_blob_only = true;
    support_inplace = true;
}

int T::forward_inplace(Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int d = bottom_blob.d;
    int size = w * h * d;
    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        return 0;
    }
    else if (dims == 2)
    {
        for()
    }
    else
    {
        NCNN_LOGE("dims must <= 2, currently %d", dims);
    }

    #pragma omp parallel for num_threads(opt.num_threads)
    
    return 0;
}

} // namespace ncnn
