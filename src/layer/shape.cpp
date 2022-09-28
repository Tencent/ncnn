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

#include "shape.h"
#include <stdio.h>

namespace ncnn {
Shape::Shape()
{
    one_blob_only = true;
    support_inplace = false;
}

int Shape::load_param(const ParamDict& pd)
{
    return 0;
}

int Shape::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    // int
    if (bottom_blob.dims != 4 && bottom_blob.dims != 3)
    {
        printf("unexpect bottom_blob.dims:%d\n", bottom_blob.dims);
    }
    top_blob.create(4, 4, opt.blob_allocator);
    if (top_blob.empty())
    {
        printf("top_blob is empty!\n");
        return -100; // return non-zero on error, -100 indicates out-of-memory
    }
    float* dptr = (float*)top_blob.data;
    dptr[0] = bottom_blob.d;
    dptr[1] = bottom_blob.c;
    dptr[2] = bottom_blob.h;
    dptr[3] = bottom_blob.w;
    // printf("dptr:%f, %f, %f,%f\n", dptr[0], dptr[1], dptr[2], dptr[3]);
    return 0;
}

} // namespace ncnn
