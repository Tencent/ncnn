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

namespace ncnn {

T::T()
{
    one_blob_only = true;
}

int T::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    if (dims == 1)
    {
        top_blob = bottom_blob.clone();
        return 0;
    }
    else if (dims == 2)
    {
        top_blob.create(h, w, elemsize);

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < w; i++)
        {
            float* top_row = top_blob.row(i);
            for (int j = 0; j < h; j++)
            {
                top_row[j] = bottom_blob[j * w + i];
            }
        }
    }
    else
    {
        NCNN_LOGE("Expects input to be 1-D Mat or 2-D Mat, current dimension is %d", dims);
        return -1;
    }

    return 0;
}

} // namespace ncnn
