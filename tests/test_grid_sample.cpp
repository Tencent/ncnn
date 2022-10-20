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

#include "layer/gridsample.h"
#include "testutil.h"

static int test_grid_sample(const ncnn::Mat& a, int resize_type, int padding_mode, int align_corner)
{
    ncnn::ParamDict pd;
    pd.set(0, resize_type);
    pd.set(1, padding_mode);
    pd.set(2, align_corner);

    std::vector<ncnn::Mat> as(2);
    as[0] = a;
    as[1] = ncnn::Mat(a.w, a.h, 2);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Grid_Sample>("Grid_Sample", pd, weights, as);
    if (ret != 0)
    {
        fprintf(stderr, "test_grid_sample failed a.dims=%d a=(%d %d %d) resize_type=%d padding_mode=%d align_corner=%d", a.dims, a.w, a.h, a.c, resize_type, padding_mode, align_corner);
    }

    return ret;
}

static int test_grid_sample_0()
{
    // ncnn::Mat a = RandomMat(15, 16, 17);
    ncnn::Mat a = RandomMat(3, 3, 1);

    // ncnn::Mat c = RandomMat(15, 16, 2);
    // std::vector<ncnn::Mat> a = {b, c};

    return 0
           || test_grid_sample(a, 1, 1, 0)
           || test_grid_sample(a, 1, 1, 1)
           || test_grid_sample(a, 1, 2, 0)
           || test_grid_sample(a, 1, 2, 1)
           || test_grid_sample(a, 1, 3, 0)
           || test_grid_sample(a, 1, 3, 1)
           || test_grid_sample(a, 2, 1, 0)
           || test_grid_sample(a, 2, 1, 1)
           || test_grid_sample(a, 2, 2, 0)
           || test_grid_sample(a, 2, 2, 1)
           || test_grid_sample(a, 2, 3, 0)
           || test_grid_sample(a, 2, 3, 1)
           || test_grid_sample(a, 3, 1, 0)
           || test_grid_sample(a, 3, 1, 1)
           || test_grid_sample(a, 3, 2, 0)
           || test_grid_sample(a, 3, 2, 1)
           || test_grid_sample(a, 3, 3, 0)
           || test_grid_sample(a, 3, 3, 1);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_grid_sample_0();
}
