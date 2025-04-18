// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
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

#include "testutil.h"

static int test_concat_oom(const std::vector<ncnn::Mat>& a, int axis)
{
    ncnn::ParamDict pd;
    pd.set(0, axis); //axis

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer_oom("Concat", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_concat_oom failed a[0].dims=%d a[0]=(%d %d %d %d) axis=%d\n", a[0].dims, a[0].w, a[0].h, a[0].d, a[0].c, axis);
    }

    return ret;
}

static int test_concat_0()
{
    std::vector<ncnn::Mat> as(8);
    as[0] = RandomMat(6, 6, 6, 6);
    as[1] = RandomMat(6, 6, 6, 6);
    as[2] = RandomMat(6, 6, 6, 6);
    as[3] = RandomMat(6, 6, 6, 6);
    as[4] = RandomMat(6, 6, 6, 6);
    as[5] = RandomMat(6, 6, 6, 6);
    as[6] = RandomMat(6, 6, 6, 6);
    as[7] = RandomMat(6, 6, 6, 6);

    return 0
           || test_concat_oom(as, 0)
           || test_concat_oom(as, 1)
           || test_concat_oom(as, 2)
           || test_concat_oom(as, 3);
}

static int test_concat_1()
{
    std::vector<ncnn::Mat> as(8);
    as[0] = RandomMat(6, 6, 6);
    as[1] = RandomMat(6, 6, 6);
    as[2] = RandomMat(6, 6, 6);
    as[3] = RandomMat(6, 6, 6);
    as[4] = RandomMat(6, 6, 6);
    as[5] = RandomMat(6, 6, 6);
    as[6] = RandomMat(6, 6, 6);
    as[7] = RandomMat(6, 6, 6);

    return 0
           || test_concat_oom(as, 0)
           || test_concat_oom(as, 1)
           || test_concat_oom(as, 2);
}

static int test_concat_2()
{
    std::vector<ncnn::Mat> as(8);
    as[0] = RandomMat(6, 6);
    as[1] = RandomMat(6, 6);
    as[2] = RandomMat(6, 6);
    as[3] = RandomMat(6, 6);
    as[4] = RandomMat(6, 6);
    as[5] = RandomMat(6, 6);
    as[6] = RandomMat(6, 6);
    as[7] = RandomMat(6, 6);

    return 0
           || test_concat_oom(as, 0)
           || test_concat_oom(as, 1);
}

static int test_concat_3()
{
    std::vector<ncnn::Mat> as(8);
    as[0] = RandomMat(6);
    as[1] = RandomMat(6);
    as[2] = RandomMat(6);
    as[3] = RandomMat(6);
    as[4] = RandomMat(6);
    as[5] = RandomMat(6);
    as[6] = RandomMat(6);
    as[7] = RandomMat(6);

    return test_concat_oom(as, 0);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_concat_0()
           || test_concat_1()
           || test_concat_2()
           || test_concat_3();
}
