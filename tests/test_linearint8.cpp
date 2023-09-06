// TODO

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

#include "layer/linearint8.h"
#include "testutil.h"

static int test_linearint8(const ncnn::Mat& a, int in_dim, int out_dim, int group_size)
{
    if (in_dim * out_dim % group_size)
    {
        fprintf(stderr, "malformed test case: in_dim=%d out_dim=%d group_size=%d\n", in_dim, out_dim, group_size);
        return -1;
    }
    if (a.w != in_dim)
    {
        fprintf(stderr, "malformed test case: in_dim=%d out_dim=%d group_size=%d\n", in_dim, out_dim, group_size);
        return -1;
    }
    ncnn::ParamDict pd;
    pd.set(0, in_dim);
    pd.set(1, out_dim);
    pd.set(2, group_size);

    std::vector<ncnn::Mat> weights(2);
    weights[0] = RandomMat(in_dim * out_dim / group_size);
    weights[1] = RandomS8Mat(in_dim * out_dim);

    int ret = test_layer<ncnn::LinearInt8>("LinearInt8", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_linearint8 failed a.dims=%d a=(%d, %d) in_dim=%d out_dim=%d group_size=%d\n", a.dims, a.h, a.w, in_dim, out_dim, group_size);
    }

    return ret;
}

static int test_lrn_0()
{
    ncnn::Mat a = RandomMat(10, 1);

    return 0
           || test_linearint8(a, 10, 6, 4)
           || test_linearint8(a, 10, 8, 4)
           || test_linearint8(a, 10, 10, 4);
}

static int test_lrn_1()
{
    ncnn::Mat a = RandomMat(16, 1);

    return 0
           || test_linearint8(a, 16, 6, 16)
           || test_linearint8(a, 16, 6, 16)
           || test_linearint8(a, 16, 6, 16)
           || test_linearint8(a, 16, 6, 16);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_lrn_0()
           || test_lrn_1();
}
