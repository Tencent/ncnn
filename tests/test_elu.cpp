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

#include "testutil.h"

static int test_elu(const ncnn::Mat& a)
{
    ncnn::ParamDict pd;
    float alpha = RandomFloat(0.001f, 1000.f);
    pd.set(0, alpha); //alpha

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("ELU", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_elu failed a.dims=%d a=(%d %d %d %d) alpha=%f\n", a.dims, a.w, a.h, a.d, a.c, alpha);
    }

    return ret;
}

static int test_elu_0()
{
    return 0
           || test_elu(RandomMat(7, 6, 5, 32))
           || test_elu(RandomMat(5, 6, 7, 24))
           || test_elu(RandomMat(7, 8, 9, 12))
           || test_elu(RandomMat(3, 4, 5, 13));
}

static int test_elu_1()
{
    return 0
           || test_elu(RandomMat(4, 7, 32))
           || test_elu(RandomMat(5, 7, 24))
           || test_elu(RandomMat(7, 9, 12))
           || test_elu(RandomMat(3, 5, 13));
}

static int test_elu_2()
{
    return 0
           || test_elu(RandomMat(13, 32))
           || test_elu(RandomMat(15, 24))
           || test_elu(RandomMat(17, 12))
           || test_elu(RandomMat(19, 15));
}

static int test_elu_3()
{
    return 0
           || test_elu(RandomMat(128))
           || test_elu(RandomMat(124))
           || test_elu(RandomMat(127))
           || test_elu(RandomMat(120));
}

int main()
{
    SRAND(7767517);

    return 0
           || test_elu_0()
           || test_elu_1()
           || test_elu_2()
           || test_elu_3();
}
