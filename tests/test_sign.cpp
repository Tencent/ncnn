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

#include "layer/sign.h"
#include "testutil.h"

static int test_sign(const ncnn::Mat& a)
{
    ncnn::ParamDict pd;

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Sign>("Sign", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_sign failed a.dims=%d a=(%d %d %d %d)\n", a.dims, a.w, a.h, a.d, a.c);
    }

    return ret;
}

static int test_sign_0()
{
    return 0
           || test_sign(RandomMat(10, 3, 12, 6))
           || test_sign(RandomMat(5, 3, 8, 24))
           || test_sign(RandomMat(3, 5, 7, 12));
}

static int test_sign_1()
{
    return 0
           || test_sign(RandomMat(6, 8, 12))
           || test_sign(RandomMat(5, 13, 24))
           || test_sign(RandomMat(6, 9, 12));
}

static int test_sign_2()
{
    return 0
           || test_sign(RandomMat(12, 28))
           || test_sign(RandomMat(18, 24))
           || test_sign(RandomMat(17, 12));
}

static int test_sign_3()
{
    return 0
           || test_sign(RandomMat(128))
           || test_sign(RandomMat(256))
           || test_sign(RandomMat(512));
}

int main()
{
    SRAND(7767517);

    return 0
           || test_sign_0()
           || test_sign_1()
           || test_sign_2()
           || test_sign_3();
}
