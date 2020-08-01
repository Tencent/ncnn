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

#include "layer/dropout.h"
#include "testutil.h"

static int test_dropout(const ncnn::Mat& a, float scale)
{
    ncnn::ParamDict pd;
    pd.set(0, scale);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::Dropout>("Dropout", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_dropout failed a.dims=%d a=(%d %d %d) scale=%f\n", a.dims, a.w, a.h, a.c, scale);
    }

    return ret;
}

static int test_dropout_0()
{
    return 0
           || test_dropout(RandomMat(5, 7, 24), 1.f)
           || test_dropout(RandomMat(5, 7, 24), 0.2f)
           || test_dropout(RandomMat(7, 9, 12), 1.f)
           || test_dropout(RandomMat(7, 9, 12), 0.3f)
           || test_dropout(RandomMat(3, 5, 13), 1.f)
           || test_dropout(RandomMat(3, 5, 13), 0.5f);
}

static int test_dropout_1()
{
    return 0
           || test_dropout(RandomMat(15, 24), 1.f)
           || test_dropout(RandomMat(15, 24), 0.6f)
           || test_dropout(RandomMat(19, 12), 1.f)
           || test_dropout(RandomMat(19, 12), 0.4f)
           || test_dropout(RandomMat(17, 15), 1.f)
           || test_dropout(RandomMat(17, 15), 0.7f);
}

static int test_dropout_2()
{
    return 0
           || test_dropout(RandomMat(128), 1.f)
           || test_dropout(RandomMat(128), 0.4f)
           || test_dropout(RandomMat(124), 1.f)
           || test_dropout(RandomMat(124), 0.1f)
           || test_dropout(RandomMat(127), 1.f)
           || test_dropout(RandomMat(127), 0.5f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_dropout_0()
           || test_dropout_1()
           || test_dropout_2();
}
