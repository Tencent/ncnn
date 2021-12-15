// Xavier Hsinyuan is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 Xavier Hsinyuan <me@lstlx.com>. All rights reserved.
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

#include "layer/selu.h"
#include "testutil.h"

static int test_selu(const ncnn::Mat& a, float alpha, float lambda)
{
    ncnn::ParamDict pd;
    pd.set(0, alpha);
    pd.set(1, lambda);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer<ncnn::SELU>("SELU", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_selu failed a.dims=%d a=(%d %d %d) alpha=%f lambda=%f\n", a.dims, a.w, a.h, a.c, alpha, lambda);
    }

    return ret;
}

static int test_selu_0()
{
    return 0
           || test_selu(RandomMat(5, 7, 24), 1.673264f, 1.050700f)
           || test_selu(RandomMat(7, 9, 12), 1.673264f, 1.050700f)
           || test_selu(RandomMat(3, 5, 13), 1.673264f, 1.050700f);
}

static int test_selu_1()
{
    return 0
           || test_selu(RandomMat(15, 24), 1.673264f, 1.050700f)
           || test_selu(RandomMat(17, 12), 1.673264f, 1.050700f)
           || test_selu(RandomMat(19, 15), 1.673264f, 1.050700f);
}

static int test_selu_2()
{
    return 0
           || test_selu(RandomMat(128), 1.673264f, 1.050700f)
           || test_selu(RandomMat(124), 1.673264f, 1.050700f)
           || test_selu(RandomMat(127), 1.673264f, 1.050700f);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_selu_0()
           || test_selu_1()
           || test_selu_2();
}
