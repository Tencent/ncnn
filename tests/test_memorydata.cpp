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

#include "layer/memorydata.h"
#include "testutil.h"

static int test_memorydata(const ncnn::Mat& a)
{
    ncnn::ParamDict pd;
    pd.set(0, a.w);
    pd.set(1, a.h);
    pd.set(2, a.c);

    std::vector<ncnn::Mat> weights(1);
    weights[0] = a;

    std::vector<ncnn::Mat> as(0);

    int ret = test_layer<ncnn::MemoryData>("MemoryData", pd, weights, as, 1);
    if (ret != 0)
    {
        fprintf(stderr, "test_memorydata failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
    }

    return ret;
}

static int test_memorydata_0()
{
    return 0
           || test_memorydata(RandomMat(5, 7, 16))
           || test_memorydata(RandomMat(3, 5, 13));
}

static int test_memorydata_1()
{
    return 0
           || test_memorydata(RandomMat(6, 16))
           || test_memorydata(RandomMat(7, 15));
}

static int test_memorydata_2()
{
    return 0
           || test_memorydata(RandomMat(128))
           || test_memorydata(RandomMat(127));
}

int main()
{
    SRAND(7767517);

    return 0
           || test_memorydata_0()
           || test_memorydata_1()
           || test_memorydata_2();
}
