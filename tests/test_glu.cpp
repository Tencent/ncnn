// Copyright (c) 2022 Xiaomi Corp.        (author: Fangjun Kuang)
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

static int test_glu(const ncnn::Mat& a, int axis)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("GLU", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_glu failed a.dims=%d a=(%d %d %d) axis=%d\n", a.dims, a.w, a.h, a.c, axis);
    }

    return ret;
}

static int test_glu_0()
{
    return 0
           || test_glu(RandomMat(6, 7, 24), 0)
           || test_glu(RandomMat(6, 8, 24), 1)
           || test_glu(RandomMat(6, 8, 24), 2)
           || test_glu(RandomMat(36, 7, 22), 0)
           || test_glu(RandomMat(5, 256, 23), -2)
           || test_glu(RandomMat(129, 9, 60), 2)
           || test_glu(RandomMat(129, 9, 30), -1);
}

static int test_glu_1()
{
    return 0
           || test_glu(RandomMat(10, 24), 0)
           || test_glu(RandomMat(7, 24), 1)
           || test_glu(RandomMat(128, 22), 0)
           || test_glu(RandomMat(128, 256), 1);
}

static int test_glu_2()
{
    return 0
           || test_glu(RandomMat(10), 0)
           || test_glu(RandomMat(20), 0)
           || test_glu(RandomMat(128), 0);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_glu_0()
           || test_glu_1()
           || test_glu_2();
}
