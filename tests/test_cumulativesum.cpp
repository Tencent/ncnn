// Copyright (c) 2023 Xiaomi Corp.        (author: Fangjun Kuang)
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

static int test_cumulativesum(const ncnn::Mat& a, int axis)
{
    ncnn::ParamDict pd;
    pd.set(0, axis);

    std::vector<ncnn::Mat> weights(0);

    int ret = test_layer("CumulativeSum", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_cumulativesum failed a.dims=%d a=(%d %d %d) axis=%d\n", a.dims, a.w, a.h, a.c, axis);
    }

    return ret;
}

static int test_cumulativesum_1d()
{
    return 0
           || test_cumulativesum(RandomMat(6), 0)
           || test_cumulativesum(RandomMat(10), 0)
           || test_cumulativesum(RandomMat(10), -1)
           || test_cumulativesum(RandomMat(10), -2)
           || test_cumulativesum(RandomMat(101), 0);
}

static int test_cumulativesum_2d()
{
    return 0
           || test_cumulativesum(RandomMat(6, 8), 0)
           || test_cumulativesum(RandomMat(20, 103), 1)
           || test_cumulativesum(RandomMat(106, 50), -1)
           || test_cumulativesum(RandomMat(106, 50), -2);
}

static int test_cumulativesum_3d()
{
    return 0
           || test_cumulativesum(RandomMat(10, 6, 8), 0)
           || test_cumulativesum(RandomMat(303, 20, 103), 1)
           || test_cumulativesum(RandomMat(106, 50, 99), 2)
           || test_cumulativesum(RandomMat(303, 200, 103), -1)
           || test_cumulativesum(RandomMat(303, 200, 103), -2)
           || test_cumulativesum(RandomMat(303, 200, 103), -2);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_cumulativesum_1d()
           || test_cumulativesum_2d()
           || test_cumulativesum_3d();
}
