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

#include "layer/padding.h"
#include "testutil.h"

static int test_padding(const ncnn::Mat& a, int top, int bottom, int left, int right, int front, int behind, int type, float value, int per_channel_pad_data_size)
{
    ncnn::ParamDict pd;
    pd.set(0, top);
    pd.set(1, bottom);
    pd.set(2, left);
    pd.set(3, right);
    pd.set(4, type);
    pd.set(5, value);
    pd.set(6, per_channel_pad_data_size);
    pd.set(7, front);
    pd.set(8, behind);

    std::vector<ncnn::Mat> weights(per_channel_pad_data_size ? 1 : 0);
    if (per_channel_pad_data_size)
        weights[0] = RandomMat(per_channel_pad_data_size);

    int ret = test_layer<ncnn::Padding>("Padding", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_padding failed a.dims=%d a=(%d %d %d) top=%d bottom=%d left=%d right=%d front=%d behind=%d type=%d value=%f per_channel_pad_data_size=%d\n", a.dims, a.w, a.h, a.c, top, bottom, left, right, front, behind, type, value, per_channel_pad_data_size);
    }

    return ret;
}

static int test_padding_0(const ncnn::Mat& a)
{
    return 0
           || test_padding(a, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test_padding(a, 0, 0, 1, 1, 0, 0, 0, 3.f, 0)
           || test_padding(a, 0, 0, 2, 2, 0, 0, 0, 0.f, 0)
           || test_padding(a, 0, 0, 12, 2, 0, 0, 0, 0.f, 0)
           || test_padding(a, 0, 0, 24, 12, 0, 0, 0, 2.f, 0)
           || test_padding(a, 0, 0, 20, 4, 0, 0, 0, 6.1f, 0)
           || test_padding(a, 0, 0, 11, 5, 0, 0, 0, -0.5f, 0)

           || test_padding(a, 0, 0, 1, 1, 0, 0, 1, 0.f, 0)
           || test_padding(a, 0, 0, 2, 2, 0, 0, 1, 0.f, 0)
           || test_padding(a, 0, 0, 12, 2, 0, 0, 1, 0.f, 0)
           || test_padding(a, 0, 0, 24, 12, 0, 0, 1, 0.f, 0)
           || test_padding(a, 0, 0, 20, 4, 0, 0, 1, 0.f, 0)
           || test_padding(a, 0, 0, 11, 5, 0, 0, 1, 0.f, 0)

           || test_padding(a, 0, 0, 1, 1, 0, 0, 2, 0.f, 0)
           || test_padding(a, 0, 0, 2, 2, 0, 0, 2, 0.f, 0)
           || test_padding(a, 0, 0, 12, 2, 0, 0, 2, 0.f, 0)
           || test_padding(a, 0, 0, 24, 12, 0, 0, 2, 0.f, 0)
           || test_padding(a, 0, 0, 20, 4, 0, 0, 2, 0.f, 0)
           || test_padding(a, 0, 0, 11, 5, 0, 0, 2, 0.f, 0);
}

static int test_padding_1(const ncnn::Mat& a)
{
    return 0
           || test_padding(a, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test_padding(a, 1, 1, 1, 1, 0, 0, 0, 3.f, 0)
           || test_padding(a, 2, 2, 2, 2, 0, 0, 0, 0.f, 0)
           || test_padding(a, 12, 2, 12, 2, 0, 0, 0, 0.f, 0)
           || test_padding(a, 24, 12, 8, 12, 0, 0, 0, 2.f, 0)
           || test_padding(a, 20, 4, 13, 4, 0, 0, 0, 6.1f, 0)
           || test_padding(a, 11, 5, 11, 5, 0, 0, 0, -0.5f, 0)

           || test_padding(a, 1, 1, 1, 1, 0, 0, 1, 0.f, 0)
           || test_padding(a, 2, 2, 2, 2, 0, 0, 1, 0.f, 0)
           || test_padding(a, 12, 2, 12, 2, 0, 0, 1, 0.f, 0)
           || test_padding(a, 24, 12, 8, 12, 0, 0, 1, 0.f, 0)
           || test_padding(a, 20, 4, 13, 4, 0, 0, 1, 0.f, 0)
           || test_padding(a, 11, 5, 11, 5, 0, 0, 1, 0.f, 0)

           || test_padding(a, 1, 1, 1, 1, 0, 0, 2, 0.f, 0)
           || test_padding(a, 2, 2, 2, 2, 0, 0, 2, 0.f, 0)
           || test_padding(a, 12, 2, 12, 2, 0, 0, 2, 0.f, 0)
           || test_padding(a, 24, 12, 8, 12, 0, 0, 2, 0.f, 0)
           || test_padding(a, 20, 4, 13, 4, 0, 0, 2, 0.f, 0)
           || test_padding(a, 11, 5, 11, 5, 0, 0, 2, 0.f, 0);
}

static int test_padding_2(const ncnn::Mat& a)
{
    return 0
           || test_padding(a, 3, 4, 5, 6, 0, 0, 0, 1.f, 0)
           || test_padding(a, 3, 4, 5, 6, 0, 0, 0, 0.f, a.c)
           || test_padding(a, 3, 4, 5, 6, 0, 0, 1, 0.f, 0)
           || test_padding(a, 3, 4, 5, 6, 0, 0, 2, 0.f, 0)

           || test_padding(a, 0, 0, 0, 0, 0, 0, 0, 0.f, 0)
           || test_padding(a, 1, 1, 1, 1, 1, 1, 0, 3.f, 0)
           || test_padding(a, 2, 2, 2, 2, 2, 2, 0, 0.f, a.c + 4)
           || test_padding(a, 12, 2, 12, 2, 12, 2, 0, 0.f, 0)
           || test_padding(a, 8, 12, 16, 12, 24, 12, 0, 2.f, 0)
           || test_padding(a, 6, 7, 13, 4, 20, 4, 0, 6.1f, 0)
           || test_padding(a, 11, 5, 11, 5, 11, 5, 0, -0.5f, 0)

           || test_padding(a, 1, 1, 1, 1, 1, 1, 1, 0.f, 0)
           || test_padding(a, 2, 2, 2, 2, 2, 2, 1, 0.f, 0)
           || test_padding(a, 12, 2, 12, 2, 12, 2, 1, 0.f, 0)
           || test_padding(a, 8, 12, 16, 12, 24, 12, 1, 0.f, 0)
           || test_padding(a, 6, 7, 13, 4, 20, 4, 1, 0.f, 0)
           || test_padding(a, 11, 5, 11, 5, 11, 5, 1, 0.f, 0)

           || test_padding(a, 1, 1, 1, 1, 1, 1, 2, 0.f, 0)
           || test_padding(a, 2, 2, 2, 2, 2, 2, 2, 0.f, 0)
           || test_padding(a, 12, 2, 12, 2, 12, 2, 2, 0.f, 0)
           || test_padding(a, 8, 12, 16, 12, 24, 12, 2, 0.f, 0)
           || test_padding(a, 6, 7, 13, 4, 20, 4, 2, 0.f, 0)
           || test_padding(a, 11, 5, 11, 5, 11, 5, 2, 0.f, 0);
}

int main()
{
    SRAND(7767517);

    return 0
           || test_padding_0(RandomMat(128))
           || test_padding_0(RandomMat(124))
           || test_padding_0(RandomMat(127))
           || test_padding_1(RandomMat(64, 64))
           || test_padding_1(RandomMat(60, 60))
           || test_padding_1(RandomMat(63, 63))
           || test_padding_2(RandomMat(64, 64, 64))
           || test_padding_2(RandomMat(60, 60, 60))
           || test_padding_2(RandomMat(63, 63, 63));
}
