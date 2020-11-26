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

static int test_padding(int w, int h, int c, int top, int bottom, int left, int right, int type, float value, int per_channel_pad_data_size, int front, int behind)
{
    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, top);    // top
    pd.set(1, bottom); // bottom
    pd.set(2, left);   // left
    pd.set(3, right);  // right
    pd.set(4, type);   // type
    pd.set(5, value);  // value
    if (per_channel_pad_data_size != 0)
    {
        per_channel_pad_data_size = c + front + behind;
    }
    pd.set(6, per_channel_pad_data_size); // per_channel_pad_data_size
    pd.set(7, front);
    pd.set(8, behind);

    std::vector<ncnn::Mat> weights(per_channel_pad_data_size ? 1 : 0);
    if (per_channel_pad_data_size)
        weights[0] = RandomMat(per_channel_pad_data_size);

    int ret = test_layer<ncnn::Padding>("Padding", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_padding failed w=%d h=%d c=%d top=%d bottom=%d left=%d right=%d front=%d behind=%d type=%d value=%f per_channel_pad_data_size=%d\n", w, h, c, top, bottom, left, right, front, behind, type, value, per_channel_pad_data_size);
    }

    return ret;
}

static int test_padding_0()
{
    return 0
           || test_padding(5, 7, 24, 1, 1, 1, 1, 0, 1.f, 0, 0, 0)
           || test_padding(5, 7, 24, 2, 3, 4, 5, 0, 0.f, 24, 0, 0)
           || test_padding(5, 7, 24, 4, 0, 0, 4, 1, 0.f, 0, 0, 0)
           || test_padding(5, 7, 24, 0, 3, 0, 3, 2, 0.f, 0, 0, 0)
           || test_padding(7, 9, 12, 2, 2, 2, 2, 0, 2.f, 0, 0, 0)
           || test_padding(7, 9, 12, 3, 4, 5, 6, 0, 0.f, 12, 0, 0)
           || test_padding(7, 9, 12, 1, 5, 5, 1, 1, 0.f, 0, 0, 0)
           || test_padding(7, 9, 12, 4, 2, 4, 1, 2, 0.f, 0, 0, 0)
           || test_padding(7, 5, 13, 3, 3, 3, 3, 0, 3.f, 0, 0, 0)
           || test_padding(7, 5, 13, 0, 1, 2, 0, 0, 0.f, 13, 0, 0)
           || test_padding(7, 5, 13, 4, 3, 4, 3, 1, 0.f, 0, 0, 0)
           || test_padding(7, 5, 13, 4, 4, 4, 3, 2, 0.f, 0, 0, 0);
}

static int test_padding_1()
{
    return 0
           || test_padding(5, 7, 16, 1, 1, 1, 1, 0, 0.f, 0, 4, 0)
           || test_padding(5, 7, 16, 4, 3, 4, 3, 1, 0.f, 0, 4, 0)
           || test_padding(5, 7, 16, 4, 3, 4, 3, 2, 0.f, 0, 4, 0)
           || test_padding(5, 7, 16, 1, 1, 1, 1, 0, 0.f, 0, 0, 4)
           || test_padding(5, 7, 17, 2, 3, 2, 3, 1, 0.f, 0, 4, 9)
           || test_padding(5, 7, 12, 4, 3, 4, 3, 0, 1.f, 0, 4, 8)
           || test_padding(5, 7, 16, 4, 3, 4, 3, 2, 0.f, 0, 4, 8)
           || test_padding(5, 7, 16, 4, 3, 4, 3, 0, 0.f, 16, 4, 2)
           || test_padding(5, 7, 8, 2, 3, 2, 3, 1, 0.f, 22, 8, 0)
           || test_padding(5, 7, 6, 4, 3, 4, 3, 2, 0.f, 0, 2, 4)
           || test_padding(5, 7, 7, 0, 1, 0, 1, 0, 233.f, 0, 3, 1)
           || test_padding(5, 7, 3, 2, 1, 2, 1, 0, 0.f, 3, 0, 4)
           || test_padding(5, 7, 16, 2, 1, 2, 1, 2, 0.f, 3, 4, 4)
           || test_padding(5, 7, 16, 2, 1, 2, 1, 0, 0.f, 3, 8, 8)
           || test_padding(5, 7, 16, 2, 1, 2, 1, 1, 0.f, 3, 8, 8)
           || test_padding(5, 7, 16, 2, 1, 2, 1, 2, 0.f, 3, 8, 8);
}

int main()
{
    SRAND(7767517);

    return test_padding_0() || test_padding_1();
}
