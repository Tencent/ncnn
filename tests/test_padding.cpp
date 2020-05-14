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

#include "testutil.h"

#include "layer/padding.h"

static int test_padding(int w, int h, int c, int top, int bottom, int left, int right, int type, float value, int per_channel_pad_data_size)
{
    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, top);// top
    pd.set(1, bottom);// bottom
    pd.set(2, left);// left
    pd.set(3, right);// right
    pd.set(4, type);// type
    pd.set(5, value);// value
    pd.set(6, per_channel_pad_data_size);// per_channel_pad_data_size

    std::vector<ncnn::Mat> weights(per_channel_pad_data_size ? 1 : 0);
    if (per_channel_pad_data_size)
        weights[0] = RandomMat(per_channel_pad_data_size);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;

    int ret = test_layer<ncnn::Padding>("Padding", pd, weights, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_padding failed w=%d h=%d c=%d top=%d bottom=%d left=%d right=%d type=%d value=%f per_channel_pad_data_size=%d\n", w, h, c, top, bottom, left, right, type, value, per_channel_pad_data_size);
    }

    return ret;
}

static int test_padding_0()
{
    return 0
        || test_padding(5, 7, 16, 1, 1, 1, 1, 0, 0.f, 0)
        || test_padding(5, 7, 16, 2, 3, 2, 3, 1, 0.f, 0)
        || test_padding(5, 7, 16, 4, 3, 4, 3, 2, 0.f, 0)
        || test_padding(5, 7, 16, 4, 3, 4, 3, 0, 0.f, 16)
        || test_padding(5, 7, 5, 2, 3, 2, 3, 1, 0.f, 0)
        || test_padding(5, 7, 6, 4, 3, 4, 3, 2, 0.f, 0)
        || test_padding(5, 7, 7, 0, 1, 0, 1, 0, 233.f, 0)
        || test_padding(5, 7, 3, 2, 1, 2, 1, 0, 0.f, 3)
        ;
}

int main()
{
    SRAND(7767517);

    return test_padding_0();
}
