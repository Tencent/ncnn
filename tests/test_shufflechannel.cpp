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

#include "layer/shufflechannel.h"

static int test_shufflechannel(int w, int h, int c, int group)
{
    ncnn::Mat a = RandomMat(w, h, c);

    ncnn::ParamDict pd;
    pd.set(0, group);// group

    std::vector<ncnn::Mat> weights(0);

    ncnn::Option opt;
    opt.num_threads = 1;
    opt.use_vulkan_compute = true;
    opt.use_int8_inference = false;

    int ret = test_layer<ncnn::ShuffleChannel>("ShuffleChannel", pd, weights, opt, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_shufflechannel failed w=%d h=%d c=%d group=%d\n", w, h, c, group);
    }

    return ret;
}

static int test_shufflechannel_0()
{
    return 0
        || test_shufflechannel(3, 7, 1, 1)
        || test_shufflechannel(3, 7, 2, 2)
        || test_shufflechannel(3, 7, 3, 3)
        || test_shufflechannel(3, 7, 4, 2)
        || test_shufflechannel(3, 7, 12, 3)
        || test_shufflechannel(3, 7, 12, 4)
        || test_shufflechannel(3, 7, 12, 6)
        || test_shufflechannel(3, 7, 15, 3)
        || test_shufflechannel(3, 7, 15, 5)
        || test_shufflechannel(3, 7, 16, 2)
        || test_shufflechannel(3, 7, 16, 4)
        || test_shufflechannel(3, 7, 16, 8)
        ;
}

int main()
{
    SRAND(7767517);

    return test_shufflechannel_0();
}
